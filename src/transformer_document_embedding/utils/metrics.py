from __future__ import annotations
from abc import abstractmethod
from dataclasses import asdict, dataclass
import itertools
import logging
from time import time

import warnings
from typing import TYPE_CHECKING, Any, Callable
from sklearn.cross_decomposition import CCA
from cca_zoo.linear import CCA as ZooCCA


import numpy as np
import torch
from torcheval.metrics import Max, Mean, Metric
from torcheval.metrics.toolkit import clone_metric
from transformer_document_embedding.datasets import col

from transformer_document_embedding.utils.cca_losses import CCALoss

if TYPE_CHECKING:
    from typing import Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetric:
    """Wrapped `torcheval` Metric with information how to handle it during training."""

    @staticmethod
    def identity_update_fn(metric: Metric, *args: Any) -> None:
        metric.update(*args)

    name: str
    metric: Metric
    log_frequency: Optional[int] = None
    update_fn: Callable = identity_update_fn
    reset_after_log: bool = True

    @property
    def device(self) -> torch.device:
        return self.metric.device

    @torch.inference_mode()
    def update(self, *args) -> None:
        self.update_fn(self.metric, *args)

    def clone(self, **kwargs_overwrite) -> TrainingMetric:
        kwargs = asdict(self)
        kwargs["metric"] = clone_metric(self.metric)
        kwargs.update(kwargs_overwrite)
        return TrainingMetric(**kwargs)

    def compute(self) -> Any:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()

    def to(self, device: torch.device) -> TrainingMetric:
        self.metric.to(device)
        return self


class VMemMetric(TrainingMetric):
    def __init__(self, log_frequency: int, reset_after_log: bool = False) -> None:
        super().__init__(
            "used_vmem", Max(), log_frequency, self._update_fn, reset_after_log
        )

    def _update_fn(self, metric: Metric, *_) -> None:
        mem_used = torch.cuda.memory_reserved(metric.device)
        mem_used = mem_used // 1024**2
        metric.update(torch.tensor(mem_used))


class EmbeddingMSEWithCol(TrainingMetric):
    def __init__(
        self,
        base_name: str,
        log_frequency: int,
        col_name: str,
        max_input_length: Optional[int] = None,
        normalize: bool = False,
        reset_after_log: bool = False,
    ):
        name_norm_mod = "_norm" if normalize else ""
        name_length_mod = f"_{max_input_length}" if max_input_length is not None else ""

        super().__init__(
            f"{base_name}_mse{name_norm_mod}{name_length_mod}",
            Mean(),
            log_frequency,
            self._update_fn,
            reset_after_log,
        )

        self.col_name = col_name
        self.normalize = normalize
        self.max_input_length = max_input_length

    def _update_fn(
        self,
        metric: Metric,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> None:
        model_embeddings = outputs[col.EMBEDDING]
        col_emebddings = batch[self.col_name]
        if self.normalize:
            model_embeddings /= torch.linalg.vector_norm(model_embeddings)
            col_emebddings /= torch.linalg.vector_norm(col_emebddings)

        # TODO: Mean instead of sum
        mse = (model_embeddings - col_emebddings) ** 2
        mse = mse.sum(dim=1)

        if self.max_input_length is not None:
            mask = batch[col.LENGTH] <= self.max_input_length
            # mask is 1D
            non_masked_idxs = mask.nonzero().squeeze()
            mse = mse.index_select(0, non_masked_idxs)

        metric.update(mse)


class EmbeddingCosineDistanceWithCol(TrainingMetric):
    def __init__(
        self,
        base_name: str,
        log_frequency: int,
        col_name: str,
        max_input_length: Optional[int] = None,
        reset_after_log: bool = False,
    ) -> None:
        name_length_suffix = (
            f"_{max_input_length}" if max_input_length is not None else ""
        )

        super().__init__(
            f"{base_name}_cos_dist{name_length_suffix}",
            Mean(),
            log_frequency,
            self._update_fn,
            reset_after_log,
        )

        self.col_name = col_name
        self.max_input_length = max_input_length

    def _update_fn(
        self,
        metric: Metric,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> None:
        cos_dist = 1 - torch.nn.functional.cosine_similarity(
            outputs[col.EMBEDDING],
            batch[self.col_name],
            dim=1,
        )
        if self.max_input_length is not None:
            mask = batch[col.LENGTH] <= self.max_input_length
            # mask is 1D
            non_masked_idxs = mask.nonzero().squeeze()
            cos_dist = cos_dist.index_select(0, non_masked_idxs)
        metric.update(cos_dist)


class WindowedMetric(Metric):
    """Base class for all metrics that need fixed window in order to be comparable."""

    def __init__(
        self, window_size: int, num_views: int, device: Optional[torch.device] = None
    ) -> None:
        super().__init__(device=device)

        self._add_state(
            "views", [torch.tensor([], device=device) for _ in range(num_views)]
        )
        self.views: list[torch.Tensor]

        self._window_size = window_size

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def has_full_window(self) -> int:
        samples = self.views[0].size(0)
        return samples == self.window_size

    @torch.inference_mode()
    def update(self, *new_views: torch.Tensor) -> WindowedMetric:
        for i, new_view in enumerate(new_views):
            self.views[i] = torch.cat((self.views[i], new_view), dim=0)

        self._shorten_views_to_size()
        return self

    def compute(self) -> Any:
        if not self.has_full_window:
            return torch.nan

        return self._compute_with_full_window()

    @abstractmethod
    def _compute_with_full_window(self) -> Any:
        pass

    def merge_state(self, metrics: Iterable[WindowedCCAMetric]) -> WindowedMetric:
        views = [[] for _ in range(len(self.views))]
        for metric in itertools.chain([self], metrics):
            assert len(metric.views) == len(views)
            for i, view in enumerate(metric.views):
                views[i].append(view)

        self.views = [torch.concat(view) for view in views]

        self._shorten_views_to_size()
        return self

    def _shorten_views_to_size(self) -> None:
        if self.views[0].size(0) > self.window_size:
            self.views = [view[-self.window_size :] for view in self.views]


class WindowedCCAMetric(WindowedMetric):
    """CCA computed by sklearn.

    It has a fixed window because the size of window influences the result. By
    having fixed window we make sure that the values are always informative and
    comparable.
    """

    # How many n_components will fit in window
    WINDOW_MULT_FACTOR = 5

    def __init__(
        self,
        n_components: int,
        window_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            window_size=window_size
            if window_size is not None
            else self.WINDOW_MULT_FACTOR * n_components,
            num_views=2,
            device=device,
        )

        self.n_components = n_components

    @torch.inference_mode()
    def _compute_with_full_window(self) -> float:
        samples = self.views[0].size(0)
        view1_dim = self.views[0].size(1)
        view2_dim = self.views[1].size(1)

        # There is a upper bound on the number of dimensions found
        if self.n_components > min(view1_dim, view2_dim, samples):
            return torch.nan

        start_time = time()
        cca = CCA(n_components=self.n_components, tol=1e-2, max_iter=100000)
        try:
            # No category keyword for python 3.10 compatibility
            with warnings.catch_warnings():
                views1_, views2_ = cca.fit_transform(
                    self.views[0].numpy(force=True),
                    self.views[1].numpy(force=True),
                )

            correlation = (
                np.corrcoef(views1_, views2_, rowvar=False)
                .diagonal(offset=self.n_components)
                .sum()
            )
            end_time = time()
            logger.info(
                "CCA: dim %d, window size %d, seconds: %f",
                self.n_components,
                self.window_size,
                end_time - start_time,
            )

            return correlation
        except np.linalg.LinAlgError as e:
            logger.warn("Error when computing CCA: %s", e)
            end_time = time()
            logger.info(
                "CCA: dim %d, window size %d, seconds: %f",
                self.n_components,
                self.window_size,
                end_time - start_time,
            )
            return np.nan


class WindowedCCAMetricTorch(WindowedCCAMetric):
    def __init__(
        self,
        n_components: int,
        window_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(n_components, window_size, device)

        self._cca_loss = CCALoss(
            output_dimension=n_components,
            regularization_constant=0,  # For much better accuracy
        )

    @torch.inference_mode()
    def _compute_with_full_window(self) -> float:
        samples = self.views[0].size(0)
        view1_dim = self.views[0].size(1)
        view2_dim = self.views[1].size(1)

        # There is a upper bound on the number of dimensions found
        if self.n_components > min(view1_dim, view2_dim, samples):
            return torch.nan

        return -self._cca_loss(self.views[0], self.views[1])["loss"]


class WindowedCCAMetricZoo(WindowedCCAMetric):
    @torch.inference_mode()
    def _compute_with_full_window(self) -> float:
        samples = self.views[0].size(0)
        view1_dim = self.views[0].size(1)
        view2_dim = self.views[1].size(1)

        # There is a upper bound on the number of dimensions found
        if self.n_components > min(view1_dim, view2_dim, samples):
            problematic_vars = []
            for var_name, val in zip(
                ["view1 dimension", "view2 dimension", "number of samples"],
                [view1_dim, view2_dim, samples],
                strict=True,
            ):
                if val < self.n_components:
                    problematic_vars.append(f"{var_name} ({val})")
            logger.warn(
                f"CCA computed with {','.join(problematic_vars)} smaller than "
                f"n_components ({self.n_components})"
            )
            return torch.nan

        views = (self.views[0].numpy(force=True), self.views[1].numpy(force=True))
        cca_model = ZooCCA(latent_dimensions=self.n_components)
        return cca_model.fit(views).score(views) / self.n_components


class WindowedAbsCrossCorrelationMetric(WindowedMetric):
    def __init__(self, window_size: int, device: Optional[torch.device] = None) -> None:
        super().__init__(window_size=window_size, num_views=2, device=device)

    def _compute_with_full_window(self) -> float:
        all_vars = torch.concat((self.views[0].T, self.views[1].T), dim=0)
        view1_dim = self.views[0].size(1)

        cross_corr_coefs = torch.corrcoef(all_vars)[:view1_dim, view1_dim:]
        mean_abs_cross_corr = cross_corr_coefs.abs().mean()

        return mean_abs_cross_corr.item()


class WindowedAbsCorrelationMetric(WindowedMetric):
    def __init__(self, window_size: int, device: Optional[torch.device] = None) -> None:
        super().__init__(window_size, num_views=1, device=device)

    def _compute_with_full_window(self) -> Any:
        abs_corr_mat = torch.corrcoef(self.views[0].T).abs()
        abs_corr_without_diagonal = (
            abs_corr_mat
            - torch.eye(self.views[0].size(1), device=self.views[0].device)
            * abs_corr_mat.diagonal()
        )

        return abs_corr_without_diagonal.mean()
