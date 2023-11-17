from __future__ import annotations
import logging

import warnings
from typing import TYPE_CHECKING, Any, Callable
from sklearn.cross_decomposition import CCA

from sklearn.exceptions import ConvergenceWarning

import numpy as np
import torch
from torcheval.metrics import Mean, Metric

if TYPE_CHECKING:
    from typing import Iterable, Optional, Union

logger = logging.getLogger(__name__)


class MeanLossMetric(Metric):
    """Metric accumulating the mean loss."""

    def __init__(self, loss_fn: torch.nn.Module, **kwargs) -> None:
        super().__init__(**kwargs)
        self._loss_fn = loss_fn
        self._mean_loss = Mean(device=self.device)

    def to(
        self, device: Union[str, torch.device], *args: Any, **kwargs: Any
    ) -> MeanLossMetric:
        self._loss_fn.to(device)
        self._mean_loss = self._mean_loss.to(device, *args, **kwargs)

        self._device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        return self

    @torch.inference_mode()
    def update(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        loss = self._loss_fn(outputs, labels)
        self._mean_loss.update(loss)

    @torch.inference_mode()
    def compute(self) -> torch.Tensor:
        return self._mean_loss.compute()

    def reset(self) -> None:
        self._mean_loss.reset()

    @torch.inference_mode()
    def merge_state(self, metrics: Iterable[MeanLossMetric]) -> MeanLossMetric:
        for metric in metrics:
            self._mean_loss.update(metric._mean_loss.compute())

        return self

    def state_dict(self) -> dict[str, Any]:
        return {
            "loss_fn": self._loss_fn.state_dict(),
            "mean_loss": self._mean_loss.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        self._loss_fn.load_state_dict(state_dict["loss_fn"], strict)
        self._mean_loss.load_state_dict(state_dict["mean_loss"], strict)


class VMemMetric(Metric):
    """Metric outputting current amount of video memory used by pytorch."""

    def __init__(
        self,
        return_MB: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._return_MB = return_MB  # pylint: disable=invalid-name

    def update(self, *args, **kwargs) -> None:
        pass

    @torch.inference_mode()
    def compute(self) -> torch.Tensor:
        mem_used = torch.cuda.memory_reserved(self._device)
        mem_used = mem_used // 1024**2 if self._return_MB else mem_used
        return torch.tensor(mem_used)

    def merge_state(self, _: Iterable[VMemMetric]) -> VMemMetric:
        return self


UpdateWrapperFn = Callable[
    [Metric, dict[str, torch.Tensor], dict[str, torch.Tensor]], Any
]


class AccessorMetric(Metric):
    def __init__(
        self, inner_metric: Metric, update_fn: UpdateWrapperFn, **kwargs
    ) -> None:
        self._inner_metric = inner_metric
        self._update_fn = update_fn
        super().__init__(**kwargs)

    @torch.inference_mode()
    def update(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> AccessorMetric:
        self._update_fn(self._inner_metric, outputs, batch)
        return self

    def merge_state(self, metrics: Iterable[Metric]) -> AccessorMetric:
        self._inner_metric.merge_state(metrics)
        return self

    def state_dict(self) -> dict[str, Any]:
        return self._inner_metric.state_dict()

    def compute(self) -> Any:
        return self._inner_metric.compute()

    def to(
        self, device: Union[str, torch.device], *args: Any, **kwargs: Any
    ) -> AccessorMetric:
        self._inner_metric.to(device, *args, **kwargs)
        return self

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        self._inner_metric.load_state_dict(state_dict, strict)

    def reset(self) -> AccessorMetric:
        self._inner_metric.reset()
        return self


def with_accessor(metric: Metric, update_fn: UpdateWrapperFn) -> Metric:
    return AccessorMetric(metric, update_fn)


class MSEWithSBERT(AccessorMetric):
    def __init__(
        self, max_input_length: Optional[int], normalize: bool = False, **kwargs
    ) -> None:
        self.max_input_length = max_input_length
        self.normalize = normalize

        super().__init__(Mean(), self._accessor, **kwargs)

    def _accessor(
        self,
        metric: Metric,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> None:
        model_embeddings = outputs["pooler_output"]
        sbert_emebddings = batch["sbert"]
        if self.normalize:
            model_embeddings /= torch.linalg.vector_norm(model_embeddings)
            sbert_emebddings /= torch.linalg.vector_norm(sbert_emebddings)

        mse = (model_embeddings - sbert_emebddings) ** 2
        mse = mse.sum(dim=1)

        if self.max_input_length is not None:
            mask = batch["length"] <= self.max_input_length
            # mask is 1D
            non_masked_idxs = mask.nonzero().squeeze()
            mse = mse.index_select(0, non_masked_idxs)

        metric.update(mse)


class CosineDistanceWithSBERT(AccessorMetric):
    def __init__(self, max_input_length: Optional[int], **kwargs) -> None:
        self.max_input_length = max_input_length
        super().__init__(Mean(), self._accessor, **kwargs)

    def _accessor(
        self,
        metric: Metric,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> None:
        cos_dist = 1 - torch.nn.functional.cosine_similarity(
            outputs["pooler_output"],
            batch["sbert"],
            dim=1,
        )
        if self.max_input_length is not None:
            mask = batch["length"] <= self.max_input_length
            # mask is 1D
            non_masked_idxs = mask.nonzero().squeeze()
            cos_dist = cos_dist.index_select(0, non_masked_idxs)
        metric.update(cos_dist)


class PositivesMaskPercentage(Metric):
    def __init__(self, *, device: Optional[torch.device] = None) -> None:
        super().__init__(device=device)

        self._add_state("positives", torch.tensor(0, device=device, dtype=torch.int32))
        self.positives: torch.Tensor

        self._add_state("totals", torch.tensor(0, device=device, dtype=torch.int32))
        self.totals: torch.Tensor

    @torch.inference_mode()
    def update(self, binary_mask: torch.Tensor) -> PositivesMaskPercentage:
        self.totals += binary_mask.size(0)
        self.positives += binary_mask.sum()
        return self

    @torch.inference_mode()
    def compute(self) -> torch.Tensor:
        return self.positives / self.totals

    def merge_state(
        self, metrics: Iterable[PositivesMaskPercentage]
    ) -> PositivesMaskPercentage:
        for other in metrics:
            self.totals += other.totals
            self.positives += other.positives

        return self


class PercentLengthBelowThres(AccessorMetric):
    def __init__(self, length_threshold: int, **kwargs) -> None:
        super().__init__(PositivesMaskPercentage(), self._accessor, **kwargs)
        self._len_thres = length_threshold

    def _accessor(
        self, metric: Metric, _: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> None:
        metric.update(batch["length"] < self._len_thres)


class WindowedNonResetableCCAMetric(Metric):
    """CCA computed by sklearn. The metric does not reset and has fixed window.

    Does not reset because it is assumed that to have enough elements to
    compute CCA this metric will need much more updates than regular metric.
    Ergo if this metric reset it would never output any values.

    It has a fixed window because the size of window influences the result. By
    having fixed window we make sure that the values are always informative and
    comparable.

    """

    # How many n_components will fit in window
    WINDOW_MULT_FACTOR = 3

    def __init__(
        self,
        n_components: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(device=device)

        self._add_state("views1", torch.tensor([], device=device))
        self.views1: torch.Tensor
        self._add_state("views2", torch.tensor([], device=device))
        self.views2: torch.Tensor

        self._window_size = self.WINDOW_MULT_FACTOR * n_components

        self.n_components = n_components

    @property
    def window_size(self) -> int:
        return self._window_size

    @torch.inference_mode()
    def update(
        self, views1: torch.Tensor, views2: torch.Tensor
    ) -> WindowedNonResetableCCAMetric:
        self.views1 = torch.cat((self.views1, views1))
        self.views2 = torch.cat((self.views2, views2))

        self._shorten_views_to_size()
        return self

    @torch.inference_mode()
    def compute(self) -> float:
        samples = self.views1.size(0)

        if samples != self.window_size:
            return torch.nan

        view1_dim = self.views1.size(1)
        view2_dim = self.views2.size(1)

        # There is a upper bound on the number of dimensions found
        if self.n_components > min(view1_dim, view2_dim, samples):
            return torch.nan

        cca = CCA(n_components=self.n_components, max_iter=5000)
        try:
            with warnings.catch_warnings(category=ConvergenceWarning):
                views1_, views2_ = cca.fit_transform(
                    self.views1.numpy(force=True),
                    self.views2.numpy(force=True),
                )

            correlation = (
                np.corrcoef(views1_, views2_, rowvar=False)
                .diagonal(offset=self.n_components)
                .sum()
            )
            return correlation
        except np.linalg.LinAlgError as e:
            logger.warn("Error when computing CCA: %s", e)
            return np.nan

    def merge_state(
        self, metrics: Iterable[WindowedNonResetableCCAMetric]
    ) -> WindowedNonResetableCCAMetric:
        views1 = [self.views1]
        views2 = [self.views2]
        for other in metrics:
            views1.append(other.views1)
            views2.append(other.views2)

        self.views1 = torch.cat(views1)
        self.views2 = torch.cat(views2)

        self._shorten_views_to_size()

        return self

    def reset(self) -> WindowedNonResetableCCAMetric:
        return self

    def _shorten_views_to_size(self) -> None:
        if self.views1.size(0) > self.window_size:
            self.views1 = self.views1[-self.window_size :, :]
        if self.views2.size(0) > self.window_size:
            self.views2 = self.views2[-self.window_size :, :]
