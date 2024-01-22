from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING, Any, Iterable
import logging

import torch
from torcheval.metrics import Max, Mean
from tqdm.auto import tqdm


from transformer_document_embedding.models.transformer.base import TransformerBase
from transformer_document_embedding.models.trainer import (
    TorchTrainer,
)
from transformer_document_embedding.tasks.teacher_embedding import TeacherEmbedding
from transformer_document_embedding.utils.metrics import (
    EmbeddingCosineDistanceWithCol,
    EmbeddingMSEWithCol,
    TrainingMetric,
    VMemMetric,
    WindowedAbsCorrelationMetric,
    WindowedCCAMetric,
    WindowedCCAMetricZoo,
    WindowedAbsCrossCorrelationMetric,
)
from transformer_document_embedding.utils.similarity_losses import (
    ContrastiveLoss,
    create_sim_based_loss,
)
import transformer_document_embedding.utils.training as train_utils
from transformer_document_embedding.utils import cca_losses

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from torcheval.metrics import (
        Metric,
    )
    from torch.utils.data import DataLoader
    from datasets import Dataset
    from transformer_document_embedding.tasks.experimental_task import ExperimentalTask
    from typing import Optional
    import numpy as np


logger = logging.getLogger(__name__)


def log_max_abs_grad(
    metric: Metric,
    *_,
    param_name: str,
    model: torch.nn.Module,
) -> None:
    param = model.get_parameter(param_name)
    grad = param.grad
    if grad is None:
        metric.update(torch.tensor([torch.nan], device=param.device))
    else:
        metric.update(grad.abs().max())


class BreadthDepthLoss(torch.nn.Module):
    def __init__(
        self,
        max_depth_length: Optional[int],
        lam: Optional[float] = None,
        breadth_loss: Optional[torch.nn.Module] = None,
        depth_loss: Optional[torch.nn.Module] = None,
        depth_col: str = TeacherEmbedding.DEPTH_COL,
        breadth_col: str = TeacherEmbedding.BREADTH_COL,
        len_col: str = "length",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._depth_col = depth_col
        self._breadth_col = breadth_col
        self._max_depth_length = max_depth_length
        self._len_key = len_col
        self.breadth_loss = breadth_loss
        self.depth_loss = depth_loss
        self._lam = lam

    @property
    def max_depth_length(self) -> Optional[int]:
        return self._max_depth_length

    def forward(
        self, inputs: torch.Tensor, targets: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        outputs = {"loss": torch.tensor(0, device=inputs.device, dtype=inputs.dtype)}

        if self.depth_loss is not None:
            mask = (
                targets[self._len_key] <= self._max_depth_length
                if self.max_depth_length is not None
                else torch.ones_like(targets[self._len_key])
            ).unsqueeze(-1)

            depth_loss = self.depth_loss(inputs, targets[self._depth_col], mask=mask)
            if isinstance(depth_loss, dict):
                just_depth_loss = depth_loss.pop("loss")
                outputs.update(
                    {f"depth_{key}": value for key, value in depth_loss.items()}
                )
                depth_loss = just_depth_loss

            if self._lam is not None:
                depth_loss *= self._lam

            outputs["depth_mask"] = mask
            outputs["depth_loss"] = depth_loss
            outputs["loss"] += depth_loss

        if self.breadth_loss is not None:
            breadth_loss_outputs = self.breadth_loss(inputs, targets[self._breadth_col])
            breadth_loss = torch.mean(breadth_loss_outputs.pop("loss"))

            outputs.update(
                {f"breadth_{key}": value for key, value in breadth_loss_outputs.items()}
            )
            outputs["breadth_loss"] = breadth_loss
            outputs["loss"] += breadth_loss

        return outputs


class _SequenceEmbeddingModel(torch.nn.Module):
    def __init__(
        self,
        transformer: PreTrainedModel,
        pooler: torch.nn.Module,
        loss: BreadthDepthLoss,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.transformer = transformer
        self.pooler = pooler
        self.loss = loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        input_kws = {}
        # Needed for Longformer
        if "global_attention_mask" in kwargs:
            input_kws["global_attention_mask"] = kwargs.pop("global_attention_mask")

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            inputs_embeds=None,
            **input_kws,
        )
        pooled_output = self.pooler(
            **outputs,
            attention_mask=attention_mask,
            **input_kws,
        )
        loss_outputs = self.loss(pooled_output, kwargs) if len(kwargs) > 0 else {}

        return {
            **outputs,
            **loss_outputs,
            "embedding": pooled_output,
        }


class TransformerStudent(TransformerBase):
    def __init__(
        self,
        transformer_model: str,
        batch_size: int,
        pooler_type: str,
        max_depth_length: Optional[int],
        depth_loss_kwargs: Optional[dict[str, Any]],
        breadth_loss_kwargs: Optional[dict[str, Any]],
        transformer_model_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            transformer_model=transformer_model,
            transformer_model_kwargs=transformer_model_kwargs,
            batch_size=batch_size,
            pooler_type=pooler_type,
        )
        loss = self._construct_loss(
            breadth_loss_kwargs,
            depth_loss_kwargs,
            max_depth_length,
            transformer_hidden_size=self._transformer.config.hidden_size,
        )

        self._model: _SequenceEmbeddingModel = _SequenceEmbeddingModel(
            self._transformer, self._pooler, loss
        )

        self.breadth_embed_dim = (
            None
            if breadth_loss_kwargs is None
            else breadth_loss_kwargs["breadth_embed_dim"]
        )

    def _construct_loss(
        self,
        breadth_loss_kwargs: Optional[dict[str, Any]],
        depth_loss_kwargs: Optional[dict[str, Any]],
        max_depth_length: Optional[int],
        transformer_hidden_size: int,
    ) -> BreadthDepthLoss:
        loss_kwargs: dict[str, Any] = {
            "breadth_loss": None,
            "depth_loss": None,
            "max_depth_length": max_depth_length,
        }
        if breadth_loss_kwargs is not None:
            loss_kwargs["breadth_loss"] = self._construct_breadth_loss(
                **breadth_loss_kwargs,
                transformer_hidden_size=transformer_hidden_size,
            )

        if depth_loss_kwargs is not None:
            loss_kwargs["lam"] = depth_loss_kwargs.pop("lam")

            loss_kwargs["depth_loss"] = self._construct_depth_loss(
                **depth_loss_kwargs,
            )

        return BreadthDepthLoss(**loss_kwargs)

    def _construct_breadth_loss(
        self,
        loss_type: str,
        transformer_projection: list[dict[str, Any]],
        breadth_projection: list[dict[str, Any]],
        breadth_embed_dim: int,
        cca_output_dim: Optional[int],
        transformer_hidden_size: int,
        soft_cca_sdl_alpha: float,
        soft_cca_lam: Optional[float] = None,
        contrastive_lam: Optional[float] = None,
    ) -> cca_losses.ProjectionLoss:
        view1_dim = (
            transformer_projection[-1]["features"]
            if len(transformer_projection) > 0
            else transformer_hidden_size
        )
        view2_dim = (
            breadth_projection[-1]["features"]
            if len(breadth_projection) > 0
            else breadth_embed_dim
        )

        loss_fn = None
        if loss_type == "cca":
            loss_fn = cca_losses.CCALoss(output_dimension=cca_output_dim)
        elif loss_type == "running_cca":
            loss_fn = cca_losses.RunningCCALoss(
                view1_dimension=view1_dim,
                view2_dimension=view2_dim,
                output_dimension=cca_output_dim,
            )
        elif loss_type == "soft_cca":
            assert (
                soft_cca_lam is not None
            ), "To use soft_cca, `soft_cca_lam` must be set"
            loss_fn = cca_losses.SoftCCALoss(
                sdl1=cca_losses.StochasticDecorrelationLoss(
                    view1_dim,
                    alpha=soft_cca_sdl_alpha,
                ),
                sdl2=cca_losses.StochasticDecorrelationLoss(
                    view2_dim,
                    alpha=soft_cca_sdl_alpha,
                ),
                lam=soft_cca_lam,
            )
        else:
            loss_fn = create_sim_based_loss(loss_type, contrastive_lam=contrastive_lam)

        def _construct_net(
            blocks_config: list[dict[str, Any]], input_features: int
        ) -> Optional[cca_losses.DeepNet]:
            if len(blocks_config) == 0:
                return None

            return cca_losses.DeepNet(
                blocks_config=blocks_config,
                input_features=input_features,
            )

        net1 = _construct_net(transformer_projection, transformer_hidden_size)
        net2 = _construct_net(breadth_projection, breadth_embed_dim)

        return cca_losses.ProjectionLoss(
            net1=net1,
            net2=net2,
            loss_fn=loss_fn,
        )

    def _construct_depth_loss(
        self, loss_type: str, contrastive_lam: Optional[float]
    ) -> torch.nn.Module:
        return create_sim_based_loss(loss_type, contrastive_lam=contrastive_lam)

    def train(
        self,
        task: ExperimentalTask,
        epochs: int,
        warmup_steps: int,
        grad_accumulation_steps: int,
        patience: Optional[int],
        weight_decay: float,
        fp16: bool,
        max_grad_norm: float,
        lr: float,
        lr_scheduler_type: str,
        log_every_step: int,
        validate_every_step: Optional[int],
        dataloader_sampling: str,
        bucket_limits: list[int],
        save_best: bool,
        global_attention_type: str,
        device: Optional[str] = None,
        log_dir: Optional[str] = None,
        **_,
    ) -> None:
        to_dataloader = partial(
            self._to_dataloader,
            sampling=dataloader_sampling,
            return_length=False,
            global_attention_type=global_attention_type,
            sampler_kwargs={
                "effective_batch_size": self._batch_size * grad_accumulation_steps,
                "bucket_limits": bucket_limits,
            },
        )

        train_data = to_dataloader(task.train)

        val_data = None
        if task.validation is not None:
            val_data = to_dataloader(task.validation, training=False)

        optimizer = torch.optim.AdamW(
            train_utils.get_optimizer_params(self._model, weight_decay),
            lr=lr,
        )

        lr_scheduler = train_utils.get_lr_scheduler(
            scheduler_type=lr_scheduler_type,
            optimizer=optimizer,
            total_steps=epochs * len(train_data) // grad_accumulation_steps,
            warmup_steps=warmup_steps // grad_accumulation_steps,
        )

        save_model_callback = self._get_save_model_callback(save_best, log_dir)

        if self._transformer.supports_gradient_checkpointing:
            self._transformer.gradient_checkpointing_enable()

        train_logger, val_logger = self._get_train_val_loggers(
            log_dir, self._get_train_metrics(val_data, log_every_step)
        )

        trainer = TorchTrainer(
            model=self._model,
            optimizer=optimizer,
            train_logger=train_logger,
            val_logger=val_logger,
            fp16=fp16,
            max_grad_norm=max_grad_norm,
            grad_accumulation_steps=grad_accumulation_steps,
            lr_scheduler=lr_scheduler,
            validate_every_step=validate_every_step,
            save_model_callback=save_model_callback,
            patience=patience,
            device=device,
        )
        trainer.train(
            epochs=epochs,
            train_data=train_data,
            val_data=val_data,
        )

    def _get_train_metrics(
        self, val_data: Optional[DataLoader], default_log_freq: int
    ) -> list[TrainingMetric]:
        depth_len_thres = self._model.loss.max_depth_length
        train_metrics = [
            VMemMetric(default_log_freq),
            TrainingMetric(
                "mean_length",
                Mean(),
                default_log_freq,
                lambda metric, _, batch: metric.update(batch["length"]),
            ),
            EmbeddingMSEWithCol(
                "depth", default_log_freq, col_name=TeacherEmbedding.DEPTH_COL
            ),
            EmbeddingMSEWithCol(
                "depth",
                default_log_freq,
                col_name=TeacherEmbedding.DEPTH_COL,
                normalize=True,
            ),
            EmbeddingCosineDistanceWithCol(
                "depth", default_log_freq, col_name=TeacherEmbedding.DEPTH_COL
            ),
            TrainingMetric(
                "max_abs_transformer_grad",
                Max(),
                default_log_freq,
                partial(
                    log_max_abs_grad,
                    param_name="transformer.encoder.layer.11.output.dense.weight",
                    model=self._model,
                ),
            ),
            *self._get_projection_metrics(self._model, val_data, default_log_freq),
        ]

        if isinstance(self._model.loss.depth_loss, ContrastiveLoss):
            train_metrics.extend(
                [
                    TrainingMetric(
                        "depth_contrastive_positive",
                        Mean(),
                        default_log_freq,
                        lambda metric, outputs, _: metric.update(
                            outputs["depth_contrastive_positive"]
                        ),
                    ),
                    TrainingMetric(
                        "depth_contrastive_negative",
                        Mean(),
                        default_log_freq,
                        lambda metric, outputs, _: metric.update(
                            outputs["depth_contrastive_negative"]
                        ),
                    ),
                ]
            )

        if isinstance(
            self._model.loss.breadth_loss, cca_losses.ProjectionLoss
        ) and isinstance(self._model.loss.breadth_loss.loss_fn, ContrastiveLoss):
            train_metrics.extend(
                [
                    TrainingMetric(
                        "breadth_contrastive_positive",
                        Mean(),
                        default_log_freq,
                        lambda metric, outputs, _: metric.update(
                            outputs["breadth_contrastive_positive"]
                        ),
                    ),
                    TrainingMetric(
                        "breadth_contrastive_negative",
                        Mean(),
                        default_log_freq,
                        lambda metric, outputs, _: metric.update(
                            outputs["breadth_contrastive_negative"]
                        ),
                    ),
                ]
            )

        # sbert_mse_None would be equal to sbert_mse, same for cos_dist
        if depth_len_thres is not None:
            train_metrics.extend(
                [
                    EmbeddingMSEWithCol(
                        "depth",
                        default_log_freq,
                        col_name=TeacherEmbedding.DEPTH_COL,
                        max_input_length=depth_len_thres,
                    ),
                    EmbeddingMSEWithCol(
                        "depth",
                        default_log_freq,
                        col_name=TeacherEmbedding.DEPTH_COL,
                        max_input_length=depth_len_thres,
                        normalize=True,
                    ),
                    EmbeddingCosineDistanceWithCol(
                        "depth",
                        default_log_freq,
                        col_name=TeacherEmbedding.DEPTH_COL,
                        max_input_length=depth_len_thres,
                    ),
                ]
            )

        if self._model.loss.depth_loss is not None:
            train_metrics.extend(
                [
                    TrainingMetric(
                        "mean_depth_loss",
                        Mean(),
                        default_log_freq,
                        lambda metric, outputs, _: metric.update(outputs["depth_loss"]),
                    ),
                    TrainingMetric(
                        "mean_depth_mask",
                        Mean(),
                        default_log_freq,
                        lambda metric, outputs, _: metric.update(outputs["depth_mask"]),
                    ),
                ]
            )

        if self._model.loss.breadth_loss is not None:
            train_metrics.append(
                TrainingMetric(
                    "mean_breadth_loss",
                    Mean(),
                    default_log_freq,
                    lambda metric, outputs, _: metric.update(outputs["breadth_loss"]),
                )
            )

            if isinstance(
                self._model.loss.breadth_loss.loss_fn, cca_losses.SoftCCALoss
            ):
                train_metrics.extend(
                    [
                        TrainingMetric(
                            "mean_projection_l2_norm",
                            Mean(),
                            default_log_freq,
                            lambda metric, outputs, _: metric.update(
                                outputs["breadth_l2"]
                            ),
                        ),
                        TrainingMetric(
                            "mean_projection_sdl1",
                            Mean(),
                            default_log_freq,
                            lambda metric, outputs, _: metric.update(
                                outputs["breadth_sdl1"]
                            ),
                        ),
                        TrainingMetric(
                            "mean_projection_sdl2",
                            Mean(),
                            default_log_freq,
                            lambda metric, outputs, _: metric.update(
                                outputs["breadth_sdl2"]
                            ),
                        ),
                    ]
                )

        return train_metrics

    def _get_projection_metrics(
        self,
        model: _SequenceEmbeddingModel,
        val_data: Optional[DataLoader],
        default_log_freq: int,
    ) -> list[TrainingMetric]:
        if model.loss.breadth_loss is None or not isinstance(
            model.loss.breadth_loss, cca_losses.ProjectionLoss
        ):
            return []

        # breadth embed should be specified since we're using breadth loss
        assert self.breadth_embed_dim is not None

        train_metrics = []

        for net_name in ["net1", "net2"]:
            net = getattr(model.loss.breadth_loss, net_name)
            if net is not None:
                layer_count = len(net.layers)
                sample_projection_weight_path = (
                    f"loss.breadth_loss.{net_name}.layers.{layer_count -1}.0.weight"
                )

                train_metrics.append(
                    TrainingMetric(
                        f"max_abs_{net_name}_grad",
                        Max(),
                        default_log_freq,
                        partial(
                            log_max_abs_grad,
                            param_name=sample_projection_weight_path,
                            model=model,
                        ),
                    )
                )

        def update_with_outputs(metric, outputs, batch) -> None:
            metric.update(outputs["embedding"], batch[TeacherEmbedding.BREADTH_COL])

        def warn_for_nans_in_validation(
            cca_metric: WindowedCCAMetric, metric_name: str
        ):
            if (
                val_data is not None
                and len(val_data) * (val_data.batch_size or 1) < cca_metric.window_size
            ):
                logger.warn(
                    "Validation data smaller than CCA window. "
                    "Metric '%s' will be output nans.",
                    metric_name,
                )

        for n_components, window_size in [
            (self.breadth_embed_dim, multiplier * self.breadth_embed_dim)
            for multiplier in [10, 15]
        ]:
            cca_metric = WindowedCCAMetricZoo(
                n_components=n_components,
                window_size=window_size,
            )
            metric_name = f"cca_outputs_{n_components}x{cca_metric.window_size}"
            warn_for_nans_in_validation(cca_metric, metric_name)

            train_metrics.extend(
                [
                    TrainingMetric(
                        metric_name,
                        cca_metric,
                        default_log_freq,
                        update_with_outputs,
                        reset_after_log=False,
                    ),
                    TrainingMetric(
                        f"crosscorr_outputs_x{cca_metric.window_size}",
                        WindowedAbsCrossCorrelationMetric(cca_metric.window_size),
                        default_log_freq,
                        update_with_outputs,
                        reset_after_log=False,
                    ),
                    TrainingMetric(
                        f"corr_transformer_outputs_x{cca_metric.window_size}",
                        WindowedAbsCorrelationMetric(cca_metric.window_size),
                        default_log_freq,
                        lambda metric, outputs, _: metric.update(outputs["embedding"]),
                        reset_after_log=False,
                    ),
                    TrainingMetric(
                        f"corr_breadth_outputs_x{cca_metric.window_size}",
                        WindowedAbsCorrelationMetric(cca_metric.window_size),
                        default_log_freq,
                        lambda metric, _, batch: metric.update(
                            batch[TeacherEmbedding.BREADTH_COL]
                        ),
                        reset_after_log=False,
                    ),
                ]
            )

        def update_with_projection(metric, outputs, _, *, layer_ind: int) -> None:
            metric.update(
                outputs["breadth_projected_views1"][layer_ind],
                outputs["breadth_projected_views2"][layer_ind],
            )

        def update_with_first_projection(metric, outputs, _, layer_ind):
            metric.update(outputs["breadth_projected_views1"][layer_ind])

        def update_with_second_projection(metric, outputs, _, layer_ind):
            metric.update(outputs["breadth_projected_views2"][layer_ind])

        if (
            isinstance(model.loss.breadth_loss, cca_losses.ProjectionLoss)
            and model.loss.breadth_loss.net1 is not None
            and model.loss.breadth_loss.net2 is not None
        ):
            net1_feats = model.loss.breadth_loss.net1.features
            net2_feats = model.loss.breadth_loss.net2.features

            for layer_ind in range(-1, -min(len(net1_feats), len(net2_feats)) - 1, -1):
                min_dim = min(net1_feats[layer_ind], net2_feats[layer_ind])

                both_views_update_fn = partial(
                    update_with_projection, layer_ind=layer_ind
                )
                view1_update_fn = partial(
                    update_with_first_projection, layer_ind=layer_ind
                )
                view2_update_fn = partial(
                    update_with_second_projection, layer_ind=layer_ind
                )

                for multiplier in [5, 10]:
                    cca_metric = WindowedCCAMetricZoo(
                        n_components=min_dim, window_size=min_dim * multiplier
                    )

                    metric_name = (
                        f"cca_projection[{layer_ind}]_"
                        f"{cca_metric.n_components}x{cca_metric.window_size}"
                    )
                    warn_for_nans_in_validation(cca_metric, metric_name)

                    train_metrics.extend(
                        [
                            TrainingMetric(
                                metric_name,
                                cca_metric,
                                default_log_freq,
                                both_views_update_fn,
                                reset_after_log=False,
                            ),
                            TrainingMetric(
                                f"crosscorr_projection[{layer_ind}]_x{cca_metric.window_size}",
                                WindowedAbsCrossCorrelationMetric(
                                    cca_metric.window_size
                                ),
                                default_log_freq,
                                both_views_update_fn,
                                reset_after_log=False,
                            ),
                            TrainingMetric(
                                f"corr_transformer_projection[{layer_ind}]_x{cca_metric.window_size}",
                                WindowedAbsCorrelationMetric(cca_metric.window_size),
                                default_log_freq,
                                view1_update_fn,
                                reset_after_log=False,
                            ),
                            TrainingMetric(
                                f"corr_breadth_projection[{layer_ind}]_x{cca_metric.window_size}",
                                WindowedAbsCorrelationMetric(cca_metric.window_size),
                                default_log_freq,
                                view2_update_fn,
                                reset_after_log=False,
                            ),
                        ]
                    )

        return train_metrics

    @torch.inference_mode()
    def predict(self, inputs: Dataset) -> Iterable[np.ndarray]:
        self._model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Predicting using {device}")
        self._model.to(device)

        data = self._to_dataloader(inputs, training=False)
        for batch in tqdm(data, desc="Predicting batches"):
            if "labels" in batch:
                del batch["labels"]
            train_utils.batch_to_device(batch, device)
            embeddings = self._model(**batch)["embedding"]
            yield embeddings.numpy(force=True)
