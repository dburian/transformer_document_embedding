from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING, Any, Iterable
import logging

import torch
from torcheval.metrics import Max, Mean
from tqdm.auto import tqdm


from transformer_document_embedding.models.transformer.base import TransformerBase
from transformer_document_embedding.models.trainer import TorchTrainer
from transformer_document_embedding.utils.metrics import (
    CosineDistanceWithSBERT,
    MSEWithSBERT,
    VMemMetric,
    WindowedNonResetableCCAMetricTorch,
    with_accessor,
)
import transformer_document_embedding.utils.training as train_utils
import transformer_document_embedding.utils.losses as losses

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


class TransformerStudent(TransformerBase):
    def __init__(
        self,
        transformer_model: str,
        batch_size: int,
        pooler_type: str,
        contextual_max_length: Optional[int],
        contextual_loss_kwargs: Optional[dict[str, Any]],
        static_loss_kwargs: Optional[dict[str, Any]],
        transformer_model_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            transformer_model=transformer_model,
            transformer_model_kwargs=transformer_model_kwargs,
            batch_size=batch_size,
            pooler_type=pooler_type,
        )
        loss = self._construct_loss(
            static_loss_kwargs,
            contextual_loss_kwargs,
            contextual_max_length,
            transformer_hidden_size=self._transformer.config.hidden_size,
        )

        self._model: _SequenceEmbeddingModel = _SequenceEmbeddingModel(
            self._transformer, self._pooler, loss
        )

    def _construct_loss(
        self,
        static_loss_kwargs: Optional[dict[str, Any]],
        contextual_loss_kwargs: Optional[dict[str, Any]],
        contextual_max_length: Optional[int],
        transformer_hidden_size: int,
    ) -> losses.StaticContextualLoss:
        loss_kwargs: dict[str, Any] = {
            "static_loss": None,
            "contextual_loss": None,
            "contextual_max_length": contextual_max_length,
        }
        if static_loss_kwargs is not None:
            loss_kwargs["static_key"] = static_loss_kwargs.pop("static_key")

            loss_kwargs["static_loss"] = self._construct_static_loss(
                **static_loss_kwargs,
                transformer_hidden_size=transformer_hidden_size,
            )

        if contextual_loss_kwargs is not None:
            for param_name in ["contextual_key", "lam"]:
                loss_kwargs[param_name] = contextual_loss_kwargs.pop(param_name)

            loss_kwargs["contextual_loss"] = self._construct_contextual_loss(
                **contextual_loss_kwargs,
            )

        return losses.StaticContextualLoss(**loss_kwargs)

    def _construct_static_loss(
        self,
        static_loss_type: str,
        transformer_projection_layers: list[int],
        static_projection_layers: list[int],
        projection_norm: Optional[str],
        static_embed_dim: int,
        cca_output_dim: Optional[int],
        transformer_hidden_size: int,
        soft_cca_lam: Optional[float],
    ) -> losses.ProjectionLoss:
        view1_dim = (
            transformer_projection_layers[-1]
            if len(transformer_projection_layers) > 0
            else transformer_hidden_size
        )
        view2_dim = (
            static_projection_layers[-1]
            if len(static_projection_layers) > 0
            else static_embed_dim
        )

        static_loss_fn = None
        if static_loss_type == "cca":
            static_loss_fn = losses.CCALoss(output_dimension=cca_output_dim)
        elif static_loss_type == "running_cca":
            static_loss_fn = losses.RunningCCALoss(
                view1_dimension=view1_dim,
                view2_dimension=view2_dim,
                output_dimension=cca_output_dim,
            )
        elif static_loss_type == "soft_cca":
            assert (
                soft_cca_lam is not None
            ), "To use soft_cca, `soft_cca_lam` must be set"
            static_loss_fn = losses.SoftCCALoss(
                sdl1=losses.StochasticDecorrelationLoss(view1_dim),
                sdl2=losses.StochasticDecorrelationLoss(view2_dim),
                lam=soft_cca_lam,
            )
        elif static_loss_type == "mse":
            static_loss_fn = torch.nn.MSELoss(reduction="none")
        elif static_loss_type == "cos_dist":
            static_loss_fn = losses.CosineDistanceLoss()

        assert static_loss_fn is not None, "Unknown `static_loss_type`."

        def _construct_net(
            layers: list[int], input_features: int
        ) -> Optional[losses.DeepNet]:
            if len(layers) == 0:
                return None

            return losses.DeepNet(
                layer_features=layers,
                input_features=input_features,
                norm=projection_norm,
            )

        net1 = _construct_net(transformer_projection_layers, transformer_hidden_size)
        net2 = _construct_net(static_projection_layers, static_embed_dim)

        return losses.ProjectionLoss(
            net1=net1,
            net2=net2,
            loss_fn=static_loss_fn,
        )

    def _construct_contextual_loss(self, contextual_loss_type: str) -> torch.nn.Module:
        if contextual_loss_type == "mse":
            return torch.nn.MSELoss(reduction="none")
        elif contextual_loss_type == "cos_dist":
            return losses.CosineDistanceLoss()

        raise ValueError(
            "Unknown contextual `loss_type`: {}".format(contextual_loss_type)
        )

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
        log_every_step: int,
        validate_every_step: Optional[int],
        dataloader_sampling: str,
        lr: float,
        bucket_limits: list[int],
        save_best: bool,
        device: Optional[str] = None,
        log_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        **_,
    ) -> None:
        summary_writer, val_summary_writer = self._get_train_val_writers(log_dir)
        to_dataloader = partial(
            self._to_dataloader,
            sampling=dataloader_sampling,
            return_length=False,
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

        lr_scheduler = train_utils.get_linear_lr_scheduler_with_warmup(
            optimizer,
            warmup_steps // grad_accumulation_steps,
            epochs * len(train_data) // grad_accumulation_steps,
        )

        save_model_callback = self._get_save_model_callback(save_best, model_dir)

        if self._transformer.supports_gradient_checkpointing:
            self._transformer.gradient_checkpointing_enable()

        trainer = TorchTrainer(
            model=self._model,
            train_data=train_data,
            val_data=val_data,
            optimizer=optimizer,
            metrics=self._get_train_metrics(val_data),
            summary_writer=summary_writer,
            val_summary_writer=val_summary_writer,
            fp16=fp16,
            max_grad_norm=max_grad_norm,
            grad_accumulation_steps=grad_accumulation_steps,
            lr_scheduler=lr_scheduler,
            log_every_step=log_every_step,
            validate_every_step=validate_every_step,
            save_model_callback=save_model_callback,
            patience=patience,
            device=device,
        )
        trainer.train(epochs=epochs)

    def _get_train_metrics(self, val_data: Optional[DataLoader]) -> dict[str, Metric]:
        contextual_len_thres = self._model.loss.contextual_max_length
        train_metrics = {
            "used_vmem": VMemMetric(),
            "mean_length": with_accessor(
                Mean(),
                lambda metric, _, batch: metric.update(batch["length"]),
            ),
            "sbert_mse": MSEWithSBERT(max_input_length=None),
            "sbert_mse_norm": MSEWithSBERT(max_input_length=None, normalize=True),
            "sbert_cos_dist": CosineDistanceWithSBERT(max_input_length=None),
            "max_abs_transformer_grad": with_accessor(
                Max(),
                partial(
                    log_max_abs_grad,
                    param_name="transformer.encoder.layer.11.output.dense.weight",
                    model=self._model,
                ),
            ),
            **self._construct_projection_metrics(self._model, val_data),
        }

        # If it is None, sbert_mse_None is same as sbert_mse (same for cos_dist)
        if contextual_len_thres is not None:
            train_metrics[f"sbert_mse_{contextual_len_thres}"] = MSEWithSBERT(
                max_input_length=contextual_len_thres
            )
            train_metrics[f"sbert_mse_norm_{contextual_len_thres}"] = MSEWithSBERT(
                max_input_length=contextual_len_thres,
                normalize=True,
            )
            train_metrics[
                f"sbert_cos_dist_{contextual_len_thres}"
            ] = CosineDistanceWithSBERT(max_input_length=contextual_len_thres)

        if self._model.loss.contextual_loss is not None:
            train_metrics["mean_contextual_loss"] = with_accessor(
                Mean(),
                lambda metric, outputs, _: metric.update(outputs["contextual_loss"]),
            )
            train_metrics["mean_contextual_mask"] = with_accessor(
                Mean(),
                lambda metric, outputs, _: metric.update(outputs["contextual_mask"]),
            )

        if self._model.loss.static_loss is not None:
            train_metrics["mean_static_loss"] = with_accessor(
                Mean(),
                lambda metric, outputs, _: metric.update(outputs["static_loss"]),
            )

            if isinstance(self._model.loss.static_loss.loss_fn, losses.SoftCCALoss):
                train_metrics["mean_projection_l2_norm"] = with_accessor(
                    Mean(),
                    lambda metric, outputs, _: metric.update(outputs["l2"]),
                )
                train_metrics["mean_projection_sdl1"] = with_accessor(
                    Mean(),
                    lambda metric, outputs, _: metric.update(outputs["sdl1"]),
                )
                train_metrics["mean_projection_sdl2"] = with_accessor(
                    Mean(),
                    lambda metric, outputs, _: metric.update(outputs["sdl2"]),
                )

        return train_metrics

    def _construct_projection_metrics(
        self,
        model: _SequenceEmbeddingModel,
        val_data: Optional[DataLoader],
    ) -> dict[str, Metric]:
        if model.loss.static_loss is None or not isinstance(
            model.loss.static_loss, losses.ProjectionLoss
        ):
            return {}

        train_metrics = {}
        sample_projection_weight_path = None

        for net_name in ["net1", "net2"]:
            net = getattr(model.loss.static_loss, net_name)
            if net is not None:
                layer_count = len(net.layers)
                sample_projection_weight_path = (
                    f"loss.static_loss.{net_name}.layers.{layer_count -1}.weight"
                )
                break

        if sample_projection_weight_path is not None:
            train_metrics["max_abs_dcca_grad"] = with_accessor(
                Max(),
                partial(
                    log_max_abs_grad,
                    param_name=sample_projection_weight_path,
                    model=model,
                ),
            )

        def _update_with_projected_views(metric, outputs, _):
            metric.update(outputs["projected_view1"], outputs["projected_view2"])

        for n_components in [64, 128, 256, 512]:
            inner_metric = WindowedNonResetableCCAMetricTorch(n_components=n_components)
            metric_name = f"cca_{n_components}x{inner_metric.window_size}"
            if (
                val_data is not None
                and (val_size := len(val_data) * (val_data.batch_size or 1))
                < inner_metric.window_size
            ):
                logger.warn(
                    "Validation data smaller than CCA window. "
                    "Metric '%s' will be outdated by %d inputs.",
                    metric_name,
                    inner_metric.window_size - val_size,
                )

            train_metrics[metric_name] = with_accessor(
                inner_metric,
                _update_with_projected_views,
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
            embeddings = self._model(**batch)["embeddings"]
            yield embeddings.numpy(force=True)


class _SequenceEmbeddingModel(torch.nn.Module):
    def __init__(
        self,
        transformer: PreTrainedModel,
        pooler: torch.nn.Module,
        loss: losses.StaticContextualLoss,
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
        )
        loss_outputs = self.loss(pooled_output, kwargs) if len(kwargs) > 0 else {}

        return {
            **outputs,
            **loss_outputs,
            "embeddings": pooled_output,
        }


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
