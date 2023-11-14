from __future__ import annotations
from functools import partial
import os
import torch
import logging

from typing import Any, cast, TYPE_CHECKING
from torch.utils.tensorboard.writer import SummaryWriter
from torcheval.metrics import Max, Mean, Metric
from transformers import AutoTokenizer, PreTrainedModel
from transformer_document_embedding.baselines.baseline import Baseline
import transformer_document_embedding.utils.torch.losses as losses
import transformer_document_embedding.utils.torch.training as train_utils
import transformer_document_embedding.baselines.longformer.train as longformer_training
from transformer_document_embedding.models.longformer import (
    LongformerConfig,
    LongformerForTextEmbedding,
)
from transformer_document_embedding.utils.metrics import (
    WindowedNonResetableCCAMetric,
    CosineDistanceWithSBERT,
    MSEWithSBERT,
    PercentLengthBelowThres,
    with_accessor,
    VMemMetric,
)
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    import numpy as np
    from transformer_document_embedding.tasks.experimental_task import ExperimentalTask
    from datasets import Dataset
    from typing import Optional, Iterable

logger = logging.getLogger(__name__)


class LongformerStudent(Baseline):
    def __init__(
        self,
        large: bool,
        batch_size: int,
        pooler_type: str,
        contextual_loss_kwargs: Optional[dict[str, Any]] = None,
        static_loss_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self._batch_size = batch_size

        model_path = f"allenai/longformer-{'large' if large else 'base'}-4096"
        config = LongformerConfig(
            pooler_type=pooler_type,
            **LongformerConfig.get_config_dict(model_path)[0],
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._longformer = cast(
            LongformerForTextEmbedding,
            LongformerForTextEmbedding.from_pretrained(model_path, config=config),
        )

        self._loss = self._construct_loss(
            static_loss_kwargs=static_loss_kwargs,
            contextual_loss_kwargs=contextual_loss_kwargs,
        )

    def _construct_loss(
        self,
        static_loss_kwargs: Optional[dict[str, Any]],
        contextual_loss_kwargs: Optional[dict[str, Any]],
    ) -> losses.StaticContextualLoss:
        loss_kwargs: dict[str, Any] = {
            "static_loss": None,
            "contextual_loss": None,
        }
        if static_loss_kwargs is not None:
            loss_kwargs["static_key"] = static_loss_kwargs.pop("static_key")

            loss_kwargs["static_loss"] = self._construct_static_loss(
                **static_loss_kwargs
            )

        if contextual_loss_kwargs is not None:
            for param_name in ["contextual_key", "contextual_max_length", "lam"]:
                loss_kwargs[param_name] = contextual_loss_kwargs.pop(param_name)

            loss_kwargs["contextual_loss"] = self._construct_contextual_loss(
                **contextual_loss_kwargs
            )

        return losses.StaticContextualLoss(**loss_kwargs)

    def _construct_static_loss(
        self,
        static_loss_type: str,
        longformer_projection_layers: list[int],
        static_projection_layers: list[int],
        projection_norm: Optional[str],
        static_embed_dim: int,
        cca_output_dim: Optional[int],
    ) -> losses.ProjectionLoss:
        longformer_output_features = self._longformer.config.hidden_size

        view1_dim = (
            longformer_projection_layers[-1]
            if len(longformer_projection_layers) > 0
            else longformer_output_features
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
            static_loss_fn = losses.SoftCCALoss(
                sdl1=losses.StochasticDecorrelationLoss(view1_dim),
                sdl2=losses.StochasticDecorrelationLoss(view2_dim),
                lam=0.8,
            )
        elif static_loss_type == "mse":
            static_loss_fn = torch.nn.MSELoss()
        elif static_loss_type == "cos":
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

        net1 = _construct_net(longformer_projection_layers, longformer_output_features)
        net2 = _construct_net(static_projection_layers, static_embed_dim)

        return losses.ProjectionLoss(
            net1=net1,
            net2=net2,
            loss_fn=static_loss_fn,
        )

    def _construct_contextual_loss(self, contextual_loss_type: str) -> torch.nn.Module:
        if contextual_loss_type == "mse":
            return torch.nn.MSELoss()
        elif contextual_loss_type == "cos":
            return losses.CosineDistanceLoss()

        raise ValueError(
            "Unknown contextual `loss_type`: {}".format(contextual_loss_type)
        )

    def train(
        self,
        task: ExperimentalTask,
        dataloader_sampling: str,
        grad_accumulation_steps: int,
        learning_rate: float,
        weight_decay: float,
        warmup_steps: int,
        epochs: int,
        log_every_step: int,
        save_best: bool,
        log_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        validate_every_step: Optional[int] = None,
        **_,
    ) -> None:
        model = _LongformerStudentWrapper(transformer=self._longformer, loss=self._loss)

        def _create_summary_writer(name: str) -> Optional[SummaryWriter]:
            return (
                SummaryWriter(os.path.join(log_dir, name))
                if log_dir is not None
                else None
            )

        def _create_data_loader(dataset, training: bool = True) -> DataLoader:
            return train_utils.create_tokenized_data_loader(
                dataset,
                tokenizer=self._tokenizer,
                batch_size=self._batch_size,
                sampling=dataloader_sampling,
                training=training,
                return_length=False,
                sampler_kwargs={
                    "effective_batch_size": self._batch_size * grad_accumulation_steps,
                    "bucket_limits": [model.loss.contextual_max_length],
                },
            )

        train_data = _create_data_loader(task.train)

        val_summary_writer = None
        val_data = None
        if task.validation is not None:
            val_summary_writer = _create_summary_writer("val")
            val_data = _create_data_loader(task.validation, training=False)

        optimizer = torch.optim.AdamW(
            train_utils.get_optimizer_params(model, weight_decay),
            lr=learning_rate,
        )

        lr_scheduler = train_utils.get_linear_lr_scheduler_with_warmup(
            optimizer,
            warmup_steps,
            epochs * len(train_data),
        )

        summary_writer = _create_summary_writer("train")

        save_model_callback = None
        if save_best and model_dir is not None:

            def save_cb(*_) -> None:
                self.save(os.path.join(model_dir, "checkpoint"))

            save_model_callback = save_cb

        model.transformer.gradient_checkpointing_enable()
        trainer = longformer_training.LongformerTrainer(
            model=model,
            train_data=train_data,
            optimizer=optimizer,
            metrics=self._construct_train_metrics(model, val_data),
            summary_writer=summary_writer,
            val_summary_writer=val_summary_writer,
            val_data=val_data,
            validate_every_step=validate_every_step,
            fp16=True,
            max_grad_norm=1.0,
            grad_accumulation_steps=grad_accumulation_steps,
            save_model_callback=save_model_callback,
            lr_scheduler=lr_scheduler,
            log_every_step=log_every_step,
            main_metric="loss",
            lower_is_better=True,
        )

        trainer.train(epochs=epochs)

    def _construct_train_metrics(
        self, model: _LongformerStudentWrapper, val_data: Optional[DataLoader]
    ) -> dict[str, Metric]:
        train_metrics = {
            "used_vmem": VMemMetric(),
            "mean_length": with_accessor(
                Mean(),
                lambda metric, _, batch: metric.update(batch["length"]),
            ),
            "sbert_mse": MSEWithSBERT(),
            "sbert_cos": CosineDistanceWithSBERT(),
            "short_percentage": PercentLengthBelowThres(
                model.loss.contextual_max_length
            ),
            "max_abs_longformer_grad": with_accessor(
                Max(),
                partial(
                    log_max_abs_grad,
                    param_name="transformer.longformer.encoder.layer.11.output.dense.weight",
                    model=model,
                ),
            ),
            **self._construct_projection_metrics(model, val_data),
        }

        if model.loss.contextual_loss is not None:
            train_metrics["mean_contextual_loss"] = with_accessor(
                Mean(),
                lambda metric, outputs, _: metric.update(outputs["contextual_loss"]),
            )

        if model.loss.static_loss is not None:
            train_metrics["mean_static_loss"] = with_accessor(
                Mean(),
                lambda metric, outputs, _: metric.update(outputs["static_loss"]),
            )

            if isinstance(model.loss.static_loss.loss_fn, losses.SoftCCALoss):
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
        model: _LongformerStudentWrapper,
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
            inner_metric = WindowedNonResetableCCAMetric(n_components=n_components)
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

    @torch.no_grad()
    def predict(self, inputs: Dataset) -> Iterable[np.ndarray]:
        self._longformer.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Predicting using {device}")
        self._longformer.to(device)

        data = train_utils.create_tokenized_data_loader(
            inputs,
            tokenizer=self._tokenizer,
            batch_size=self._batch_size,
            training=False,
        )
        for batch in tqdm(data, desc="Predicting batches"):
            train_utils.batch_to_device(batch, device)
            embeddings = self._longformer(**batch).pooler_output
            yield embeddings.numpy(force=True)

    @classmethod
    def _longformer_path(cls, dir_path: str) -> str:
        return os.path.join(dir_path, "longformer")

    @classmethod
    def _loss_path(cls, dir_path: str) -> str:
        return os.path.join(dir_path, "loss")

    def save(self, dir_path: str) -> None:
        self._longformer.save_pretrained(self._longformer_path(dir_path))
        torch.save(self._loss.state_dict(), self._loss_path(dir_path))

    def load(self, dir_path: str) -> None:
        self._longformer = cast(
            LongformerForTextEmbedding,
            LongformerForTextEmbedding.from_pretrained(self._longformer_path(dir_path)),
        )
        try:
            self._loss.load_state_dict(torch.load(self._loss_path(dir_path)))
        except Exception as e:
            logger.warn("Error when loading loss. Skipping...: %s", e)


class _LongformerStudentWrapper(torch.nn.Module):
    def __init__(
        self,
        transformer: PreTrainedModel,
        loss: losses.StaticContextualLoss,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.transformer = transformer
        self.loss = loss

    def forward(self, **batch) -> dict[str, Any]:
        outputs = self.transformer(**batch)
        loss_outputs = self.loss(outputs["pooler_output"], batch)
        return {
            **outputs,
            **loss_outputs,
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
