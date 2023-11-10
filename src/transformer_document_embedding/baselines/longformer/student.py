from __future__ import annotations
from functools import partial
import os
import torch
import logging

from typing import Any, cast, TYPE_CHECKING
from torch.utils.tensorboard.writer import SummaryWriter
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
    WindowedCCAMetric,
    WindowedCosineDistanceWithSBERT,
    WindowedMSEWithSBERT,
    PercentLengthBelowThres,
    WindowedMax,
    WindowedMean,
    with_accessor,
    VMemMetric,
)
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from torcheval import metrics
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
        static_embed_dim: int,
        contextual_max_length: int,
        static_contextual_lambda: float,
        static_loss_type: str,
        longformer_projection_layers: list[int],
        static_projection_layers: list[int],
        cca_output_dim: Optional[int] = None,
        projection_norm: Optional[str] = None,
    ) -> None:
        self._batch_size = batch_size
        self._contextual_max_length = contextual_max_length

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
            static_loss_type=static_loss_type,
            longformer_projection_layers=longformer_projection_layers,
            static_projection_layers=static_projection_layers,
            projection_norm=projection_norm,
            static_embed_dim=static_embed_dim,
            cca_output_dim=cca_output_dim,
            static_contextual_lambda=static_contextual_lambda,
            contextual_max_length=contextual_max_length,
        )

    def _construct_loss(
        self,
        static_loss_type: str,
        longformer_projection_layers: list[int],
        static_projection_layers: list[int],
        projection_norm: Optional[str],
        static_embed_dim: int,
        cca_output_dim: Optional[int],
        static_contextual_lambda: float,
        contextual_max_length: int,
    ) -> losses.AlwaysStaticShortContextual:
        static_loss = self._construct_static_loss(
            static_loss_type=static_loss_type,
            longformer_projection_layers=longformer_projection_layers,
            static_projection_layers=static_projection_layers,
            projection_norm=projection_norm,
            static_embed_dim=static_embed_dim,
            cca_output_dim=cca_output_dim,
        )

        return losses.AlwaysStaticShortContextual(
            contextual_key="sbert",
            static_key="dbow",
            static_loss=static_loss,
            lambda_=static_contextual_lambda,
            len_threshold=contextual_max_length,
        )

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
                view1_dimension=view1_dim,
                view2_dimension=view2_dim,
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
        log_dir: Optional[str] = None,
        **_,
    ) -> None:
        model = _LongformerStudentWrapper(transformer=self._longformer, loss=self._loss)

        train_data = train_utils.create_tokenized_data_loader(
            task.train,
            tokenizer=self._tokenizer,
            batch_size=self._batch_size,
            sampling=dataloader_sampling,
            training=True,
            return_length=True,
            sampler_kwargs={
                "effective_batch_size": self._batch_size * grad_accumulation_steps,
                "bucket_limits": [384],
                # "short_count": short_inputs_in_effective_batch,
                # "short_threshold": self._contextual_max_length,
            },
        )

        optimizer = torch.optim.AdamW(
            train_utils.get_optimizer_params(model, weight_decay),
            lr=learning_rate,
        )

        lr_scheduler = train_utils.get_linear_lr_scheduler_with_warmup(
            optimizer,
            warmup_steps,
            epochs * len(train_data),
        )

        summary_writer = (
            SummaryWriter(os.path.join(log_dir, "train"))
            if log_dir is not None
            else None
        )

        model.transformer.gradient_checkpointing_enable()
        trainer = longformer_training.LongformerTrainer(
            model=model,
            train_data=train_data,
            optimizer=optimizer,
            metrics=self._construct_train_metrics(model),
            summary_writer=summary_writer,
            fp16=True,
            max_grad_norm=1.0,
            grad_accumulation_steps=grad_accumulation_steps,
            lr_scheduler=lr_scheduler,
            log_every_step=log_every_step,
        )

        trainer.train(epochs=epochs)

    def _construct_train_metrics(
        self, model: _LongformerStudentWrapper
    ) -> dict[str, metrics.Metric]:
        def log_max_abs_grad(metric, *_, param: str) -> None:
            grad = model.get_parameter(param).grad
            assert grad is not None

            metric.update(grad.abs().max())

        # torch.from_numpy(np.load("notebooks/wiki_sample_dbow_sigma.npy")).to(
        #     torch.device("cuda")
        # )

        train_metrics = {
            "used_vmem": VMemMetric(),
            "mean_contextual_loss": with_accessor(
                WindowedMean(),
                lambda metric, outputs, _: metric.update(outputs["contextual_loss"]),
            ),
            "mean_static_loss": with_accessor(
                WindowedMean(),
                lambda metric, outputs, _: metric.update(outputs["static_loss"]),
            ),
            "mean_length": with_accessor(
                WindowedMean(),
                lambda metric, _, batch: metric.update(batch["length"]),
            ),
            "sbert_mse": WindowedMSEWithSBERT(),
            "sbert_cos": WindowedCosineDistanceWithSBERT(),
            "short_percentage": PercentLengthBelowThres(model.loss.len_threshold),
            # "mean_cov_mat_sum": with_accessor(
            #     metrics.Mean(),
            #     lambda metric, outputs, _: metric.update(
            #         outputs["covariance_mat"].sum()
            #     ),
            # ),
            # "abs_diff_dbow_sigma": with_accessor(
            #     metrics.Mean(),
            #     lambda metric, outputs, _: metric.update(
            #         torch.abs(outputs["sigma2"] - true_dbow_sigma).sum()
            #     ),
            # ),
            "max_abs_longformer_grad": with_accessor(
                WindowedMax(),
                partial(
                    log_max_abs_grad,
                    param="transformer.longformer.encoder.layer.11.output.dense.weight",
                ),
            ),
        }

        for n_components in [128, 256, 512]:
            train_metrics[f"cca_{n_components}"] = with_accessor(
                WindowedCCAMetric(
                    n_components=n_components,
                    max_window_size=5 * n_components,
                ),
                lambda metric, outputs, _: metric.update(
                    outputs["projected_view1"], outputs["projected_view2"]
                ),
            )

        if isinstance(model.loss.static_loss, losses.ProjectionLoss):
            sample_projection_weight_path = None

            if model.loss.static_loss.net1 is not None:
                layer_count = len(model.loss.static_loss.net1.layers)
                sample_projection_weight_path = (
                    f"loss.static_loss.net1.layers.{layer_count -1}.weight"
                )
            elif model.loss.static_loss.net2 is not None:
                layer_count = len(model.loss.static_loss.net2.layers)
                sample_projection_weight_path = (
                    f"loss.static_loss.net2.layers.{layer_count -1}.weight"
                )

            if sample_projection_weight_path is not None:
                train_metrics["max_abs_dcca_grad"] = with_accessor(
                    WindowedMax(),
                    partial(
                        log_max_abs_grad,
                        param=sample_projection_weight_path,
                    ),
                )

            if isinstance(model.loss.static_loss.loss_fn, losses.SoftCCALoss):
                train_metrics["mean_projection_l2_norm"] = with_accessor(
                    WindowedMean(),
                    lambda metric, outputs, _: metric.update(outputs["l2"]),
                )
                train_metrics["mean_projection_sdl1"] = with_accessor(
                    WindowedMean(),
                    lambda metric, outputs, _: metric.update(outputs["sdl1"]),
                )
                train_metrics["mean_projection_sdl2"] = with_accessor(
                    WindowedMean(),
                    lambda metric, outputs, _: metric.update(outputs["sdl2"]),
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
        loss: losses.AlwaysStaticShortContextual,
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
