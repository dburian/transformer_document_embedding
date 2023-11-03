from __future__ import annotations
import os
import torch
import logging

from typing import Any, cast, TYPE_CHECKING
from torch.utils.tensorboard.writer import SummaryWriter
from torcheval import metrics
from transformers import AutoTokenizer, PreTrainedModel
from transformer_document_embedding.baselines.baseline import Baseline
import transformer_document_embedding.utils.torch.losses as losses
import transformer_document_embedding.utils.torch.training as train_utils
import transformer_document_embedding.baselines.longformer.train as longformer_training
from transformer_document_embedding.models.longformer import (
    LongformerConfig,
    LongformerForTextEmbedding,
)
from transformer_document_embedding.utils.metrics import with_accessor, VMemMetric
from tqdm.auto import tqdm
import numpy as np

if TYPE_CHECKING:
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
        use_dcca: bool,
        cca_output_dim: Optional[int] = None,
        dcca_norm: Optional[str] = None,
        dcca_model_layers: Optional[list[int]] = None,
        dcca_static_layers: Optional[list[int]] = None,
    ) -> None:
        self._batch_size = batch_size
        self._contextual_max_length = contextual_max_length

        model_path = f"allenai/longformer-{'large' if large else 'base'}-4096"
        config = LongformerConfig(
            pooler_type=pooler_type,
            **LongformerConfig.get_config_dict(model_path)[0],
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        transformer = cast(
            LongformerForTextEmbedding,
            LongformerForTextEmbedding.from_pretrained(model_path, config=config),
        )

        model_output_features = transformer.config.hidden_size
        if use_dcca:
            assert (
                dcca_model_layers is not None
                and dcca_static_layers is not None
                and dcca_norm is not None
                and cca_output_dim is not None
            ), "To use DCCA all DCCA-related parameters must be set."

            static_loss = losses.DCCALoss(
                net1=losses.DeepNet(
                    layer_features=dcca_model_layers,
                    input_features=model_output_features,
                    norm=dcca_norm,
                ),
                net2=losses.DeepNet(
                    layer_features=dcca_static_layers,
                    input_features=static_embed_dim,
                    norm=dcca_norm,
                ),
                cca_loss=losses.RunningCCALoss(
                    view1_dimension=dcca_model_layers[-1]
                    if len(dcca_model_layers) > 0
                    else model_output_features,
                    view2_dimension=dcca_static_layers[-1]
                    if len(dcca_static_layers) > 0
                    else static_embed_dim,
                    output_dimension=cca_output_dim,
                ),
            )
        else:
            static_loss = losses.ProjectedMSE(
                input_features=model_output_features,
                output_features=static_embed_dim,
                loss=torch.nn.functional.mse_loss,
            )

        loss = losses.AlwaysStaticShortContextual(
            contextual_key="sbert",
            static_key="dbow",
            static_loss=static_loss,
            lambda_=static_contextual_lambda,
            len_threshold=contextual_max_length,
        )

        self._model = TransformerWithTrainableLoss(transformer, loss)

    def train(
        self,
        task: ExperimentalTask,
        dataloader_sampling: str,
        grad_accumulation_steps: int,
        short_inputs_in_effective_batch: int,
        learning_rate: float,
        weight_decay: float,
        warmup_steps: int,
        epochs: int,
        early_stopping: bool,
        log_every_step: int,
        log_dir: Optional[str] = None,
        **_,
    ) -> None:
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
            train_utils.get_optimizer_params(self._model, weight_decay),
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

        def log_grad_norm(metric, outputs, _):
            def log_(grad):
                metric.update(torch.linalg.vector_norm(grad, dim=1))

            outputs["projected_view1"].register_hook(log_)

        def update_with_grad_norm(metric, grad) -> None:
            metric.update(torch.linalg.vector_norm(grad.detach(), dim=1))

        true_dbow_sigma = torch.from_numpy(
            np.load("notebooks/wiki_sample_dbow_sigma.npy")
        ).to(torch.device("cuda"))

        self._model.transformer.gradient_checkpointing_enable()
        longformer_training.train(
            model=self._model,
            train_data=train_data,
            epochs=epochs,
            optimizer=optimizer,
            metrics={
                "used_vmem": VMemMetric(),
                "mean_contextual_loss": with_accessor(
                    metrics.Mean(),
                    lambda metric, outputs, _: metric.update(
                        outputs["contextual_loss"]
                    ),
                ),
                "mean_static_loss": with_accessor(
                    metrics.Mean(),
                    lambda metric, outputs, _: metric.update(outputs["static_loss"]),
                ),
                "mean_length": with_accessor(
                    metrics.Mean(),
                    lambda metric, _, batch: metric.update(batch["length"]),
                ),
                # "mean_view1_grad_norm": with_accessor(
                #     metrics.Mean(),
                #     log_grad_norm,
                # ),
                # "mean_view2_grad_norm": with_accessor(
                #     metrics.Mean(),
                #     lambda metric, outputs, _: metric.update(
                #         torch.linalg.vector_norm(
                #            outputs["projected_view2"].grad, dim=1
                #       )
                #     ),
                # ),
                "mean_cov_mat_sum": with_accessor(
                    metrics.Mean(),
                    lambda metric, outputs, _: metric.update(
                        outputs["covariance_mat"].sum()
                    ),
                ),
                "mse_dbow_sigma": with_accessor(
                    metrics.Mean(),
                    lambda metric, outputs, _: metric.update(
                        torch.abs(outputs["sigma2"] - true_dbow_sigma).sum()
                    ),
                ),
            },
            summary_writer=summary_writer,
            fp16=True,
            max_grad_norm=1.0,
            grad_accumulation_steps=grad_accumulation_steps,
            lr_scheduler=lr_scheduler,
            patience=3 if early_stopping else None,
            checkpoint_path=None,
            log_every_step=log_every_step,
        )

    @torch.no_grad()
    def predict(self, inputs: Dataset) -> Iterable[np.ndarray]:
        self._model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Predicting using {device}")
        self._model.to(device)

        data = train_utils.create_tokenized_data_loader(
            inputs,
            tokenizer=self._tokenizer,
            batch_size=self._batch_size,
            training=False,
        )
        for batch in tqdm(data, desc="Predicting batches"):
            train_utils.batch_to_device(batch, self._model.device)
            embeddings = self._model(**batch).pooler_output
            yield embeddings.numpy(force=True)

    def save(self, dir_path: str) -> None:
        self._model.save_pretrained(dir_path)

    def load(self, dir_path: str) -> None:
        self._model = cast(
            TransformerWithTrainableLoss,
            TransformerWithTrainableLoss.from_pretrained(dir_path),
        )


# TODO: Not actually a pre-trained model
class TransformerWithTrainableLoss(torch.nn.Module):
    def __init__(
        self, transformer: PreTrainedModel, loss: torch.nn.Module, *args, **kwargs
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

    def predit(self, **batch) -> dict[str, Any]:
        return self.transformer(**batch)

    @classmethod
    def from_pretrained(cls, path: str) -> TransformerWithTrainableLoss:
        raise NotImplementedError()

    def save_pretrained(self, path: str) -> None:
        pass
