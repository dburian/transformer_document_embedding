from __future__ import annotations
from dataclasses import dataclass
from os import path
from typing import Any, Callable, Optional, TYPE_CHECKING
import torch
from transformers import AutoTokenizer
from transformer_document_embedding.torch_trainer import MetricLogger, TorchTrainer

from transformer_document_embedding.pipelines.pipeline import TrainPipeline
from transformer_document_embedding.utils.metrics import VMemMetric
from transformer_document_embedding.utils.net_helpers import save_model_weights
from transformer_document_embedding.utils.tokenizers import (
    create_tokenized_data_loader,
)
import transformer_document_embedding.utils.training as train_utils

from transformer_document_embedding.utils.metrics import TrainingMetric

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from transformer_document_embedding.models.embedding_model import EmbeddingModel
    from datasets import Dataset
    from transformer_document_embedding.datasets.document_dataset import DocumentDataset
    from transformer_document_embedding.models.transformer import (
        TransformerEmbedder,
    )


@dataclass(kw_only=True)
class TorchTrainPipeline(TrainPipeline):
    batch_size: int
    epochs: int

    weight_decay: float
    lr: float
    lr_scheduler_type: str

    warmup_steps: int
    fp16: bool
    grad_accumulation_steps: int
    max_grad_norm: float

    log_every_step: int
    validate_every_step: Optional[int]

    dataloader_sampling: str
    sampler_kwargs: Optional[dict[str, Any]]

    global_attention_type: str

    patience: Optional[int]
    save_best: bool
    save_after_steps: Optional[list[int]] = None

    # TODO: This custom interface isn't good. Solutions:
    # - fully accept that models will be dumb and pipelines will be sort of an
    # interface (which isn't that bad)
    # - ?
    def to_dataloader(
        self, split: Dataset, model: TransformerEmbedder, training: bool = True
    ) -> DataLoader:
        return create_tokenized_data_loader(
            split,
            batch_size=self.batch_size,
            training=training,
            min_length=model.min_sequence_length,
            pad_to_multiple_of=model.pad_to_multiple_of,
            tokenizer=AutoTokenizer.from_pretrained(model.transformer_name),
            sampling=self.dataloader_sampling,
            global_attention_type=self.global_attention_type,
            return_length=False,
            sampler_kwargs={
                "effective_batch_size": self.batch_size * self.grad_accumulation_steps,
                **({} if self.sampler_kwargs is None else self.sampler_kwargs),
            },
        )

    def get_train_val_loggers(
        self,
        log_dir: Optional[str],
        train_metrics: list[TrainingMetric],
        val_metrics: Optional[list[TrainingMetric]] = None,
    ) -> tuple[Optional[MetricLogger], Optional[MetricLogger]]:
        if log_dir is None:
            return None, None

        if val_metrics is None:
            val_metrics = [m.clone() for m in train_metrics]

        train_logger = MetricLogger("train", train_metrics, log_dir)
        val_logger = MetricLogger("val", val_metrics, log_dir, log_lr=False)

        return train_logger, val_logger

    def get_train_metrics(
        self, log_freq: int, model: torch.nn.Module
    ) -> list[TrainingMetric]:
        return [VMemMetric(log_freq)]

    def get_save_model_callback(
        self,
        saving_permitted: bool,
        model: EmbeddingModel,
        head: torch.nn.Module,
        log_dir: Optional[str],
    ) -> Optional[Callable[[torch.nn.Module, int], None]]:
        if not saving_permitted or log_dir is None:
            return None

        def _cb(_, total_steps: int) -> None:
            model_save_path = path.join(
                log_dir, "model_checkpoints", f"checkpoint_{total_steps}"
            )
            model.save_weights(model_save_path)
            save_model_weights(
                head,
                path.join(log_dir, "head_checkpoints", f"checkpoint_{total_steps}"),
            )

        return _cb

    def get_encompassing_model(
        self, model: TransformerEmbedder, _: torch.nn.Module
    ) -> torch.nn.Module:
        return model

    def __call__(
        self,
        encoder: TransformerEmbedder,
        head: torch.nn.Module,
        dataset: DocumentDataset,
        log_dir: Optional[str],
    ) -> None:
        save_model_callback = self.get_save_model_callback(
            self.save_best or (self.save_after_steps is not None),
            encoder,
            head,
            log_dir,
        )

        train_batches = self.to_dataloader(dataset.splits["train"], encoder)
        val_batches = None
        if (val_split := dataset.splits.get("validation", None)) is not None:
            val_batches = self.to_dataloader(val_split, encoder, training=False)

        model = self.get_encompassing_model(encoder, head)
        optimizer = torch.optim.AdamW(
            train_utils.get_optimizer_params(model, self.weight_decay),
            lr=self.lr,
        )

        total_steps = self.epochs * len(train_batches) // self.grad_accumulation_steps
        warmup_steps = self.warmup_steps // self.grad_accumulation_steps

        lr_scheduler = train_utils.get_lr_scheduler(
            scheduler_type=self.lr_scheduler_type,
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )

        if encoder.transformer.supports_gradient_checkpointing:
            encoder.transformer.gradient_checkpointing_enable()

        train_logger, val_logger = self.get_train_val_loggers(
            log_dir, self.get_train_metrics(self.log_every_step, model)
        )

        trainer = TorchTrainer(
            model=model,
            optimizer=optimizer,
            train_logger=train_logger,
            val_logger=val_logger,
            fp16=self.fp16,
            max_grad_norm=self.max_grad_norm,
            grad_accumulation_steps=self.grad_accumulation_steps,
            lr_scheduler=lr_scheduler,
            validate_every_step=self.validate_every_step,
            save_model_callback=save_model_callback,
            patience=self.patience,
        )
        trainer.train(
            epochs=self.epochs,
            train_data=train_batches,
            val_data=val_batches,
            save_after_steps=self.save_after_steps,
        )
