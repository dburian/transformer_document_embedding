from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from datasets import Dataset
from torch.utils.data import DataLoader
from transformer_document_embedding.datasets import col
from transformer_document_embedding.datasets.document_dataset import EvaluationKind
from transformer_document_embedding.pipelines.helpers import classification_metrics
from transformer_document_embedding.torch_trainer import MetricLogger, TorchTrainer
from transformer_document_embedding.pipelines.classification_eval import smart_unbatch
import transformer_document_embedding.utils.training as train_utils

from transformer_document_embedding.pipelines.pipeline import TrainPipeline
from transformer_document_embedding.utils.metrics import TrainingMetric, VMemMetric
import torch

if TYPE_CHECKING:
    from torcheval.metrics import (
        Metric,
    )
    from transformer_document_embedding.datasets.document_dataset import DocumentDataset
    from transformer_document_embedding.models.embedding_model import EmbeddingModel


def get_default_features(split: Dataset, model: EmbeddingModel, **kwargs) -> Dataset:
    def gen_embeds():
        split_columns = split.remove_columns(list(col.IDS & set(split.column_names)))
        for batch, embedding in zip(
            split_columns,
            smart_unbatch(model.predict_embeddings(split, **kwargs), 1),
            strict=True,
        ):
            batch[col.EMBEDDING] = embedding
            yield batch

    return Dataset.from_generator(gen_embeds)


def get_pair_bin_cls_features(
    split: Dataset, model: EmbeddingModel, **kwargs
) -> Dataset:
    def gen_embeds():
        split_columns = split.remove_columns(list(col.IDS & set(split.column_names)))

        embeds_0 = split_columns.rename_column(col.TEXT_0, col.TEXT)
        embeds_1 = split_columns.rename_column(col.TEXT_1, col.TEXT)
        for doc, embed_0, embed_1 in zip(
            split_columns,
            smart_unbatch(model.predict_embeddings(embeds_0, **kwargs), 1),
            smart_unbatch(model.predict_embeddings(embeds_1, **kwargs), 1),
            strict=True,
        ):
            doc[col.EMBEDDING] = torch.concat((embed_0, embed_1), 0)
            yield doc

    return Dataset.from_generator(gen_embeds)


def get_head_features(
    eval_kind: EvaluationKind,
    split: Dataset,
    model: EmbeddingModel,
    **kwargs,
) -> Dataset:
    get_features = {
        EvaluationKind.CLAS: get_default_features,
        EvaluationKind.PAIR_CLAS: get_pair_bin_cls_features,
    }

    return get_features[eval_kind](split, model, **kwargs)


@dataclass(kw_only=True)
class GenericTorchFinetune(TrainPipeline):
    epochs: int
    batch_size: int

    weight_decay: float
    lr: float
    lr_scheduler_type: str

    warmup_steps: int
    fp16: bool
    grad_accumulation_steps: int
    max_grad_norm: float

    log_every_step: int
    validate_every_step: Optional[int]

    patience: Optional[int]
    # TODO: Load the best model after?
    save_best: bool
    save_after_steps: Optional[int] = None

    embed_pred_batch_size: int

    def get_features(self, split: Dataset, model: EmbeddingModel) -> Dataset:
        return get_default_features(split, model, batch_size=self.embed_pred_batch_size)

    def to_dataloader(
        self, split: Dataset, model: EmbeddingModel, training: bool = True
    ) -> DataLoader[torch.Tensor]:
        embeddings = (
            split
            if col.EMBEDDING in split.column_names
            else self.get_features(split, model)
        )
        return DataLoader(
            embeddings.with_format("torch"),
            batch_size=self.batch_size,
            shuffle=training,
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

    def __call__(
        self,
        model: EmbeddingModel,
        head: torch.nn.Module,
        dataset: DocumentDataset,
        log_dir: Optional[str],
    ) -> None:
        train_batches = self.to_dataloader(dataset.splits["train"], model)
        val_batches = None
        if (val_split := dataset.splits.get("validation", None)) is not None:
            val_batches = self.to_dataloader(val_split, model, training=False)

        optimizer = torch.optim.AdamW(
            train_utils.get_optimizer_params(head, self.weight_decay),
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

        train_logger, val_logger = self.get_train_val_loggers(
            log_dir, self.get_train_metrics(self.log_every_step, head)
        )

        trainer = TorchTrainer(
            model=head,
            optimizer=optimizer,
            train_logger=train_logger,
            val_logger=val_logger,
            fp16=self.fp16,
            max_grad_norm=self.max_grad_norm,
            grad_accumulation_steps=self.grad_accumulation_steps,
            lr_scheduler=lr_scheduler,
            validate_every_step=self.validate_every_step,
            save_model_callback=None,
            patience=self.patience,
        )
        trainer.train(
            epochs=self.epochs,
            train_data=train_batches,
            val_data=val_batches,
        )


class ClassificationFinetune(GenericTorchFinetune):
    def __call__(
        self,
        model: EmbeddingModel,
        head: torch.nn.Module,
        dataset: DocumentDataset,
        log_dir: Optional[str],
    ) -> None:
        self.num_classes = len(dataset.splits["test"].unique(col.LABEL))
        return super().__call__(model, head, dataset, log_dir)

    def get_train_metrics(
        self, log_freq: int, model: torch.nn.Module
    ) -> list[TrainingMetric]:
        def update_with_logits(
            metric: Metric,
            outputs: dict[str, torch.Tensor],
            batch: dict[str, torch.Tensor],
        ) -> None:
            metric.update(torch.argmax(outputs["logits"], dim=1), batch[col.LABEL])

        return super().get_train_metrics(log_freq, model) + [
            TrainingMetric(
                name,
                metric,
                log_freq,
                update_with_logits,
            )
            for name, metric in classification_metrics(self.num_classes).items()
        ]


class PairClassificationFinetune(ClassificationFinetune):
    def get_features(self, split: Dataset, model: EmbeddingModel) -> Dataset:
        return get_pair_bin_cls_features(
            split, model, batch_size=self.embed_pred_batch_size
        )
