from math import floor
import os
from typing import Any, Iterable, Optional

import torch
import datasets
import numpy as np
from torcheval.metrics import (
    Metric,
    MulticlassAUPRC,
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
)
from tqdm.auto import tqdm
from transformer_document_embedding.models.cls_head import ClsHead
from transformer_document_embedding.models.paragraph_vector.paragraph_vector import (
    ParagraphVector,
)

from transformer_document_embedding.models.trainer import MetricLogger, TorchTrainer
from transformer_document_embedding.tasks.experimental_task import ExperimentalTask
from transformer_document_embedding.utils.metrics import TrainingMetric, VMemMetric
from transformer_document_embedding.utils.net_helpers import (
    load_model_weights,
    save_model_weights,
)
import transformer_document_embedding.utils.training as train_utils
from torch.utils.data import DataLoader

from transformer_document_embedding.utils.gensim import (
    IterableFeaturesDataset,
)


class ParagraphVectorClassifier(ParagraphVector):
    def __init__(
        self,
        dm_kwargs: Optional[dict[str, Any]],
        dbow_kwargs: Optional[dict[str, Any]],
        pre_process: Optional[str],
        cls_head_kwargs: dict[str, Any],
        label_smoothing: float,
        batch_size: int,
    ) -> None:
        super().__init__(dm_kwargs, dbow_kwargs, pre_process)

        cls_head = ClsHead(
            **cls_head_kwargs,
            in_features=self._pv.vector_size,
            out_features=2,
        )
        loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self._model = _SequenceClassificationModel(
            cls_head,
            loss,
        )

        self._batch_size = batch_size

    def train(
        self,
        task: ExperimentalTask,
        start_at_epoch: Optional[int],
        save_at_epochs: Optional[list[int]],
        cls_epochs: int,
        log_every_step: int,
        validate_every_step: int,
        log_dir: Optional[str] = None,
        **_,
    ) -> None:
        super().train(task, start_at_epoch, save_at_epochs, log_dir, **_)

        train_batches = self._feature_dataloader(task.splits["train"], training=True)
        val_batches = None
        if (val_split := task.splits.get("validation", None)) is not None:
            print("validation exists")
            val_batches = self._feature_dataloader(val_split, training=False)

        optimizer = torch.optim.Adam(
            train_utils.get_optimizer_params(self._model, 0),
            lr=1e-3,
        )

        total_steps = cls_epochs * len(train_batches)
        lr_scheduler = train_utils.get_lr_scheduler(
            scheduler_type="cos",
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=floor(0.1 * total_steps),
        )

        train_logger, val_logger = None, None
        if log_dir is not None:
            train_metrics = self._get_train_metrics(log_every_step)
            train_logger = MetricLogger("train", train_metrics, log_dir)

            val_metrics = [m.clone() for m in train_metrics]
            val_logger = MetricLogger("val", val_metrics, log_dir, log_lr=False)

        trainer = TorchTrainer(
            model=self._model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_logger=train_logger,
            val_logger=val_logger,
            validate_every_step=validate_every_step,
        )
        trainer.train(cls_epochs, train_batches, val_batches)

    def _get_train_metrics(self, log_freq: int) -> list[TrainingMetric]:
        def update_with_logits(
            metric: Metric,
            outputs: dict[str, torch.Tensor],
            batch: dict[str, torch.Tensor],
        ) -> None:
            metric.update(outputs["logits"], batch["labels"])

        return [
            TrainingMetric(
                "accuracy", MulticlassAccuracy(), log_freq, update_with_logits
            ),
            TrainingMetric("recall", MulticlassRecall(), log_freq, update_with_logits),
            TrainingMetric(
                "precision",
                MulticlassPrecision(),
                log_freq,
                update_with_logits,
            ),
            TrainingMetric(
                "auprc",
                MulticlassAUPRC(num_classes=2),
                log_freq,
                update_with_logits,
            ),
            VMemMetric(log_freq),
        ]

    @torch.inference_mode()
    def predict(self, inputs: datasets.Dataset) -> Iterable[np.ndarray]:
        features_batches = self._feature_dataloader(inputs, training=False)

        self._model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)

        for batch in tqdm(
            features_batches, desc="Evaluation batches", total=len(features_batches)
        ):
            train_utils.batch_to_device(batch, device)
            outputs = self._model(embeddings=batch["embeddings"])
            yield torch.argmax(outputs["logits"], dim=1).numpy(force=True)

    def save(self, dir_path: str) -> None:
        self._pv.save(self._pv_dirpath(dir_path))
        save_model_weights(self._model, self._cls_head_filepath(dir_path))

    def load(self, dir_path: str, *, strict: bool) -> None:
        self._pv.load(self._pv_dirpath(dir_path))
        load_model_weights(
            self._model, self._cls_head_filepath(dir_path), strict=strict
        )

    @classmethod
    def _pv_dirpath(cls, dir_path: str) -> str:
        new_dir = os.path.join(dir_path, "paragraph_vector")
        os.makedirs(new_dir, exist_ok=True)
        return new_dir

    @classmethod
    def _cls_head_filepath(cls, dir_path: str) -> str:
        return os.path.join(dir_path, "cls_head")

    def _feature_dataloader(self, data: datasets.Dataset, training: bool) -> DataLoader:
        features_dataset = IterableFeaturesDataset(
            data.shuffle() if training else data,
            text_pre_processor=self._text_pre_processor,
            pv=self._pv,
            lookup_vectors=training,
        )

        return DataLoader(
            features_dataset,
            batch_size=self._batch_size,
        )


class _SequenceClassificationModel(torch.nn.Module):
    def __init__(
        self, cls_head: torch.nn.Module, loss: torch.nn.Module, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.cls_head = cls_head
        self.loss = loss

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        logits = self.cls_head(embeddings)

        outputs = {"logits": logits}
        if labels is not None:
            outputs["loss"] = self.loss(logits, labels)

        return outputs
