import math
import os
from typing import Iterable, Optional, cast

import numpy as np
import torch
from datasets.arrow_dataset import Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from torcheval.metrics import MulticlassAccuracy
from transformers import AutoTokenizer

import transformer_document_embedding.utils.torch.training as train_utils
from transformer_document_embedding.baselines import ExperimentalModel
from transformer_document_embedding.models.longformer import (
    LongformerConfig,
    LongformerForSequenceClassification,
)
from transformer_document_embedding.tasks.imdb import IMDBClassification
from transformer_document_embedding.utils.metrics import VMemMetric


class LongformerIMDB(ExperimentalModel):
    def __init__(
        self,
        *,
        large: bool = False,
        epochs: int = 10,
        label_smoothing: float = 0.1,
        warmup_steps: int = math.floor(0.1 * 25000),
        weight_decay: float = 0.0,
        batch_size: int = 1,
        pooler_type: Optional[str] = None,
        classifier_dropout: Optional[float] = None,
        classifier_activation: Optional[str] = None,
        classifier_dim: Optional[int] = None,
    ) -> None:
        model_path = f"allenai/longformer-{'large' if large else 'base'}-4096"
        self._epochs = epochs
        self._label_smoothing = label_smoothing
        self._warmup_steps = warmup_steps
        self._batch_size = batch_size
        self._weight_decay = weight_decay

        config = LongformerConfig(
            classifier_dropout_prob=classifier_dropout,
            classifier_activation=classifier_activation,
            classifier_hidden_size=classifier_dim,
            pooler_type=pooler_type,
            num_labels=2,
            **LongformerConfig.get_config_dict(model_path)[0],
        )

        self._model = cast(
            LongformerForSequenceClassification,
            LongformerForSequenceClassification.from_pretrained(
                model_path,
                config=config,
            ),
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

    def train(
        self,
        task: IMDBClassification,
        *,
        early_stopping: bool = False,
        save_best_path: Optional[str] = None,
        log_dir: Optional[str] = None,
    ) -> None:
        assert (
            torch.cuda.is_available()
        ), f"Training {LongformerIMDB.__name__} is available only with gpu."

        train_data = train_utils.create_tokenized_data_loader(
            task.train, tokenizer=self._tokenizer, batch_size=self._batch_size
        )

        val_data = None
        val_summary_writer = None
        summary_writer = None

        if task.validation is not None:
            val_data = train_utils.create_tokenized_data_loader(
                task.validation,
                tokenizer=self._tokenizer,
                batch_size=self._batch_size,
            )

            if log_dir is not None:
                val_summary_writer = SummaryWriter(os.path.join(log_dir, "val"))

        if log_dir is not None:
            summary_writer = SummaryWriter(os.path.join(log_dir, "train"))

        optimizer = torch.optim.AdamW(
            train_utils.get_optimizer_params(self._model, self._weight_decay), lr=3e-5
        )

        self._model.gradient_checkpointing_enable()
        train_utils.train(
            model=self._model,
            train_data=train_data,
            val_data=val_data,
            epochs=self._epochs,
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=optimizer,
            metrics={
                "accuracy": MulticlassAccuracy(),
                "used_vmem": VMemMetric(),
            },
            summary_writer=summary_writer,
            val_summary_writer=val_summary_writer,
            fp16=True,
            max_grad_norm=1.0,
            grad_accumulation_steps=max(1, int(32 / self._batch_size)),
            lr_scheduler=train_utils.get_linear_lr_scheduler_with_warmup(
                optimizer,
                self._warmup_steps,
                self._epochs * len(train_data),
            ),
            patience=3 if early_stopping else None,
            checkpoint_path=save_best_path,
        )

        if save_best_path is not None:
            if val_data is None:
                # We have not saved anything
                self.save(save_best_path)
            else:
                # We have saved at least something, lets restore it
                self.load(save_best_path)

    @torch.no_grad()
    def predict(self, inputs: Dataset) -> Iterable[np.ndarray]:
        self._model.eval()

        data = train_utils.create_tokenized_data_loader(
            inputs,
            tokenizer=self._tokenizer,
            batch_size=self._batch_size,
            training=False,
        )
        for batch in data:
            train_utils.batch_to_device(batch, self._model.device)
            logits = self._model(**batch)["logits"]
            preds = torch.argmax(logits, 1)
            yield preds.numpy(force=True)

    def save(self, dir_path: str) -> None:
        self._model.save_pretrained(dir_path)

    def load(self, dir_path: str) -> None:
        self._model = cast(
            LongformerForSequenceClassification,
            LongformerForSequenceClassification.from_pretrained(dir_path),
        )
