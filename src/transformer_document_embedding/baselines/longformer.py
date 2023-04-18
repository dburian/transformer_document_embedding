import math
import os
from typing import Any, Iterable, Optional, cast

import numpy as np
import torch
from datasets.arrow_dataset import Dataset
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torcheval.metrics import Mean, Metric, MulticlassAccuracy
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedModel
from transformers.data.data_collator import DataCollatorWithPadding

from transformer_document_embedding.baselines import ExperimentalModel
from transformer_document_embedding.models.longformer import (
    LongformerConfig, LongformerForSequenceClassification)
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

        train_data = self._prepare_data(task.train)

        val_data = None
        val_summary_writer = None
        summary_writer = None

        if task.validation is not None:
            val_data = self._prepare_data(task.validation)

            if log_dir is not None:
                val_summary_writer = SummaryWriter(os.path.join(log_dir, "val"))

        if log_dir is not None:
            summary_writer = SummaryWriter(os.path.join(log_dir, "train"))

        optimizer = torch.optim.Adam(self._model.parameters(), lr=3e-5)

        self._model.gradient_checkpointing_enable()
        train(
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
            lr_scheduler=get_linear_lr_scheduler_with_warmup(
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

    def predict(self, inputs: Dataset) -> Iterable[np.ndarray]:
        self._model.eval()

        data = self._prepare_data(inputs, training=False)
        for batch in data:
            batch_to_device(batch, self._model.device)
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

    def _prepare_data(self, data: Dataset, training: bool = True) -> DataLoader:
        def _tokenize(doc: dict[str, Any]) -> dict[str, Any]:
            return self._tokenizer(doc["text"], padding=False, truncation=True)

        data = data.map(_tokenize, batched=True)
        data = data.with_format("torch")
        data = data.remove_columns(["text", "id"])

        collator = DataCollatorWithPadding(tokenizer=self._tokenizer, padding="longest")
        dataloader = DataLoader(
            data,
            batch_size=self._batch_size,
            shuffle=training,
            collate_fn=collator,
        )
        return dataloader


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> None:
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)


def get_linear_lr_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> LambdaLR:
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step)
            / float(max(1, total_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


def training_step(
    *,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    metrics: dict[str, Metric],
    loss_metric: Metric,
    fp16: bool = False,
    grad_accumulation_steps: int = 1,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    scaler: torch.cuda.amp.grad_scaler.GradScaler,
    steps_in_epoch: int,
    lr_scheduler: Optional[object] = None,
    max_grad_norm: Optional[float] = None,
    step: int,
) -> None:
    assert (
        scaler is not None or not fp16
    ), f"{training_step.__name__}: scaler must be set for fp16."

    device = torch.device("cuda")
    batch_to_device(batch, device)

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
        outputs = model(**batch)
        loss = loss_fn(outputs["logits"], batch["labels"]) / grad_accumulation_steps

    loss_metric.update(loss * grad_accumulation_steps)
    for metric in metrics.values():
        metric.update(outputs["logits"], batch["labels"])

    loss = scaler.scale(loss) if fp16 else loss
    loss.backward()

    if (step + 1) % grad_accumulation_steps == 0 or (step + 1) == steps_in_epoch:
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer_was_run = True
        if fp16:
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scale_after = scaler.get_scale()
            optimizer_was_run = scale_before <= scale_after

            scaler.update()
        else:
            optimizer.step()

        if optimizer_was_run and lr_scheduler is not None:
            lr_scheduler.step()

        # Or model.zero_grad()?
        optimizer.zero_grad()


def validation_step(
    *,
    model: PreTrainedModel,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    fp16: bool,
    loss_fn: torch.nn.Module,
    metrics: dict[str, Metric],
    mean_val_loss: Metric,
) -> None:
    batch_to_device(batch, device)

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = loss_fn(outputs["logits"], batch["labels"])

    mean_val_loss.update(loss)
    for metric in metrics.values():
        metric.update(outputs["logits"], batch["labels"])


def compute_optimizer_step(
    *,
    epoch: int,
    step: int,
    steps_in_epoch: int,
    grad_accumulation_steps: int,
) -> int:
    batch_steps = (epoch + 1) * steps_in_epoch + step
    return math.floor(batch_steps / grad_accumulation_steps)


def log_and_reset_metrics(
    *,
    metrics: dict[str, Metric],
    summary_writer: SummaryWriter,
    step: int,
    loss_metric: Metric,
) -> None:
    summary_writer.add_scalar("loss", loss_metric.compute(), step)
    loss_metric.reset()
    for name, metric in metrics.items():
        summary_writer.add_scalar(name, metric.compute(), step)
        metric.reset()

    summary_writer.flush()


def train(
    *,
    model: PreTrainedModel,
    train_data: DataLoader,
    val_data: Optional[DataLoader] = None,
    epochs: int,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, Metric],
    summary_writer: Optional[SummaryWriter] = None,
    val_summary_writer: Optional[SummaryWriter] = None,
    fp16: bool = False,
    max_grad_norm: Optional[float] = None,
    grad_accumulation_steps: int = 1,
    steps_in_epoch: Optional[int] = None,
    lr_scheduler=None,
    progress_bar: bool = True,
    validate_every: int = 1,
    checkpoint_path: Optional[str] = None,
    patience: Optional[int] = None,
) -> None:
    device = torch.device("cuda")
    model.to(device)
    min_val_loss = float("inf")
    epochs_without_improvement = 0

    steps_in_epoch = steps_in_epoch if steps_in_epoch is not None else len(train_data)

    scaler = GradScaler()

    for name, metric in metrics.items():
        metrics[name] = metric.to(device)
    loss_metric = Mean(device=device)

    for epoch in tqdm(range(epochs), desc="Epoch", disable=not progress_bar):
        for i, batch in tqdm(
            enumerate(train_data),
            desc="Iteration",
            disable=not progress_bar,
            total=steps_in_epoch,
        ):
            training_step(
                model=model,
                optimizer=optimizer,
                batch=batch,
                metrics=metrics,
                loss_metric=loss_metric,
                fp16=fp16,
                grad_accumulation_steps=grad_accumulation_steps,
                loss_fn=loss_fn,
                scaler=scaler,
                steps_in_epoch=steps_in_epoch,
                step=i,
                max_grad_norm=max_grad_norm,
                lr_scheduler=lr_scheduler,
            )

        if summary_writer is not None:
            log_and_reset_metrics(
                metrics=metrics,
                summary_writer=summary_writer,
                step=compute_optimizer_step(
                    epoch=epoch,
                    step=0,
                    steps_in_epoch=steps_in_epoch,
                    grad_accumulation_steps=grad_accumulation_steps,
                ),
                loss_metric=loss_metric,
            )

        if val_data is not None and (1 + epoch) % validate_every == 0:
            model.eval()
            mean_val_loss = Mean(device=device)

            for batch in val_data:
                validation_step(
                    model=model,
                    batch=batch,
                    device=device,
                    fp16=fp16,
                    loss_fn=loss_fn,
                    metrics=metrics,
                    mean_val_loss=mean_val_loss,
                )

            val_loss = mean_val_loss.compute()
            if val_summary_writer is not None:
                log_and_reset_metrics(
                    metrics=metrics,
                    summary_writer=val_summary_writer,
                    step=compute_optimizer_step(
                        epoch=epoch,
                        step=0,
                        steps_in_epoch=steps_in_epoch,
                        grad_accumulation_steps=grad_accumulation_steps,
                    ),
                    loss_metric=mean_val_loss,
                )

            if val_loss < min_val_loss:
                if checkpoint_path is not None:
                    model.save_pretrained(checkpoint_path)
                    min_val_loss = val_loss

                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if patience is not None and epochs_without_improvement == patience:
                break

            model.train()
