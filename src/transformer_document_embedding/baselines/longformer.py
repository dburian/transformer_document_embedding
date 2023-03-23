import math
import numpy as np
from typing import Optional, Any, Iterable, cast
from torch.optim.lr_scheduler import LambdaLR
from transformer_document_embedding.baselines import ExperimentalModel
from transformers import (
    AutoTokenizer,
    LongformerConfig,
    LongformerForSequenceClassification,
)

from transformer_document_embedding.tasks.imdb import IMDBClassification, IMDBData
from transformers.data.data_collator import DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from troch.cuda.amp import GradScaler
from torcheval.metrics import Metric, MulticlassAccuracy, Mean
from tqdm import tqdm


class LongformerIMBD(ExperimentalModel):
    def __init__(
        self,
        *,
        epochs: int = 10,
        label_smoothing: float = 0.1,
        warmup_steps: int = math.floor(0.1 * 25000),
        batch_size: int = 1,
        classifier_dropout: float = 0.1,
    ) -> None:
        self._epochs = epochs
        self._label_smoothing = label_smoothing
        self._warmup_steps = warmup_steps
        self._batch_size = batch_size

        config = LongformerConfig(
            num_labels=2,
            max_position_embeddings=4096,
            classifier_dropout=classifier_dropout,
        )
        self._model = cast(
            LongformerForSequenceClassification,
            LongformerForSequenceClassification.from_pretrained(
                "allenai/longformer-base-4096", config
            ),
        )
        self._tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

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
        ), f"Training {LongformerIMBD.__name__} is available only with gpu."

        train = self._prepare_data(task.train)
        val = None
        if hasattr(task, "validation"):
            # TODO: Better implementation of this
            #   - property on `ExperimentalTask`
            #   - default implementation of validation returning None
            val = self._prepare_data(task.validation)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=3e-5)

        self._model.to(torch.device("cuda"))
        self._train(
            train=train,
            val=val,
            epochs=self._epochs,
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=optimizer,
            metrics={
                "accuracy": MulticlassAccuracy(),
            },
            summary_writer=SummaryWriter(log_dir) if log_dir is not None else None,
            fp16=True,
            max_grad_norm=1.0,
            grad_accumulation_steps=16,
            lr_scheduler=get_linear_lr_scheduler_with_warmup(
                optimizer,
                self._warmup_steps,
                self._epochs * len(train),
            ),
            patience=3 if early_stopping else None,
            checkpoint_path=save_best_path,
        )

    def predict(self, inputs: IMDBData) -> Iterable[np.ndarray]:
        self._model.eval()

        data = self._prepare_data(inputs, training=False)
        return self._model(data)

    def save(self, dir_path: str) -> None:
        self._model.save_pretrained(dir_path)

    def load(self, dir_path: str) -> None:
        self._model = cast(
            LongformerForSequenceClassification,
            LongformerForSequenceClassification.from_pretrained(dir_path),
        )

    def _train(
        self,
        *,
        train: DataLoader,
        val: Optional[DataLoader] = None,
        epochs: int,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: dict[str, Metric],
        summary_writer: Optional[SummaryWriter] = None,
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
        model = self._model
        device = model.device
        min_val_loss = float("inf")
        epochs_without_improvement = 0

        steps_in_epoch = steps_in_epoch if steps_in_epoch is not None else len(train)

        scaler = GradScaler()

        for name, metric in metrics.items():
            metrics[name] = metric.to(device)
        loss_metric = Mean(device=device)

        for epoch in tqdm(range(epochs), desc="Epoch", disable=not progress_bar):
            for i, batch in tqdm(
                enumerate(train),
                desc="Iteration",
                disable=not progress_bar,
                total=steps_in_epoch,
            ):
                batch_to_device(batch, device)

                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=fp16
                ):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    loss = (
                        loss_fn(outputs["logits"], batch["labels"])
                        / grad_accumulation_steps
                    )

                    loss_metric.update(loss * grad_accumulation_steps)

                for metric in metrics.values():
                    metric.update(outputs["logits"], batch["labels"])

                loss = scaler.scale(loss) if fp16 else loss
                loss.backward()

                if (i + 1) % grad_accumulation_steps == 0 or (i + 1) == steps_in_epoch:
                    if max_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_grad_norm
                        )

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

            if summary_writer is not None:
                summary_writer.add_scalar("loss", loss_metric.compute(), epoch)
                loss_metric.reset()
                for name, metric in metrics.items():
                    summary_writer.add_scalar(name, metric.compute(), epoch)
                    metric.reset()

            if val is not None and (1 + epoch) % validate_every == 0:
                model.eval()

                for batch in val:
                    batch_to_device(batch, device)

                    with torch.autocast(
                        device_type=device.type, dtype=torch.float16, enabled=fp16
                    ):
                        with torch.no_grad():
                            outputs = model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                            )
                            loss = loss_fn(outputs["logits"], batch["labels"])
                    loss_metric.update(loss * grad_accumulation_steps)

                    for metric in metrics.values():
                        metric.update(outputs["logits"], batch["labels"])

                val_loss = loss_metric.compute()
                if val_loss < min_val_loss:
                    if checkpoint_path is not None:
                        model.save_pretrained(checkpoint_path)
                        min_val_loss = val_loss

                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if patience is not None and epochs_without_improvement == patience:
                    break

                if summary_writer is not None:
                    summary_writer.add_scalar("val_loss", val_loss, epoch)
                    loss_metric.reset()
                    for name, metric in metrics.items():
                        summary_writer.add_scalar(
                            f"val_{name}", metric.compute(), epoch
                        )
                        metric.reset()

                model.train()

    def _prepare_data(self, data: IMDBData, training: bool = True) -> DataLoader:
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
