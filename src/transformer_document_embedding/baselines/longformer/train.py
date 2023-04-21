import math
from typing import Any, Optional

import datasets
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torcheval.metrics import Mean, Metric
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.trainer_pt_utils import get_parameter_names


def prepare_data(
    data: datasets.Dataset,
    *,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    training: bool = True,
) -> DataLoader:
    def _tokenize(doc: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(doc["text"], padding=False, truncation=True)

    data = data.map(_tokenize, batched=True)
    data = data.with_format("torch")
    data = data.remove_columns(["text", "id"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=training,
        collate_fn=collator,
    )
    return dataloader


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> None:
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)


def get_optimizer_params(
    model: torch.nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if n not in decay_parameters
            ],
            "weight_decay": 0.0,
        },
    ]


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
