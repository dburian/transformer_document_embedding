from __future__ import annotations
from typing import TYPE_CHECKING

import logging
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torcheval.metrics import Mean, Metric
from tqdm.auto import tqdm

import transformer_document_embedding.utils.torch.training as train_utils

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard.writer import SummaryWriter
    from typing import Optional
    from typing import Any
    from typing import Callable

    DictGetter = Callable[[dict[str, torch.Tensor]], Any]


logger = logging.getLogger(__name__)


def training_step(
    *,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    metrics: dict[str, Metric],
    loss_metric: Metric,
    lr_metric: Metric,
    fp16: bool = False,
    grad_accumulation_steps: int = 1,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.grad_scaler.GradScaler,
    steps_in_epoch: int,
    lr_scheduler: Optional[Any] = None,
    max_grad_norm: Optional[float] = None,
    step: int,
) -> None:
    assert (
        scaler is not None or not fp16
    ), f"{training_step.__name__}: scaler must be set for fp16."

    device = torch.device("cuda")
    train_utils.batch_to_device(batch, device)

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
        outputs = model(**batch)
        loss = outputs["loss"] / grad_accumulation_steps

    # Log loss unscaled, not influenced by grad accumulation steps
    loss_metric.update(loss * grad_accumulation_steps)
    current_lr = (
        lr_scheduler.get_last_lr()[0]
        if lr_scheduler is not None
        else optimizer.state_dict()["param_groups"][0]["lr"]
    )
    lr_metric.update(torch.tensor(current_lr, device=device))

    for metric in metrics.values():
        metric.update(outputs, batch)

    # For fp16 there could be an underflow in gradients...
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
            # If the scale stayed the same or increased, gradients were not
            # NaNs or Infs and optimizer was run by the scaler
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
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    fp16: bool,
    metrics: dict[str, Metric],
    mean_val_loss: Metric,
) -> None:
    train_utils.batch_to_device(batch, device)

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs["loss"]

    mean_val_loss.update(loss)
    for metric in metrics.values():
        metric.update(outputs, batch)


def log_and_reset_metrics(
    *,
    metrics: dict[str, Metric],
    summary_writer: SummaryWriter,
    step: int,
    loss_metric: Metric,
    lr_metric: Optional[Metric] = None,  # Not needed for validation metrics
) -> None:
    summary_writer.add_scalar("loss", loss_metric.compute(), step)
    loss_metric.reset()

    if lr_metric is not None:
        summary_writer.add_scalar("learning_rate", lr_metric.compute(), step)
        lr_metric.reset()
    for name, metric in metrics.items():
        summary_writer.add_scalar(name, metric.compute(), step)
        metric.reset()

    summary_writer.flush()


def train(
    *,
    model: torch.nn.Module,
    train_data: DataLoader,
    val_data: Optional[DataLoader] = None,
    epochs: int,
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
    validate_every_epoch: int = 1,
    checkpoint_path: Optional[str] = None,
    patience: Optional[int] = None,
    log_every_step: Optional[int] = None,
) -> None:
    """Trains a model.

    Parameters:
    -----------
        log_every_step
    """
    device = torch.device("cuda")
    model.to(device)
    model.train()

    min_val_loss = float("inf")
    epochs_without_improvement = 0

    steps_in_epoch = steps_in_epoch if steps_in_epoch is not None else len(train_data)
    if log_every_step is None:
        log_every_step = grad_accumulation_steps

    scaler = GradScaler()

    for name, metric in metrics.items():
        metrics[name] = metric.to(device)
    loss_metric = Mean(device=device)
    lr_metric = Mean(device=device)

    for epoch in tqdm(range(epochs), desc="Epoch", disable=not progress_bar):
        for step, batch in tqdm(
            enumerate(train_data),
            desc="Batches",
            disable=not progress_bar,
            total=steps_in_epoch,
        ):
            training_step(
                model=model,
                optimizer=optimizer,
                batch=batch,
                metrics=metrics,
                loss_metric=loss_metric,
                lr_metric=lr_metric,
                fp16=fp16,
                grad_accumulation_steps=grad_accumulation_steps,
                scaler=scaler,
                steps_in_epoch=steps_in_epoch,
                step=step,
                max_grad_norm=max_grad_norm,
                lr_scheduler=lr_scheduler,
            )

            if summary_writer is not None and (step + 1) % log_every_step == 0:
                log_and_reset_metrics(
                    metrics=metrics,
                    summary_writer=summary_writer,
                    step=epoch * steps_in_epoch + step,
                    loss_metric=loss_metric,
                    lr_metric=lr_metric,
                )

        if val_data is not None and (1 + epoch) % validate_every_epoch == 0:
            model.eval()
            mean_val_loss = Mean(device=device)

            for batch in val_data:
                validation_step(
                    model=model,
                    batch=batch,
                    device=device,
                    fp16=fp16,
                    metrics=metrics,
                    mean_val_loss=mean_val_loss,
                )

            val_loss = mean_val_loss.compute()
            if val_summary_writer is not None:
                log_and_reset_metrics(
                    metrics=metrics,
                    summary_writer=val_summary_writer,
                    step=(epoch + 1) * steps_in_epoch,
                    loss_metric=mean_val_loss,
                )

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                epochs_without_improvement = 0

                if checkpoint_path is not None:
                    model.save_pretrained(checkpoint_path)
            else:
                epochs_without_improvement += 1

            if patience is not None and epochs_without_improvement == patience:
                logger.info(
                    "%n epochs without improvement. Stopping training...",
                    epochs_without_improvement,
                )
                break

            model.train()
