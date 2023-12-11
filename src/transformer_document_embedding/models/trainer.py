from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from os import path
from typing import TYPE_CHECKING

import logging
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torcheval.metrics import Mean, Metric
from tqdm.auto import tqdm

import transformer_document_embedding.utils.training as train_utils
from torch.utils.tensorboard.writer import SummaryWriter

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from typing import Callable, Iterator, Union, Any, Optional

    DictGetter = Callable[[dict[str, torch.Tensor]], Any]


logger = logging.getLogger(__name__)


class TorchTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        train_logger: Optional[MetricLogger] = None,
        val_logger: Optional[MetricLogger] = None,
        main_metric: str = "loss",
        lower_is_better: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        val_data: Optional[DataLoader] = None,
        fp16: bool = False,
        max_grad_norm: Optional[float] = None,
        grad_accumulation_steps: int = 1,
        lr_scheduler=None,
        save_model_callback: Optional[Callable[[torch.nn.Module, int], None]] = None,
        patience: Optional[int] = None,
        validate_every_step: Optional[int] = None,
    ) -> None:
        """
        Parameters:
        - patience: int, optional
            Maximum number of validation rounds without improvement. If
            patience is reached, the training is stopped.
        """
        self._model = model
        # TODO: Move train and validation data to train method
        self._train_data = train_data
        self._val_data = val_data

        self._logger = train_logger
        self._val_logger = val_logger

        self._device = (
            torch.device(device) if device is not None else self.get_default_device()
        )
        logger.info("Using %s for training.", self._device.type)

        self._optim = optimizer
        self._lr_scheduler = lr_scheduler

        # Use either float16 for cuda or bfloat16 for cpu
        self._fp16 = fp16
        self._max_grad_norm = max_grad_norm
        self._grad_accumulation_steps = grad_accumulation_steps

        self._validate_every_step = validate_every_step
        self._patience = patience
        self._main_metric_name = main_metric
        self._lower_is_better = lower_is_better
        self._save_model_callback = save_model_callback

    @classmethod
    def get_default_device(cls) -> torch.device:
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def _init_train(self) -> None:
        """Resets variables for training."""
        self._model.to(self._device)
        self._model.train()

        self._best_val_score = float("inf") if self._lower_is_better else float("-inf")
        self._validations_without_improvement = 0

        self._scaler = GradScaler() if self._fp16 else None

        if self._logger is not None:
            self._logger.reset_all()
            self._logger.to(self._device)
        if self._val_logger is not None:
            self._val_logger.reset_all()
            self._val_logger.to(self._device)

    def train(self, epochs: int, progress_bar: bool = True) -> None:
        self._init_train()

        steps_in_epoch = len(self._train_data)
        step_count = steps_in_epoch * epochs

        for epoch in tqdm(range(epochs), desc="Epoch", disable=not progress_bar):
            for step, batch in tqdm(
                enumerate(self._train_data),
                desc="Batches",
                disable=not progress_bar,
                total=len(self._train_data),
            ):
                total_step = epoch * steps_in_epoch + step
                self._training_step(batch, total_step)

                if (
                    self._validate_every_step is not None
                    and (1 + total_step) % self._validate_every_step == 0
                ):
                    self._validate(total_step, progress_bar)

                    # TODO: Too much indentation
                    if (
                        self._patience is not None
                        and self._validations_without_improvement >= self._patience
                    ):
                        logger.info(
                            "%n validations without improvement. Stopping training...",
                            self._validations_without_improvement,
                        )
                        return

        if (
            self._validate_every_step is None
            or step_count % self._validate_every_step != 0
        ):
            self._validate(total_step=step_count - 1, progress_bar=progress_bar)

    def _validate(self, total_step: int, progress_bar: bool) -> None:
        if self._val_data is None:
            return

        self._model.eval()
        if self._val_logger is not None:
            self._val_logger.reset_all()

        for batch in tqdm(
            self._val_data,
            total=len(self._val_data),
            desc="Validation batches",
            disable=not progress_bar,
        ):
            self._validation_step(batch)

        if self._val_logger is not None:
            self._val_logger.log(total_step, force=True)

        self._model.train()

        # TODO: Option to not do this at all?
        new_score = (
            self._val_logger.get_value(self._main_metric_name, None)
            if self._val_logger is not None
            else None
        )
        if new_score is not None:
            is_better = (
                (new_score < self._best_val_score)
                if self._lower_is_better
                else (new_score > self._best_val_score)
            )
            if is_better:
                self._best_val_score = new_score
                self._validations_without_improvement = 0

                if self._save_model_callback is not None:
                    logger.info("Saving checkpoint of model at step %d", total_step)
                    self._save_model_callback(self._model, total_step)
            else:
                self._validations_without_improvement += 1

    def _validation_step(self, batch: dict[str, torch.Tensor]) -> None:
        train_utils.batch_to_device(batch, self._device)

        with self._autocast():
            with torch.no_grad():
                outputs = self._model(**batch)
                loss = outputs["loss"]

        if self._val_logger is not None:
            self._val_logger.add_scalars(
                outputs,
                batch,
                loss=loss,
                auto_log=False,
            )

    def _training_step(self, batch: dict[str, torch.Tensor], total_step: int) -> None:
        train_utils.batch_to_device(batch, self._device)

        with self._autocast():
            outputs = self._model(**batch)
            loss = outputs["loss"] / self._grad_accumulation_steps

        if self._fp16:
            assert self._scaler is not None, "Scaler must be set for fp16."
            # For fp16 there could be an underflow in gradients...
            loss = self._scaler.scale(loss) if self._fp16 else loss
            assert isinstance(loss, torch.Tensor)

        loss.backward()

        if self._logger is not None:
            current_lr = (
                self._lr_scheduler.get_last_lr()[0]
                if self._lr_scheduler is not None
                else self._optim.state_dict()["param_groups"][0]["lr"]
            )

            # After `loss.backward()` so that gradients are populated
            self._logger.add_scalars(
                outputs,
                batch,
                # Log loss unscaled, not influenced by grad accumulation steps
                loss=loss * self._grad_accumulation_steps,
                lr=torch.tensor(current_lr, device=self._device),
                total_step=total_step,
                auto_log=True,
            )

        if (total_step + 1) % self._grad_accumulation_steps == 0 or (
            total_step + 1
        ) == len(self._train_data):
            if self._max_grad_norm is not None:
                if self._scaler is not None:
                    self._scaler.unscale_(self._optim)

                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), self._max_grad_norm
                )

            optimizer_was_run = True
            if self._fp16:
                assert self._scaler is not None

                scale_before = self._scaler.get_scale()
                self._scaler.step(self._optim)
                scale_after = self._scaler.get_scale()
                # If the scale stayed the same or increased, gradients were not
                # NaNs or Infs and optimizer was run by the scaler
                optimizer_was_run = scale_before <= scale_after
                self._scaler.update()
            else:
                self._optim.step()

            if optimizer_was_run and self._lr_scheduler is not None:
                self._lr_scheduler.step()

            self._optim.zero_grad()

    @contextmanager
    def _autocast(self) -> Iterator[None]:
        """Enables optional autocast.

        Note that `enabled` param in torch's autocast doesn't do that."""
        if self._fp16:
            with torch.autocast(device_type=self._device.type):
                yield
        else:
            yield


@dataclass
class TrainingMetric:
    """`torcheval` Metric wrapped with information how to handle it during training."""

    @staticmethod
    def identity_update_fn(metric: Metric, *args: Any) -> None:
        metric.update(*args)

    name: str
    metric: Metric
    log_frequency: Optional[int] = None
    update_fn: Callable = identity_update_fn
    reset_after_log: bool = True

    @property
    def device(self) -> torch.device:
        return self.metric.device

    def update(self, *args) -> None:
        self.update_fn(self.metric, *args)

    def clone(self, **kwargs_overwrite) -> TrainingMetric:
        kwargs = asdict(self)
        kwargs.update(kwargs_overwrite)
        return TrainingMetric(**kwargs)

    def compute(self) -> Any:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()

    def to(self, device: torch.device) -> TrainingMetric:
        self.metric.to(device)
        return self


class MetricLogger:
    LOSS_NAME = "loss"
    LR_NAME = "learning_rate"

    def __init__(
        self,
        name: str,
        metrics: list[TrainingMetric],
        log_dir: str,
        log_lr: bool = True,
        log_loss: bool = True,
    ) -> None:
        self.metrics = {metric.name: metric for metric in metrics}

        log_frequencies = [
            metric.log_frequency
            for metric in metrics
            if metric.log_frequency is not None
        ]
        special_metrics_log_frequency = (
            min(log_frequencies) if len(log_frequencies) > 0 else None
        )

        self.loss_metric = None
        if log_loss:
            self.metrics[self.LOSS_NAME] = TrainingMetric(
                self.LOSS_NAME,
                Mean(),
                log_frequency=special_metrics_log_frequency,
                reset_after_log=True,
            )

        if log_lr:
            self.metrics[self.LR_NAME] = TrainingMetric(
                self.LR_NAME,
                Mean(),
                log_frequency=special_metrics_log_frequency,
                reset_after_log=True,
            )

        self.writer = SummaryWriter(path.join(log_dir, name))

    def add_scalars(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        lr: Optional[torch.Tensor] = None,
        loss: Optional[torch.Tensor] = None,
        total_step: Optional[int] = None,
        auto_log: bool = False,
    ) -> None:
        for metric in self.metrics.values():
            if metric.name == self.LOSS_NAME:
                if loss is not None:
                    metric.update(loss)
            elif metric.name == self.LR_NAME:
                if lr is not None:
                    metric.update(lr)
            else:
                metric.update(outputs, batch)

        if auto_log and total_step is not None:
            self.log(total_step, force=False)

    def to(self, device: torch.device) -> None:
        for metric in self.metrics.values():
            metric.to(device)

    def log(self, total_step: int, force: bool = False) -> None:
        for metric in self.metrics.values():
            if force or (
                metric.log_frequency is not None
                and total_step % metric.log_frequency == 0
            ):
                self.writer.add_scalar(metric.name, metric.compute(), total_step)
                if metric.reset_after_log:
                    metric.reset()

    def reset_all(self) -> None:
        for metric in self.metrics.values():
            metric.reset()

    def get_value(self, metric_name: str, default=None) -> Optional[torch.Tensor]:
        if metric_name not in self.metrics:
            return default

        return self.metrics[metric_name].compute()
