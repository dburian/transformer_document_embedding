from __future__ import annotations
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterable

import logging
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torcheval.metrics import Mean, Metric, toolkit
from tqdm.auto import tqdm

import transformer_document_embedding.utils.torch.training as train_utils

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard.writer import SummaryWriter
    from typing import Optional
    from typing import Any
    from typing import Callable, Iterator

    DictGetter = Callable[[dict[str, torch.Tensor]], Any]


logger = logging.getLogger(__name__)


class LongformerTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        metrics: Optional[dict[str, Metric]] = None,
        main_metric: str = "loss",
        lower_is_better: bool = True,
        device: Optional[torch.device] = None,
        val_data: Optional[DataLoader] = None,
        summary_writer: Optional[SummaryWriter] = None,
        val_summary_writer: Optional[SummaryWriter] = None,
        fp16: bool = False,
        max_grad_norm: Optional[float] = None,
        grad_accumulation_steps: int = 1,
        lr_scheduler=None,
        save_model_callback: Optional[Callable[[torch.nn.Module, int], None]] = None,
        patience: Optional[int] = None,
        log_every_step: Optional[int] = None,
        validate_every_step: Optional[int] = None,
    ) -> None:
        """
        Parameters:
        - patience: int, optional
            Maximum number of validation rounds without improvement. If
            patience is reached, the training is stopped.
        """
        self._model = model
        self._train_data = train_data
        self._val_data = val_data

        self._metrics = metrics if metrics is not None else {}
        self._special_metrics: dict[str, Metric] = {
            "loss": Mean(),
            "learning_rate": Mean(),
        }
        self._log_every_step = (
            log_every_step if log_every_step is not None else grad_accumulation_steps
        )

        self._val_metrics = {
            name: toolkit.clone_metric(metric) for name, metric in self._metrics.items()
        }
        self._val_special_metrics: dict[str, Metric] = {
            "loss": Mean(),
        }

        self._summary_writer = summary_writer
        self._val_summary_writer = val_summary_writer

        self._device = device if device is not None else self.get_default_device()
        logger.info("Using %s for training.", self._device.type)

        self._optim = optimizer
        self._lr_scheduler = lr_scheduler

        # Use either float16 for cuda or bfloat16 for cpu
        self._fp16 = fp16
        self._max_grad_norm = max_grad_norm
        self._grad_accumulation_steps = grad_accumulation_steps

        default_validation_frequency = len(val_data) if val_data is not None else 1
        self._validate_every_step = (
            validate_every_step
            if validate_every_step is not None
            else default_validation_frequency
        )
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

        self._reset_metrics(
            (
                self._metrics,
                self._special_metrics,
                self._val_metrics,
                self._val_special_metrics,
            )
        )

    def _reset_metrics(self, metrics: Iterable[dict[str, Metric]]) -> None:
        for metric_set in metrics:
            for metric in metric_set.values():
                metric.reset()
                if metric.device != self._device:
                    metric.to(self._device)

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
                self._training_step(batch=batch, step=step)

                total_step = epoch * steps_in_epoch + step

                should_log_train = ((total_step + 1) % self._log_every_step == 0) or (
                    total_step + 1 == step_count
                )
                if self._summary_writer is not None and should_log_train:
                    self._log_and_reset(
                        total_step=total_step,
                        writer=self._summary_writer,
                        metrics=(self._metrics, self._special_metrics),
                    )

                if (
                    self._validate_every_step is not None
                    and (1 + total_step) % self._validate_every_step == 0
                ):
                    self._validate(total_step=total_step)

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
            self._validate(total_step=step_count - 1)

    def _log_and_reset(
        self,
        total_step: int,
        writer: SummaryWriter,
        metrics: Iterable[dict[str, Metric]],
    ) -> None:
        for metric_set in metrics:
            for name, metric in metric_set.items():
                writer.add_scalar(name, metric.compute(), total_step)
                metric.reset()

        writer.flush()

    def _validate(self, total_step: int) -> None:
        if self._val_data is None:
            return

        self._model.eval()
        self._reset_metrics((self._val_metrics, self._val_special_metrics))

        # TODO: Progress bar to global options
        for batch in tqdm(
            self._val_data, total=len(self._val_data), desc="Validation batches"
        ):
            self._validation_step(batch=batch)

        if self._val_summary_writer is not None:
            self._log_and_reset(
                total_step=total_step,
                writer=self._val_summary_writer,
                metrics=(self._val_metrics, self._val_special_metrics),
            )

        self._model.train()

        # TODO: Option to not do this at all?
        main_metric = self._val_metrics.get(
            self._main_metric_name,
            self._val_special_metrics.get(self._main_metric_name, None),
        )
        assert main_metric is not None, "Main metric cannot be found."
        new_score = main_metric.compute()
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

        self._val_special_metrics["loss"].update(loss)
        for metric in self._val_metrics.values():
            metric.update(outputs, batch)

    def _training_step(self, batch: dict[str, torch.Tensor], step: int) -> None:
        train_utils.batch_to_device(batch, self._device)

        with self._autocast():
            outputs = self._model(**batch)
            loss = outputs["loss"] / self._grad_accumulation_steps

        # Log loss unscaled, not influenced by grad accumulation steps
        self._special_metrics["loss"].update(loss * self._grad_accumulation_steps)
        current_lr = (
            self._lr_scheduler.get_last_lr()[0]
            if self._lr_scheduler is not None
            else self._optim.state_dict()["param_groups"][0]["lr"]
        )
        self._special_metrics["learning_rate"].update(
            torch.tensor(current_lr, device=self._device)
        )

        if self._fp16:
            assert self._scaler is not None, "Scaler must be set for fp16."
            # For fp16 there could be an underflow in gradients...
            loss = self._scaler.scale(loss) if self._fp16 else loss

        loss.backward()

        # After `loss.backward()` so that gradients are populated
        for metric in self._metrics.values():
            metric.update(outputs, batch)

        if (step + 1) % self._grad_accumulation_steps == 0 or (step + 1) == len(
            self._train_data
        ):
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
