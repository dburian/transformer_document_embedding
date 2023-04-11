from __future__ import annotations

from typing import Any, Iterable, Optional, Union

import torch
from torcheval.metrics import Mean, Metric


class MeanLossMetric(Metric):
    """Metric accumulating the mean loss."""

    def __init__(self, loss_fn: torch.nn.Module, **kwargs) -> None:
        super().__init__(**kwargs)
        self._loss_fn = loss_fn
        self._mean_loss = Mean(device=self.device)

    def to(
        self, device: Union[str, torch.device], *args: Any, **kwargs: Any
    ) -> MeanLossMetric:
        self._loss_fn.to(device)
        self._mean_loss = self._mean_loss.to(device, *args, **kwargs)

        self._device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        return self

    @torch.inference_mode()
    def update(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        loss = self._loss_fn(outputs, labels)
        self._mean_loss.update(loss)

    @torch.inference_mode()
    def compute(self) -> torch.Tensor:
        return self._mean_loss.compute()

    def reset(self) -> None:
        self._mean_loss.reset()

    @torch.inference_mode()
    def merge_state(self, metrics: Iterable[MeanLossMetric]) -> MeanLossMetric:
        for metric in metrics:
            self._mean_loss.update(metric._mean_loss.compute())

        return self

    def state_dict(self) -> dict[str, Any]:
        return {
            "loss_fn": self._loss_fn.state_dict(),
            "mean_loss": self._mean_loss.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        self._loss_fn.load_state_dict(state_dict["loss_fn"], strict)
        self._mean_loss.load_state_dict(state_dict["mean_loss"], strict)


class VMemMetric(Metric):
    """Metric outputting current amount of video memory used by pytorch."""

    def __init__(
        self,
        return_MB: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._return_MB = return_MB  # pylint: disable=invalid-name

    def update(self, *args, **kwargs) -> None:
        pass

    @torch.inference_mode()
    def compute(self) -> torch.Tensor:
        mem_used = torch.cuda.memory_reserved(self._device)
        mem_used = mem_used // 1024**2 if self._return_MB else mem_used
        return torch.tensor(mem_used)

    def merge_state(self, _: Iterable[VMemMetric]) -> VMemMetric:
        return self
