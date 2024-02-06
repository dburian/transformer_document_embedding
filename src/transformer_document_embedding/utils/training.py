from __future__ import annotations
from functools import partial

import math

import logging
from typing import TYPE_CHECKING, Iterator

import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers.trainer_pt_utils import get_parameter_names

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> None:
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, dict):
            batch_to_device(v, device)


def get_optimizer_params(
    model: torch.nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    def _named_params() -> Iterator[tuple[str, torch.nn.Parameter]]:
        for n, p in model.named_parameters():
            if p.requires_grad:
                yield n, p

    return [
        {
            "params": [p for n, p in _named_params() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in _named_params() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]


def linear_lambda_lr(
    current_step: int, /, total_steps: int, warmup_steps: int
) -> float:
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(
        0.0,
        float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)),
    )


def cos_lambda_lr(current_step: int, /, total_steps: int, warmup_steps: int) -> float:
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))

    # Set the lr to 0 after total_steps
    progress = min((current_step - warmup_steps) / (total_steps - warmup_steps), 1)
    return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))


def get_lr_scheduler(
    scheduler_type: str,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 0,
) -> LambdaLR:
    scheduler_lambda = None
    if scheduler_type == "linear":
        scheduler_lambda = partial(
            linear_lambda_lr, total_steps=total_steps, warmup_steps=warmup_steps
        )
    elif scheduler_type == "cos":
        scheduler_lambda = partial(
            cos_lambda_lr, total_steps=total_steps, warmup_steps=warmup_steps
        )

    assert scheduler_lambda is not None, f"scheduler type '{scheduler_type}' unknown"

    return LambdaLR(optimizer, scheduler_lambda)
