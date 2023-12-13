from __future__ import annotations

import math

import logging
from typing import TYPE_CHECKING

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


def get_cosine_lr_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int
) -> LambdaLR:
    def cos_lambda(current_step: int) -> float:
        progress = current_step / total_steps
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, cos_lambda)
