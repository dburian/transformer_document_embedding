from typing import Any

import datasets
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
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
