from dataclasses import dataclass
from typing import Any, Optional, Union, cast

import datasets
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import (BatchEncoding, PreTrainedTokenizerBase,
                          PreTrainedTokenizerFast)
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.tokenization_utils import TruncationStrategy
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils import PaddingStrategy


@dataclass
class FastDataCollator:
    """Custom data collator to be used with FastTokenizers.

    Implemented
        - to avoid the warning caused by first encoding with fast tokenizer and
          then padding with it on a separate call.
        - to have the feature of ensuring encoded input is of minimal length.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    min_length: Optional[int] = None
    truncation: Union[bool, str, TruncationStrategy] = "longest_first"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # Convert from list of dicts to dict of lists
        batch = {
            key: [example[key] for example in features] for key in features[0].keys()
        }
        texts = batch["text"]
        del batch["text"]

        tokenized_batch = cast(
            BatchEncoding,
            self.tokenizer(
                texts,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=None,
                truncation=self.truncation,
            ),
        )

        if self.min_length:
            batch_size = len(tokenized_batch["input_ids"])
            for i in range(batch_size):
                input_length = len(tokenized_batch["input_ids"][i])
                if input_length >= self.min_length:
                    continue
                difference = self.min_length - input_length

                if "attention_mask" in tokenized_batch:
                    tokenized_batch["attention_mask"][i] += [0] * difference

                if "token_type_ids" in tokenized_batch:
                    tokenized_batch["token_type_ids"][i] += [
                        self.tokenizer.pad_token_type_id
                    ] * difference
                if "special_tokens_mask" in tokenized_batch:
                    tokenized_batch["special_tokens_mask"][i] += [1] * difference

                tokenized_batch["input_ids"][i] += [
                    self.tokenizer.pad_token_id
                ] * difference

        tokenized_batch.convert_to_tensors(
            self.return_tensors, prepend_batch_axis=False
        )

        batch.update(tokenized_batch)

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch


def create_tokenized_data_loader(
    data: datasets.Dataset,
    *,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int,
    training: bool = True,
    min_sequence_length: Optional[int] = None
) -> DataLoader:
    """Creates DataLoder giving batches of tokenized text."""
    data = data.with_format("torch")
    data = data.remove_columns(["id"])

    collator = FastDataCollator(
        tokenizer=tokenizer,
        padding="longest",
        min_length=min_sequence_length,
    )
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
