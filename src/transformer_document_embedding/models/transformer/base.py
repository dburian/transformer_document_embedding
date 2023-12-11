from __future__ import annotations
from os import path
import torch
from typing import TYPE_CHECKING, Callable
from torch.utils.tensorboard.writer import SummaryWriter

from transformers import AutoModel, AutoTokenizer
from transformer_document_embedding.models.experimental_model import ExperimentalModel
from transformer_document_embedding.utils.training import (
    create_tokenized_data_loader,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from datasets import Dataset
    from typing import Optional, Any


class MeanPooler(torch.nn.Module):
    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor, **_
    ) -> torch.Tensor:
        summed = torch.sum(last_hidden_state * attention_mask[:, :, None], 1)
        row_lengths = torch.sum(attention_mask, 1)
        return summed / row_lengths[:, None]


class LocalMeanPooler(torch.nn.Module):
    """Does mean pooling from tokens without global attention."""

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        global_attention_mask: torch.Tensor,
        **_,
    ) -> torch.Tensor:
        attn = attention_mask * (1 - global_attention_mask)
        attn.unsqueeze_(-1)
        masked = last_hidden_state * attn
        return masked.sum(dim=1) / attn.sum(dim=1)


class ClsPooler(torch.nn.Module):
    def forward(self, last_hidden_state: torch.Tensor, **_) -> torch.Tensor:
        return last_hidden_state[:, 0]


class SumPooler(torch.nn.Module):
    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        **_,
    ) -> torch.Tensor:
        return torch.sum(last_hidden_state * attention_mask[:, :, None], 1)


AVAILABLE_POOLERS = {
    "sum": SumPooler,
    "mean": MeanPooler,
    "local_mean": LocalMeanPooler,
    "cls": ClsPooler,
}


class TransformerBase(ExperimentalModel):
    def __init__(
        self,
        transformer_model: str,
        batch_size: int,
        pooler_type: str,
        transformer_model_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._batch_size = batch_size

        if transformer_model_kwargs is None:
            transformer_model_kwargs = {}

        self._transformer = AutoModel.from_pretrained(
            transformer_model, **transformer_model_kwargs
        )
        self._tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        self._pooler = AVAILABLE_POOLERS[pooler_type]()

        self._min_sequence_length = None
        # For BigBird's sparse attention we need to assert minimum sequence length
        if (
            hasattr(self._transformer.config, "attention_type")
            and self._transformer.config.attention_type == "block_sparse"
        ):
            # To use `block_sparse` attention, sequence_length has to be padded
            # to multiples of `block_size` and bigger than:
            # 2 * block_size (global attentions) +
            # 3 * block_size (sliding tokens) +
            # 2 * num_random_blocks * block_size (random tokens)
            block_size = self._transformer.config.block_size
            num_random_blocks = self._transformer.config.num_random_blocks
            self._min_sequence_length = (6 + 2 * num_random_blocks) * block_size

        # Will fill-in child classes
        self._model: torch.nn.Module

    def _to_dataloader(
        self, dataset: Dataset, training: bool = True, **kwargs
    ) -> DataLoader:
        return create_tokenized_data_loader(
            dataset,
            batch_size=self._batch_size,
            training=training,
            min_length=self._min_sequence_length,
            tokenizer=self._tokenizer,
            **kwargs,
        )

    def _get_train_val_writers(
        self, log_dir: Optional[str]
    ) -> tuple[Optional[SummaryWriter], Optional[SummaryWriter]]:
        if log_dir is None:
            return None, None

        return SummaryWriter(path.join(log_dir, "train")), SummaryWriter(
            path.join(log_dir, "val")
        )

    def _get_save_model_callback(
        self, save_best: bool, model_dir: Optional[str]
    ) -> Optional[Callable[[torch.nn.Module, int], None]]:
        if not save_best or model_dir is None:
            return None

        def _cb(*_) -> None:
            self.save(path.join(model_dir, "checkpoint"))

        return _cb

    def _model_save_file_path(self, dir_path: str) -> str:
        return path.join(dir_path, "model")

    def save(self, dir_path: str) -> None:
        torch.save(self._model.state_dict(), self._model_save_file_path(dir_path))

    def load(self, dir_path: str) -> None:
        state_dict = torch.load(
            self._model_save_file_path(dir_path),
            map_location=torch.device("cpu"),
        )
        self._model.load_state_dict(state_dict)
