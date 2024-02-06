from __future__ import annotations
from os import path
import torch
import logging
from typing import TYPE_CHECKING, Callable

from transformers import AutoModel, AutoTokenizer
from transformer_document_embedding.models.experimental_model import ExperimentalModel
from transformer_document_embedding.models.trainer import MetricLogger
from transformer_document_embedding.utils.net_helpers import (
    load_model_weights,
    save_model_weights,
)
from transformer_document_embedding.utils.tokenizers import create_tokenized_data_loader

if TYPE_CHECKING:
    from transformer_document_embedding.utils.metrics import TrainingMetric
    from torch.utils.data import DataLoader
    from datasets import Dataset
    from typing import Optional, Any


logger = logging.getLogger(__name__)


class MeanPooler(torch.nn.Module):
    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor, **_
    ) -> torch.Tensor:
        summed = torch.sum(last_hidden_state * attention_mask[:, :, None], 1)
        row_lengths = torch.sum(attention_mask, 1)
        return summed / torch.clamp(row_lengths[:, None], min=1e-9)


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
        return masked.sum(dim=1) / torch.clamp(attn.sum(dim=1), min=1e-9)


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

        if "add_pooling_layer" not in transformer_model_kwargs:
            # Never add pooling layer through HF unless explicitly said so.
            # Normally is not configurable.
            transformer_model_kwargs["add_pooling_layer"] = False

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

    def _get_train_val_loggers(
        self,
        log_dir: Optional[str],
        train_metrics: list[TrainingMetric],
        val_metrics: Optional[list[TrainingMetric]] = None,
    ) -> tuple[Optional[MetricLogger], Optional[MetricLogger]]:
        if log_dir is None:
            return None, None

        if val_metrics is None:
            val_metrics = [m.clone() for m in train_metrics]

        train_logger = MetricLogger("train", train_metrics, log_dir)
        val_logger = MetricLogger("val", val_metrics, log_dir, log_lr=False)

        return train_logger, val_logger

    def _get_save_model_callback(
        self,
        saving_permitted: bool,
        log_dir: Optional[str],
    ) -> Optional[Callable[[torch.nn.Module, int], None]]:
        if not saving_permitted or log_dir is None:
            return None

        def _cb(_, total_steps: int) -> None:
            save_path = path.join(log_dir, f"checkpoint_{total_steps}")
            logger.info("Saving after %d step to '%s'", total_steps, save_path)
            self.save_weights(save_path)

        return _cb

    def _model_save_file_path(self, dir_path: str) -> str:
        return path.join(dir_path, "model")

    def save_weights(self, dir_path: str) -> None:
        save_model_weights(self._model, self._model_save_file_path(dir_path))

    def load_weights(self, dir_path: str, *, strict: bool = True) -> None:
        load_model_weights(
            self._model, self._model_save_file_path(dir_path), strict=strict
        )


class TransformerBaseModule(torch.nn.Module):
    def __init__(
        self, transformer: torch.nn.Module, pooler: torch.nn.Module, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.transformer = transformer
        self.pooler = pooler

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        input_kws = {}
        # Needed for Longformer
        if "global_attention_mask" in kwargs:
            # we need to pass global attn. mask if it was passed and don't if it wasn't
            input_kws["global_attention_mask"] = kwargs.pop("global_attention_mask")

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            inputs_embeds=None,
            **input_kws,
        )
        pooled_outputs = self.pooler(
            **outputs,
            attention_mask=attention_mask,
            **input_kws,
        )

        return {**outputs, "pooled_outputs": pooled_outputs}
