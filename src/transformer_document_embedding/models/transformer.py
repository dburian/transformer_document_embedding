from __future__ import annotations
import torch
import logging
from typing import TYPE_CHECKING, Iterator
from tqdm.auto import tqdm

from transformers import AutoModel, AutoTokenizer
from transformer_document_embedding.datasets import col
from transformer_document_embedding.models.embedding_model import EmbeddingModel
from transformer_document_embedding.utils.net_helpers import (
    load_model_weights,
    save_model_weights,
)
from transformer_document_embedding.utils.tokenizers import create_tokenized_data_loader
import transformer_document_embedding.utils.training as train_utils

if TYPE_CHECKING:
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


class TransformerEmbedder(torch.nn.Module, EmbeddingModel):
    def __init__(
        self,
        transformer_name: str,
        pooler_type: str,
        transformer_model_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.transformer_name = transformer_name

        if transformer_model_kwargs is None:
            transformer_model_kwargs = {}

        if "add_pooling_layer" not in transformer_model_kwargs:
            # Never add pooling layer through HF unless explicitly said so.
            # Normally is not configurable.
            transformer_model_kwargs["add_pooling_layer"] = False

        self.transformer = AutoModel.from_pretrained(
            transformer_name, **transformer_model_kwargs
        )
        self.pooler = AVAILABLE_POOLERS[pooler_type]()

        self.min_sequence_length = None
        # For BigBird's sparse attention we need to assert minimum sequence length
        if (
            hasattr(self.transformer.config, "attention_type")
            and self.transformer.config.attention_type == "block_sparse"
        ):
            # To use `block_sparse` attention, sequence_length has to be padded
            # to multiples of `block_size` and bigger than:
            # 2 * block_size (global attentions) +
            # 3 * block_size (sliding tokens) +
            # 2 * num_random_blocks * block_size (random tokens)
            block_size = self.transformer.config.block_size
            num_random_blocks = self.transformer.config.num_random_blocks
            self.min_sequence_length = (6 + 2 * num_random_blocks) * block_size

    @property
    def embedding_dim(self) -> int:
        return self.transformer.config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        # For kwargs that are used only for some models. If passed, forward them.
        encoder_kwargs = {}

        if (
            global_attention_mask := kwargs.get("global_attention_mask", None)
        ) is not None:
            encoder_kwargs = {"global_attention_mask": global_attention_mask}

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
            **encoder_kwargs,
        )

        pooled_outputs = self.pooler(
            **outputs,
            attention_mask=attention_mask,
            **encoder_kwargs,
        )

        return {**outputs, col.EMBEDDING: pooled_outputs}

    @torch.inference_mode()
    def predict_embeddings(self, dataset: Dataset) -> Iterator[torch.Tensor]:
        # TODO: To function kwargs
        batch_size = 4

        # Remove all supervised columns
        dataset = dataset.remove_columns(
            list(
                {col.LABEL, col.STRUCTURAL_EMBED, col.CONTEXTUAL_EMBED}
                & set(dataset.column_names)
            )
        )

        batches = create_tokenized_data_loader(
            dataset,
            batch_size=batch_size,
            training=False,
            min_length=self.min_sequence_length,
            tokenizer=AutoTokenizer.from_pretrained(self.transformer_name),
        )

        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Predicting using {device}")
        self.to(device)
        for batch in tqdm(batches, desc="Predicting batches"):
            train_utils.batch_to_device(batch, device)
            yield self(**batch)[col.EMBEDDING]

        # Move the model back to cpu when not in use
        self.to("cpu")

    def save_weights(self, filepath: str) -> None:
        save_model_weights(self, filepath)

    def load_weights(self, dir_path: str, *, strict: bool = True) -> None:
        load_model_weights(self, dir_path, strict=strict)
