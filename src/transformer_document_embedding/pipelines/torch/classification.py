from __future__ import annotations
from copy import copy
from typing import Any, Optional, TYPE_CHECKING
import torch
from transformers import AutoTokenizer
from transformer_document_embedding.datasets import col
from transformer_document_embedding.pipelines.helpers import classification_metrics
from transformer_document_embedding.pipelines.torch.train import TorchTrainPipeline

from transformer_document_embedding.utils.tokenizers import (
    FastDataCollator,
)

from transformer_document_embedding.utils.metrics import TrainingMetric
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from transformer_document_embedding.datasets.document_dataset import DocumentDataset
    from datasets import Dataset
    from transformer_document_embedding.models.transformer import (
        TransformerEmbedder,
    )


class _Classifier(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, head: torch.nn.Module) -> None:
        super().__init__()

        self.encoder = encoder
        self.head = head

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        outputs = self.encoder(**kwargs)
        if col.LABEL in kwargs:
            outputs.update(
                self.head(
                    **{
                        col.EMBEDDING: outputs[col.EMBEDDING],
                        col.LABEL: kwargs[col.LABEL],
                    }
                )
            )

        return outputs


class TorchClassifiactionPipeline(TorchTrainPipeline):
    def __call__(
        self,
        encoder: TransformerEmbedder,
        head: torch.nn.Module,
        dataset: DocumentDataset,
        log_dir: Optional[str],
    ) -> None:
        self.num_classes = len(dataset.splits["train"].unique(col.LABEL))

        return super().__call__(encoder, head, dataset, log_dir)

    def get_encompassing_model(
        self, model: TransformerEmbedder, head: torch.nn.Module
    ) -> torch.nn.Module:
        return _Classifier(model, head)

    def get_train_metrics(
        self, log_freq: int, model: _Classifier
    ) -> list[TrainingMetric]:
        def logits_accessor(metric, outputs, batch):
            metric.update(outputs["logits"], batch[col.LABEL])

        supers_metrics = super(TorchClassifiactionPipeline, self).get_train_metrics(
            log_freq, model
        )

        cls_metrics = [
            TrainingMetric(name, metric, log_freq, logits_accessor)
            for name, metric in classification_metrics(self.num_classes).items()
        ]

        return supers_metrics + cls_metrics


class _PairClassifier(_Classifier):
    def forward(
        self,
        inputs_0: dict[str, Any],
        inputs_1: dict[str, Any],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        pair_pooled_outputs = [
            self.encoder(**inputs)[col.EMBEDDING] for inputs in [inputs_0, inputs_1]
        ]

        outputs = {col.EMBEDDING: torch.cat(pair_pooled_outputs, dim=1)}
        if col.LABEL in kwargs:
            outputs.update(
                self.head(
                    **{
                        col.EMBEDDING: outputs[col.EMBEDDING],
                        col.LABEL: kwargs[col.LABEL],
                    }
                )
            )

        return outputs


class TorchTrainPairClassificationPipeline(TorchClassifiactionPipeline):
    def to_dataloader(
        self, split: Dataset, model: TransformerEmbedder, training: bool = True
    ) -> DataLoader:
        split = split.with_format("torch")
        split = split.remove_columns([col.ID_0, col.ID_1])

        collator = PairFastDataCollator(
            padding="longest",
            tokenizer=AutoTokenizer.from_pretrained(model.transformer_name),
            min_length=model.min_sequence_length,
        )

        return DataLoader(
            split,
            batch_size=self.batch_size,
            shuffle=training,
            collate_fn=collator,
        )

    def get_encompassing_model(
        self, model: TransformerEmbedder, head: torch.nn.Module
    ) -> torch.nn.Module:
        return _PairClassifier(model, head)


class PairFastDataCollator(FastDataCollator):
    """Tokenizes pair of texts and puts their tokenization to two keys."""

    TOKENIZER_OUTPUT_KEYS = {
        "length",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "special_tokens_mask",
    }

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batches = [
            self._tokenize_single(features, text_key)
            for text_key in [col.TEXT_0, col.TEXT_1]
        ]

        global_inputs = {}
        for batch in batches:
            # If value was not inserted by tokenizer, assume it is the same
            # for both inputs
            for key in set(batch.keys()) - self.TOKENIZER_OUTPUT_KEYS:
                global_inputs[key] = batch.pop(key)

        global_inputs["inputs_0"] = batches[0]
        global_inputs["inputs_1"] = batches[1]

        return global_inputs

    def _tokenize_single(
        self, features: list[dict[str, Any]], text_key: str
    ) -> dict[str, Any]:
        vanilla_feats = []
        for example in features:
            example = copy(example)
            example[col.TEXT] = example[text_key]
            del example[text_key]
            vanilla_feats.append(example)

        return super().__call__(vanilla_feats)
