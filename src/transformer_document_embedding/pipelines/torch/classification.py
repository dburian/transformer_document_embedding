from __future__ import annotations
from copy import copy
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING
import torch
from torcheval.metrics import (
    MulticlassAUPRC,
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
)
from transformers import AutoTokenizer
from transformer_document_embedding.datasets import col
from transformer_document_embedding.pipelines.torch.train import TorchTrainPipeline

from transformer_document_embedding.utils.tokenizers import (
    FastDataCollator,
)

from transformer_document_embedding.utils.metrics import TrainingMetric
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from datasets import Dataset
    from transformer_document_embedding.models.transformer import (
        TransformerEmbedder,
    )


class _Classifier(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, head: torch.nn.Module) -> None:
        super().__init__()

        self.encoder = encoder
        self.head = head

    def get_encompassing_model(
        self, model: TransformerEmbedder, head: torch.nn.Module
    ) -> torch.nn.Module:
        return _Classifier(model, head)

    def forward(
        self, labels: Optional[torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder(**kwargs)
        outputs.update(self.head(outputs["embeddings"], labels))

        return outputs


class TorchBinaryClassifiactionPipeline(TorchTrainPipeline):
    def get_train_metrics(
        self, log_freq: int, model: _Classifier
    ) -> list[TrainingMetric]:
        def logits_accessor(metric, outputs, batch):
            metric.update(outputs["logits"], batch["labels"])

        supers_metrics = super(
            TorchBinaryClassifiactionPipeline, self
        ).get_train_metrics(log_freq, model)

        # TODO: Isn't this the same as for PV?
        classification_metrics = [
            TrainingMetric("accuracy", MulticlassAccuracy(), log_freq, logits_accessor),
            TrainingMetric("recall", MulticlassRecall(), log_freq, logits_accessor),
            TrainingMetric(
                "precision",
                MulticlassPrecision(),
                log_freq,
                logits_accessor,
            ),
            TrainingMetric(
                "auprc",
                MulticlassAUPRC(num_classes=2),
                log_freq,
                logits_accessor,
            ),
        ]

        return supers_metrics + classification_metrics


@dataclass
class TorchTrainPairClassificationPipeline(TorchTrainPipeline):
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


class _PairClassifier(_Classifier):
    def forward(
        self,
        inputs_0: dict[str, Any],
        inputs_1: dict[str, Any],
        labels: Optional[torch.Tensor],
        **_,
    ) -> dict[str, torch.Tensor]:
        pair_pooled_outputs = [
            self.encoder(**inputs)[col.EMBEDDING] for inputs in [inputs_0, inputs_1]
        ]

        embeddings = torch.cat(pair_pooled_outputs, dim=1)
        head_outputs = self.head(embeddings, labels)

        head_outputs[col.EMBEDDING] = embeddings
        return head_outputs
