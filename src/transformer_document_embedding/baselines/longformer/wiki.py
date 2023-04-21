from typing import Iterable, Optional, cast

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer

import transformer_document_embedding.baselines.longformer.train as train_utils
from transformer_document_embedding.baselines.experimental_model import \
    ExperimentalModel
from transformer_document_embedding.models.longformer import (
    LongformerConfig, LongformerForTextEmbedding)
from transformer_document_embedding.tasks import ExperimentalTask


class LongformerWikipediaSimilarities(ExperimentalModel):
    def __init__(
        self, *, large: bool = False, batch_size: int = 1, pooler_type: str = "mean"
    ) -> None:
        self._batch_size = batch_size

        model_path = f"allenai/longformer-{'large' if large else 'base'}-4096"
        config = LongformerConfig(
            pooler_type=pooler_type,
            **LongformerConfig.get_config_dict(model_path)[0],
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = cast(
            LongformerForTextEmbedding,
            LongformerForTextEmbedding.from_pretrained(model_path, config=config),
        )

    def train(
        self,
        task: ExperimentalTask,
        *,
        log_dir: Optional[str] = None,
        early_stopping: bool = False,
        save_best_path: Optional[str] = None,
    ) -> None:
        pass

    def predict(self, inputs: Dataset) -> Iterable[np.ndarray]:
        self._model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)

        data = train_utils.prepare_data(
            inputs,
            tokenizer=self._tokenizer,
            batch_size=self._batch_size,
            training=False,
        )
        for batch in data:
            train_utils.batch_to_device(batch, self._model.device)
            logits = self._model(**batch).pooler_output
            preds = torch.argmax(logits, 1)
            yield preds.numpy(force=True)

    def save(self, dir_path: str) -> None:
        self._model.save_pretrained(dir_path)

    def load(self, dir_path: str) -> None:
        self._model = cast(
            LongformerForTextEmbedding,
            LongformerForTextEmbedding.from_pretrained(dir_path),
        )
