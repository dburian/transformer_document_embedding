from typing import Iterable, Optional

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer

from transformer_document_embedding.baselines.experimental_model import \
    ExperimentalModel
from transformer_document_embedding.tasks.experimental_task import \
    ExperimentalTask


class SBERTWikipediaSimilarities(ExperimentalModel):
    def __init__(
        self,
        *,
        transformer_model: str = "all-distilroberta-v1",
        batch_size: int = 6,
    ) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(transformer_model, device=self._device)
        self._batch_size = batch_size

    def train(
        self,
        task: ExperimentalTask,
        *,
        log_dir: Optional[str] = None,
        save_best_path: Optional[str] = None,
        early_stopping: bool = False,
    ) -> None:
        pass

    def predict(self, inputs: Dataset) -> Iterable[np.ndarray]:
        for i in range(0, len(inputs), self._batch_size):
            batch = inputs[i : i + self._batch_size]
            yield self._model.encode(
                batch["text"],
                batch_size=self._batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

    def save(self, dir_path: str) -> None:
        self._model.save(dir_path)

    def load(self, dir_path: str) -> None:
        self._model = SentenceTransformer(dir_path, device=self._device)
