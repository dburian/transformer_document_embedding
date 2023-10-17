from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from transformer_document_embedding.baselines.baseline import (
    Baseline,
)
import numpy as np
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from typing import Iterable, Optional
    from datasets import Dataset
    from transformer_document_embedding.tasks.experimental_task import ExperimentalTask


class SBERTWikipediaSimilarities(Baseline):
    def __init__(
        self,
        *,
        transformer_model: str = "all-distilroberta-v1",
        batch_size: int = 12,
    ) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(transformer_model, device=self._device)
        logger.info("Using '%s' for both training and prediction.", self._device)
        self._batch_size = batch_size

    def train(
        self,
        task: ExperimentalTask,
        log_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        pass

    def predict(self, inputs: Dataset) -> Iterable[np.ndarray]:
        # SentenceTransformer.encode returns all results at once -> mem
        # overflows if there is enough `inputs`
        for i in tqdm(
            range(0, len(inputs), self._batch_size), desc="Predicting batches"
        ):
            batch = inputs[i : i + self._batch_size]
            preds = self._model.encode(
                batch["text"],
                batch_size=len(batch),
                device=self._device,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            assert isinstance(preds, np.ndarray)
            yield preds

    def save(self, dir_path: str) -> None:
        self._model.save(dir_path)

    def load(self, dir_path: str) -> None:
        self._model = SentenceTransformer(dir_path, device=self._device)
