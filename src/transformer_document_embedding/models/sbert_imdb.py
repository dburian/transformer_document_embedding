from __future__ import annotations
from typing import Iterable

import numpy as np
import torch
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    datasets,
    models,
    evaluation,
)

from torch.utils.data import DataLoader

from transformer_document_embedding.models.experimental_model import ExperimentalModel
from transformer_document_embedding.tasks.imdb import IMDBData
from transformer_document_embedding.utils import sentence_transformers as tde_st_utils


class SBertIMDB(ExperimentalModel):
    class STDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset: datasets.Dataset) -> None:
            self._hf_dataset = hf_dataset

        def __len__(self) -> int:
            return len(self._hf_dataset)

        def __getitem__(self, idx: int) -> InputExample:
            doc = self._hf_dataset[idx]
            return InputExample(texts=[doc["text"]], label=doc["label"])

    def __init__(
        self,
        log_dir: str,
        transformer_model: str = "all-distilroberta-v1",
        batch_size: int = 64,
        epochs: int = 10,
        warmup_steps: int = 10000,
    ) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._batch_size = batch_size
        self._epochs = epochs
        self._warmup_steps = warmup_steps

        sent_transformer = SentenceTransformer(transformer_model)
        embed_dim = sent_transformer.get_sentence_embedding_dimension()
        # TODO: What is this??
        cls_head = models.Dense(
            in_features=embed_dim,
            out_features=1,
            activation_function=torch.nn.Sigmoid(),
        )
        self._model = SentenceTransformer(
            modules=[sent_transformer, cls_head],
            device=self._device,
        )

        self._log_dir = log_dir
        self._loss = tde_st_utils.losses.BCELoss(self._model)

    def train(self, *, train: IMDBData, **_) -> None:
        train_size = len(train)
        training_data = self._to_st_dataset(train)

        loss_evaluator = tde_st_utils.evaluation.LossEvaluator(
            train,
            self._log_dir,
            torch.nn.BCELoss(),
            batch_size=self._batch_size,
        )
        accuracy_evaluator = tde_st_utils.evaluation.AccuracyEvaluator(
            train, self._log_dir, batch_size=self._batch_size
        )
        evaluator = evaluation.SequentialEvaluator([loss_evaluator, accuracy_evaluator])

        self._model.fit(
            train_objectives=[(training_data, self._loss)],
            epochs=self._epochs,
            warmup_steps=self._warmup_steps,
            evaluator=evaluator,
            evaluation_steps=train_size,
        )

    def predict(self, inputs: IMDBData) -> Iterable[np.ndarray]:
        for i in range(0, len(inputs), self._batch_size):
            batch = inputs[i : i + self._batch_size]
            yield self._model.encode(
                batch["text"],
                batch_size=self._batch_size,
                convert_to_numpy=True,
            )

    def save(self, dir_path: str) -> None:
        self._model.save(dir_path)

    def load(self, dir_path: str) -> None:
        self._model = SentenceTransformer(dir_path, device=self._device)

    def _to_st_dataset(self, data: IMDBData) -> DataLoader:
        ex_inputs = SBertIMDB.STDataset(data.with_format("torch"))
        return DataLoader(ex_inputs, batch_size=self._batch_size, shuffle=True)


Model = SBertIMDB
