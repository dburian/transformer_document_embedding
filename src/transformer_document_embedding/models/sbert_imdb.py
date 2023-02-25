from typing import Iterable, Sequence

import numpy as np
import tensorflow as tf
import torch
from sentence_transformers import InputExample, SentenceTransformer, models
from torch.utils.data import DataLoader

from transformer_document_embedding.models.experimental_model import ExperimentalModel
from transformer_document_embedding.tasks.imdb import IMDBData


class SBertIMDB(ExperimentalModel):
    def __init__(
        self,
        log_dir: str,
        transformer_model: str = "bert-base-uncased",
        max_seq_length: int = 512,
        batch_size: int = 64,
        epochs: int = 10,
        warmup_steps: int = 10000,
    ) -> None:
        self._batch_size = batch_size
        self._epochs = epochs
        self._warmup_steps = warmup_steps

        transformer = models.Transformer(
            transformer_model, max_seq_length=max_seq_length
        )
        embed_dim = transformer.get_word_embedding_dimension()
        pooling = models.Pooling(embed_dim)
        cls_head = models.Dense(
            in_features=embed_dim,
            out_features=1,
            activation_function=torch.nn.Sigmoid(),
        )
        self._sent_transformer = SentenceTransformer(
            modules=[transformer, pooling, cls_head]
        )

        self._loss = SentTransformerBCELoss(self._sent_transformer)
        self._train_logger = SentLogger(log_dir, "loss")

    def train(self, *, train: IMDBData, **_) -> None:
        training_data = self._to_sent_transformer_inputs(train)

        self._sent_transformer.fit(
            train_objectives=[(training_data, self._loss)],
            epochs=self._epochs,
            warmup_steps=self._warmup_steps,
            callback=self._train_logger,
        )

    def predict(self, inputs: IMDBData) -> Iterable[np.ndarray]:
        for input_batch in inputs.iter(self._batch_size):
            yield self._sent_transformer.encode(
                input_batch["text"], convert_to_numpy=True
            )

    def save(self, dir_path: str) -> None:
        self._sent_transformer.save(dir_path)

    def load(self, dir_path: str) -> None:
        self._sent_transformer = SentenceTransformer(dir_path)

    def _to_sent_transformer_inputs(self, data: IMDBData) -> DataLoader:
        inputs = [InputExample(texts=[doc["text"]], label=doc["label"]) for doc in data]
        return DataLoader(inputs, batch_size=self._batch_size, shuffle=True)


class SentLogger:
    def __init__(self, log_dir: str, name: str) -> None:
        self._name = name
        self._writer = tf.summary.create_file_writer(log_dir)

    def __call__(self, score: float, epoch: int, steps: int) -> None:
        with self._writer.as_default():
            tf.summary.scalar(self._name, score, step=steps)


class SentTransformerBCELoss(torch.nn.Module):
    def __init__(self, sent_transformer: SentenceTransformer) -> None:
        super().__init__()
        self._sent_transformer = sent_transformer
        self._loss_fn = torch.nn.BCELoss()

    def forward(
        self, inputs: Sequence[dict[str, torch.Tensor]], labels: torch.Tensor
    ) -> torch.Tensor:
        pred_labels = self._sent_transformer(inputs[0])["sentence_embedding"]
        pred_labels = pred_labels[:, 0]
        return self._loss_fn(pred_labels, labels.float())


Model = SBertIMDB
