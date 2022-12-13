from typing import Iterable

import datasets
import tensorflow as tf

from transformer_document_embedding.tasks import BaseTask


class IMDBClassification(BaseTask):
    """Classification task done using the IMDB dataset"""

    def __init__(self) -> None:
        self.train = None
        self.test = None

    @property
    def metrics(self) -> list[tf.metrics.Metric]:
        return [tf.metrics.BinaryAccuracy()]

    @property
    def data_splits(self) -> list[str]:
        return ["train", "test"]

    def get_data(self, split: str) -> tf.data.Dataset:
        if split not in self.data_splits:
            raise TypeError(
                f"Data split {split} is not supported by {self.__class__} task."
            )

        if getattr(self, split) is None:
            setattr(self, split, datasets.load_dataset("imdb", split=split))

        if split == "train":
            return tf.data.Dataset.from_generator(
                lambda: map(lambda ex: (ex["text"], ex["label"]), self.train),
                output_signature=(
                    tf.TensorSpec(shape=(), dtype=tf.string),
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                ),
            )

        return tf.data.Dataset.from_generator(
            lambda: map(lambda ex: ex["text"], self.test),
            output_signature=tf.TensorSpec(shape=(), dtype=tf.string),
        )

    def evaluate(self, y_pred: Iterable) -> dict:
        return super().evaluate(y_pred)
