from typing import Iterable

import tensorflow as tf

from ..models.base_model import BaseModel


class BaseTask:
    """Defines interface for tasks."""

    @property
    def data_splits(self) -> list[str]:
        """Available data splits."""
        raise NotImplementedError()

    @property
    def metrics(self) -> dict:
        """Dictionary of metrics used by the task."""
        raise NotImplementedError()

    def get_data(self, split: str) -> tf.data.Dataset:
        """Returns specified split from the dataset as `tf.data.Dataset` of pairs
        (features, true_labels). For test split returns features only."""
        raise NotImplementedError()

    def evaluate(self, model: BaseModel) -> dict[str, float]:
        """Computes metrics for given `Iterable` of predicted test split labels."""
        raise NotImplementedError()

    def _eval_with_data(
        self, data: tf.data.Dataset, model: BaseModel
    ) -> dict[str, float]:
        """Evaluates the `model` on `data` with `self.metrics` used as metrics."""
        pass
