from typing import Iterable

import numpy as np


class ExperimentalModel:
    """Defines the minimal interface for models."""

    def train(self, *, train, unsupervised, test, **kwargs) -> None:
        """Trains the model.

        Args:
            - train: training input-output pairs,
            - unsupervised: unsupervised inputs,
            - test: testing inputs,
        """
        raise NotImplementedError()

    def predict(self, inputs) -> Iterable[np.ndarray]:
        """Returns predictions for given `inputs`."""
        raise NotImplementedError()

    def save(self, dir_path: str) -> None:
        """Saves the model's weights into given directory."""
        raise NotImplementedError()

    def load(self, dir_path: str) -> None:
        """Loads the model's weights from given directory."""
        raise NotImplementedError()
