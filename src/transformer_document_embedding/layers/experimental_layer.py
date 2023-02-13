from typing import Iterable

import numpy as np


class ExperimentalLayer:
    """Layer of experimental model."""

    def train(self, training_data) -> None:
        """Trains on provided data.

        Args:
            training_data: data to be trained on
        """
        raise NotImplementedError()

    def predict(self, inputs) -> Iterable[np.ndarray]:
        """Predicts outputs on given inputs.

        Args:
            inputs: inputs to the layer
        """
        raise NotImplementedError()

    def save(self, dir_path: str) -> None:
        """Saves the model's weights into given directory."""
        raise NotImplementedError()

    def load(self, dir_path: str) -> None:
        """Loads the model's weights from given directory."""
        raise NotImplementedError()
