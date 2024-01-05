from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Iterable, Optional
    import numpy as np
    from transformer_document_embedding.tasks.experimental_task import ExperimentalTask


class ExperimentalModel:
    """Defines the minimal interface for models."""

    def train(
        self,
        task: ExperimentalTask,
        log_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Trains the model.

        Args:
            - task: task to train on with optional validation data,
            - log_dir: directory where to save logs,
            - model_dir: directory where to save the model's parameters,
        """
        raise NotImplementedError()

    def predict(self, inputs) -> Iterable[np.ndarray]:
        """Returns predictions for given `inputs`."""
        raise NotImplementedError()

    def save(self, dir_path: str) -> None:
        """Saves the model's weights into given directory."""
        raise NotImplementedError()

    def load(self, dir_path: str, *, strict: bool = False) -> None:
        """Loads the model's weights from given directory.

        Parameters
        ----------
        dir_path: str
            Path to dir where to save the model.
        strict: bool, default = True
            If true, loading will fail if some parameters are unexpected or missing.
        """
        raise NotImplementedError()
