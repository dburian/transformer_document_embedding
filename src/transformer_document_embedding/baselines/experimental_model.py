from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Iterable, Optional
    import numpy as np
    from transformer_document_embedding.tasks.experimental_task import ExperimentalTask


# TODO: Rename just to baseline?
class ExperimentalModel:
    """Defines the minimal interface for models."""

    def train(
        self,
        task: ExperimentalTask,
        *,
        early_stopping: Optional[bool],
        save_best_path: Optional[str],
        log_dir: Optional[str],
    ) -> None:
        """Trains the model.

        Args:
            - task: task to train on with optional validation data,
            - early_stopping: whether to stop when validation loss stops decreasing,
            - save_best_path: Path to save (and then restore from) the
              best checkpoint, i.e. with the lowest validation loss or at the
              end of a training.
            - log_dir: directory where to save logs,
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
