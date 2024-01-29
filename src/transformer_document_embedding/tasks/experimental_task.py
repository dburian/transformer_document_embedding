from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class ExperimentalTask:
    """Defines the minimal interface for tasks.

    For each split there is a property. `train` and `test` properties are
    required, while other splits can return `None` to signify the dataset does
    not contain any data in given split.
    """

    @property
    def splits(self) -> dict[str, Any]:
        """Returns dictionary of all available splits."""
        raise NotImplementedError()

    def evaluate(self, split, pred_batches) -> dict[str, float]:
        """Evaluates given split predictions on task-specific metrics.

        Returns dictionary mapping metric names to their respective score.
        """
        raise NotImplementedError()
