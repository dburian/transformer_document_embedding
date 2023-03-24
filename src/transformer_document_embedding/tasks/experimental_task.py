class ExperimentalTask:
    """Defines the minimal interface for tasks."""

    @property
    def train(self):
        """Returns all training data."""
        return None

    @property
    def test(self):
        """Returns testing inputs."""
        return None

    @property
    def validation(self):
        """Returns validation data."""
        return None

    @property
    def has_validation(self) -> bool:
        return self.validation is not None

    def evaluate(self, pred_batches) -> dict[str, float]:
        """Evaluates given test predictions on task-specific metrics.

        Returns dictionary mapping metric names to their respecitve score.
        """
        raise NotImplementedError()
