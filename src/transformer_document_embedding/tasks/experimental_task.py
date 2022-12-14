class ExperimentalTask:
    """Defines the minimal interface for tasks."""

    @property
    def train(self):
        """
        Returns all training data.
        """
        raise NotImplementedError()

    @property
    def test(self):
        """
        Returns testing inputs.
        """
        raise NotImplementedError()

    def evaluate(self, test_predictions) -> dict[str, float]:
        """
        Evaluates given test predictions on task-specific metrics.

        Returns dictionary mapping metric names to their respecitve score.
        """
        raise NotImplementedError()
