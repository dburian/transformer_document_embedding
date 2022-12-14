class ExperimentalModel:
    """Defines the minimal interface for models."""

    def train(self, training_data) -> None:
        """Trains the model from the `training_data`."""
        raise NotImplementedError()

    def predict(self, inputs):
        """Returns predictions for given `inputs`."""
        raise NotImplementedError()

    def save(self, dir: str) -> None:
        """Saves the model's weights into given directory."""
        raise NotImplementedError()

    def load(self, dir: str) -> None:
        """Loads the model's weights from given directory."""
        raise NotImplementedError()
