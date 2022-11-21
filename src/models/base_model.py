from typing import Any, Union


class BaseModel:
    """Type for all model-like types. Meant to bridge the gap between
    tensorflow, pytorch or other models. Modeled to `tf.keras.Model`."""

    def fit(self, x: Any = None, y: Any = None) -> None:
        """Train the model with features `x` and labels `y`."""
        raise NotImplementedError()

    def predict(self, x: Any = None) -> None:
        """Predict labels for features `x`."""
        raise NotImplementedError()

    def evaluate(
        self, x: Any = None, y: Any = None, return_dict: bool = True
    ) -> Union[dict[str, float], list[float]]:
        """
        Evaluate model on given data. If `return_dict` is True, return
        dictionary with metric names as keys, otherwise return just the list of
        results.
        """
        raise NotImplementedError()
