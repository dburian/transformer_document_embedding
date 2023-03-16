from typing import Iterable, Optional

import numpy as np
import tensorflow as tf


class TensorflowClsHead:
    def __init__(
        self,
        *,
        log_dir: str,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_activation: str | tf.keras.layers.Layer,
        hidden_dropout: float,
        output_activation: str | tf.keras.layers.Layer,
        epochs: int,
        learning_rate: float,
        label_smoothing: Optional[float] = None,
        batch_size: int,
    ) -> None:
        self._epochs = epochs
        self._log_dir = log_dir
        self._batch_size = batch_size

        self._model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(input_dim),
                tf.keras.layers.Dense(hidden_dim, activation=hidden_activation),
                tf.keras.layers.Dropout(hidden_dropout),
                tf.keras.layers.Dense(output_dim, activation=output_activation),
            ]
        )
        loss = None
        if output_dim == 1:
            loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
        elif label_smoothing is not None:
            loss = tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=label_smoothing
            )
        else:
            loss = tf.keras.losses.SparseCategoricalCrossentropy()

        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=loss,
            metrics=[
                tf.keras.metrics.BinaryAccuracy()
                if output_dim == 1
                else tf.keras.metrics.CategoricalAccuracy()
            ],
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def train(self, training_data: tf.data.Dataset) -> None:
        self._model.fit(
            training_data,
            epochs=self._epochs,
            callbacks=[tf.keras.callbacks.TensorBoard(self._log_dir)],
        )

    def predict(self, inputs: tf.data.Dataset) -> Iterable[np.ndarray]:
        return self._model.predict(inputs)

    def save(self, dir_path: str) -> None:
        self._model.save(dir_path)

    def load(self, dir_path: str) -> None:
        loaded_model = tf.keras.models.load_model(dir_path)
        assert loaded_model is not None
        self._model = loaded_model


class ClsHead(tf.keras.Sequential):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_activation: str | tf.keras.layers.Layer,
        hidden_dropout: float,
        output_activation: str | tf.keras.layers.Layer,
    ) -> None:
        super().__init__(
            [
                tf.keras.layers.Input(input_dim),
                tf.keras.layers.Dense(hidden_dim, activation=hidden_activation),
                tf.keras.layers.Dropout(hidden_dropout),
                tf.keras.layers.Dense(output_dim, activation=output_activation),
            ]
        )
