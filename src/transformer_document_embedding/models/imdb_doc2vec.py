import os

import numpy as np
import tensorflow as tf

from transformer_document_embedding.models.doc2vec import Doc2Vec
from transformer_document_embedding.models.experimental_model import \
    ExperimentalModel
from transformer_document_embedding.tasks.imdb import IMDBData


class IMDBDoc2Vec(ExperimentalModel):
    def __init__(self, *, log_dir: str, softmax_learning_rate=1e-3) -> None:
        self._pv_dim = 400

        # Arguments to match the Paragraph Vector paper
        self._doc2vec = Doc2Vec(
            log_dir=log_dir,
            vector_size=self._pv_dim,
            window=10,
            shrink_windows=True,
            dm_concat=1,
            hs=1,
        )
        self._softmax_head = tf.keras.Sequential(
            [
                tf.keras.layers.Input(self._pv_dim * 2),
                tf.keras.layers.Dense(50, activation=tf.nn.relu),
                tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
            ]
        )
        self._softmax_head.compile(
            optimizer=tf.keras.optimizers.Adam(softmax_learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
        )
        self._log_dir = log_dir

    def train(self, training_data: IMDBData) -> None:
        self._doc2vec.train(training_data)
        tf_ds = self._to_tf_dataset(training_data)

        self._softmax_head.fit(
            tf_ds,
            callbacks=[tf.keras.callbacks.TensorBoard(self._log_dir)],
        )

    def predict(self, test_data: IMDBData) -> np.ndarray:
        tf_ds = self._to_tf_dataset(test_data, training=False)

        return self._softmax_head.predict(tf_ds).to_numpy()

    def save(self, save_dir: str) -> None:
        self._doc2vec.save(save_dir)
        self._softmax_head.save(IMDBDoc2Vec._get_softmax_save_dir(save_dir))

    def load(self, save_dir: str) -> None:
        self._doc2vec.load(save_dir)
        new_softmax_head = tf.keras.models.load_model(
            IMDBDoc2Vec._get_softmax_save_dir(save_dir)
        )
        if new_softmax_head is not None:
            self._softmax_head = new_softmax_head

    @staticmethod
    def _get_softmax_save_dir(save_dir: str) -> str:
        return os.path.join(save_dir, "softmax_head")

    def _to_tf_dataset(self, data: IMDBData, training: bool = True) -> tf.data.Dataset:
        if training:
            data = data.filter(lambda doc: "label" in doc)

        features_iter = self._doc2vec.predict(data)
        tf_ds = data.map(
            lambda _: {"features": next(features_iter)},
            remove_columns=["text", "id"],
        )
        tf_ds = tf_ds.to_tf_dataset(1).unbatch()

        map_train = lambda doc: (doc["features"], doc["label"])
        map_test = lambda doc: doc["features"]
        tf_ds = tf_ds.map(map_train if training else map_test)

        if training:
            tf_ds.shuffle(1000)

        tf_ds.batch(2)

        return tf_ds


Model = IMDBDoc2Vec
