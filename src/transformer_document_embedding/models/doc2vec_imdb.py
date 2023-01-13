import os
from typing import Any, Optional

import numpy as np
import tensorflow as tf

from transformer_document_embedding.models.doc2vec import Doc2Vec
from transformer_document_embedding.models.experimental_model import \
    ExperimentalModel
from transformer_document_embedding.tasks.imdb import IMDBData


class Doc2VecIMDB(ExperimentalModel):
    def __init__(
        self,
        *,
        log_dir: str,
        cls_head_epochs: int = 10,
        doc2vec_epochs: int = 10,
        cls_head_learning_rate=1e-3,
        doc2vec_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self._pv_dim = 400
        if doc2vec_kwargs is None:
            doc2vec_kwargs = {
                "workers": 6,
                "window": 5,
                "dm_concat": 1,
                "hs": 1,
            }
        self._doc2vec_epochs = doc2vec_epochs
        self._log_dir = log_dir
        self._cls_head_epochs = cls_head_epochs

        # Arguments to match the Paragraph Vector paper
        self._doc2vec = Doc2Vec(
            log_dir=log_dir,
            vector_size=self._pv_dim,
            **doc2vec_kwargs,
        )
        self._cls_head = tf.keras.Sequential(
            [
                tf.keras.layers.Input(self._pv_dim * 2),
                tf.keras.layers.Dense(50, activation=tf.nn.relu),
                tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
            ]
        )
        self._cls_head.compile(
            optimizer=tf.keras.optimizers.Adam(cls_head_learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
        )

    def train(self, training_data: IMDBData) -> None:
        self._doc2vec.train(
            training_data,
            epochs=self._doc2vec_epochs,
        )
        tf_ds = self._to_tf_dataset(training_data)

        self._cls_head.fit(
            tf_ds,
            epochs=self._cls_head_epochs,
            callbacks=[tf.keras.callbacks.TensorBoard(self._log_dir)],
        )

    def predict(self, inputs: IMDBData) -> np.ndarray:
        tf_ds = self._to_tf_dataset(inputs, training=False)

        return self._cls_head.predict(tf_ds)

    def save(self, dir_path: str) -> None:
        self._doc2vec.save(dir_path)
        self._cls_head.save(Doc2VecIMDB._get_cls_head_save_dir(dir_path))

    def load(self, dir_path: str) -> None:
        self._doc2vec.load(dir_path)
        new_cls_head = tf.keras.models.load_model(
            Doc2VecIMDB._get_cls_head_save_dir(dir_path)
        )
        if new_cls_head is not None:
            self._cls_head = new_cls_head

    @staticmethod
    def _get_cls_head_save_dir(save_dir: str) -> str:
        return os.path.join(save_dir, "softmax_head")

    def _to_tf_dataset(self, data: IMDBData, training: bool = True) -> tf.data.Dataset:
        if training:
            data = data.filter(
                lambda doc: doc["label"] >= 0,
                keep_in_memory=True,
                load_from_cache_file=False,
            )

        features_iter = self._doc2vec.predict(data)
        tf_ds = data.map(
            lambda _: {"features": next(features_iter)},
            remove_columns=["text", "id"],
            keep_in_memory=True,
            load_from_cache_file=False,
        )
        tf_ds = tf_ds.to_tf_dataset(1).unbatch()

        if training:
            tf_ds = tf_ds.map(lambda doc: (doc["features"], doc["label"]))
        else:
            tf_ds = tf_ds.map(lambda doc: doc["features"])

        tf_ds = tf_ds.shuffle(25000) if training else tf_ds
        tf_ds = tf_ds.batch(2)

        return tf_ds


Model = Doc2VecIMDB
