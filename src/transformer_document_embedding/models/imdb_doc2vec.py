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
            workers=6,
            vector_size=self._pv_dim,
            window=10,
            shrink_windows=True,
            dm_concat=1,
            hs=1,
        )
        self._cls_head = tf.keras.Sequential(
            [
                tf.keras.layers.Input(self._pv_dim * 2),
                tf.keras.layers.Dense(50, activation=tf.nn.relu),
                tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
            ]
        )
        self._cls_head.compile(
            optimizer=tf.keras.optimizers.Adam(softmax_learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
        )
        self._log_dir = log_dir

    def train(self, training_data: IMDBData) -> None:
        self._doc2vec.train(
            training_data,
            min_doc_id=25000,
            max_doc_id=100000 - 1,
            epochs=5,
        )
        tf_ds = self._to_tf_dataset(training_data)

        self._cls_head.fit(
            tf_ds,
            callbacks=[tf.keras.callbacks.TensorBoard(self._log_dir)],
        )

    def predict(self, inputs: IMDBData) -> np.ndarray:
        tf_ds = self._to_tf_dataset(inputs, training=False)

        return self._cls_head.predict(tf_ds)

    def save(self, dir_path: str) -> None:
        self._doc2vec.save(dir_path)
        self._cls_head.save(IMDBDoc2Vec._get_cls_head_save_dir(dir_path))

    def load(self, dir_path: str) -> None:
        self._doc2vec.load(dir_path)
        new_cls_head = tf.keras.models.load_model(
            IMDBDoc2Vec._get_cls_head_save_dir(dir_path)
        )
        if new_cls_head is not None:
            self._cls_head = new_cls_head

    @staticmethod
    def _get_cls_head_save_dir(save_dir: str) -> str:
        return os.path.join(save_dir, "softmax_head")

    def _to_tf_dataset(self, data: IMDBData, training: bool = True) -> tf.data.Dataset:
        if training:
            data = data.filter(lambda doc: "label" in doc)

        features_iter = self._doc2vec.predict(data)
        tf_ds = data.map(
            lambda _: {"features": next(features_iter)},
            remove_columns=["text", "id"],
            keep_in_memory=True,
        )
        tf_ds = tf_ds.to_tf_dataset(1).unbatch()

        if training:
            tf_ds = tf_ds.map(lambda doc: (doc["features"], doc["label"]))
        else:
            tf_ds = tf_ds.map(lambda doc: doc["features"])

        tf_ds = tf_ds.shuffle(1000) if training else tf_ds
        tf_ds = tf_ds.batch(2)

        return tf_ds


Model = IMDBDoc2Vec
