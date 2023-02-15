import os
from typing import Any, Iterable, Optional

import datasets
import numpy as np
import tensorflow as tf

from transformer_document_embedding import layers
from transformer_document_embedding.models.experimental_model import \
    ExperimentalModel
from transformer_document_embedding.tasks.imdb import IMDBData


class Doc2VecIMDB(ExperimentalModel):
    def __init__(
        self,
        *,
        log_dir: str,
        cls_head_kwargs: Optional[dict[str, Any]] = None,
        use_dm: bool = True,
        use_dbow: bool = True,
        dm_kwargs: Optional[dict[str, Any]] = None,
        dbow_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self._log_dir = log_dir

        if dm_kwargs is None:
            dm_kwargs = {}

        if dbow_kwargs is None:
            dbow_kwargs = {}

        if cls_head_kwargs is None:
            cls_head_kwargs = {}

        doc2vec_common_kwargs = {
            "workers": os.cpu_count(),
            "vector_size": 400,
        }
        for key, value in doc2vec_common_kwargs.items():
            dbow_kwargs.setdefault(key, value)
            dm_kwargs.setdefault(key, value)

        self._doc2vec = layers.Doc2Vec(
            log_dir=log_dir,
            use_dm=use_dm,
            use_dbow=use_dbow,
            dm_kwargs=dm_kwargs,
            dbow_kwargs=dbow_kwargs,
        )

        default_cls_kwargs = {
            "hidden_dim": 50,
            "hidden_dropout": 0,
            "hidden_activation": "relu",
            "learning_rate": 1e-3,
            "label_smoothing": 0,
            "epochs": 10,
            "batch_size": 2,
        }
        default_cls_kwargs.update(cls_head_kwargs)

        self._cls_head = layers.TensorflowClsHead(
            **default_cls_kwargs,
            log_dir=log_dir,
            input_dim=dm_kwargs["vector_size"] * int(use_dm)
            + dbow_kwargs["vector_size"] * int(use_dbow),
            output_activation=tf.nn.sigmoid,
            output_dim=1,
        )

    def train(
        self, *, train: IMDBData, unsupervised: IMDBData, test: IMDBData, **_
    ) -> None:
        doc2vec_train = datasets.combine.concatenate_datasets(
            [train, unsupervised, test]
        )
        doc2vec_train = doc2vec_train.shuffle()
        self._doc2vec.train(doc2vec_train)

        tf_ds = self._cls_head_dataset(train)

        self._cls_head.train(tf_ds)

    def predict(self, inputs: IMDBData) -> Iterable[np.ndarray]:
        tf_ds = self._cls_head_dataset(inputs, training=False)

        return self._cls_head.predict(tf_ds)

    def save(self, dir_path: str) -> None:
        self._doc2vec.save(Doc2VecIMDB._doc2vec_dir_path(dir_path))
        self._cls_head.save(Doc2VecIMDB._cls_head_dir_path(dir_path))

    def load(self, dir_path: str) -> None:
        self._doc2vec.load(Doc2VecIMDB._doc2vec_dir_path(dir_path))
        self._doc2vec.load(Doc2VecIMDB._cls_head_dir_path(dir_path))

    @staticmethod
    def _doc2vec_dir_path(dir_path: str) -> str:
        doc2vec_path = os.path.join(dir_path, "doc2vec")
        os.makedirs(doc2vec_path, exist_ok=True)
        return doc2vec_path

    @staticmethod
    def _cls_head_dir_path(dir_path: str) -> str:
        cls_path = os.path.join(dir_path, "cls_head")
        os.makedirs(cls_path, exist_ok=True)
        return cls_path

    def _cls_head_dataset(
        self, data: IMDBData, training: bool = True
    ) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices(
            [tf.convert_to_tensor(features) for features in self._doc2vec.predict(data)]
        )

        if training:
            labels_ds = data.to_tf_dataset(1, columns=["label"]).unbatch()
            ds = tf.data.Dataset.zip((ds, labels_ds), name="some_name")

        ds = ds.shuffle(25000) if training else ds
        ds = ds.batch(self._cls_head.batch_size)

        return ds


Model = Doc2VecIMDB
