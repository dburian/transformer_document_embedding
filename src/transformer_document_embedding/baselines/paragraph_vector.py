import os
from typing import Any, Iterable, Optional

import datasets
import numpy as np
import tensorflow as tf
from datasets.arrow_dataset import Dataset

from transformer_document_embedding.baselines.experimental_model import \
    ExperimentalModel
from transformer_document_embedding.models.paragraph_vector import \
    ParagraphVector
from transformer_document_embedding.models.tf.cls_head import ClsHead
from transformer_document_embedding.tasks.imdb import IMDBClassification
from transformer_document_embedding.utils.gensim.data import GensimCorpus


class ParagraphVectorIMDB(ExperimentalModel):
    def __init__(
        self,
        cls_head_kwargs: Optional[dict[str, Any]] = None,
        dm_kwargs: Optional[dict[str, Any]] = None,
        dbow_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        # TODO: With so many handlers, maybe create separate dataclasses just
        # for configuration arguments. Its like 45 lines of just argument parsing :(.
        if dm_kwargs is not None:
            dm_kwargs["dm_tag_count"] = 1

        self._pv = ParagraphVector(dm_kwargs=dm_kwargs, dbow_kwargs=dbow_kwargs)

        if cls_head_kwargs is None:
            cls_head_kwargs = {}

        self._batch_size = cls_head_kwargs.pop("batch_size", 2)
        self._cls_epochs = cls_head_kwargs.pop("epochs", 10)

        default_cls_kwargs = {
            "hidden_dim": 50,
            "hidden_dropout": 0,
            "hidden_activation": "relu",
        }
        default_cls_kwargs.update(cls_head_kwargs)

        force_cls_kwargs = {
            "input_dim": self._pv.vector_size,
            "output_activation": tf.nn.sigmoid,
            "output_dim": 1,
        }

        for key, value in force_cls_kwargs.items():
            default_cls_kwargs[key] = value

        compile_kwargs = {
            "label_smoothing": 0.1,
            "learning_rate": 1e-3,
        }
        for key, value in compile_kwargs.items():
            compile_kwargs[key] = default_cls_kwargs.pop(key, value)

        self._cls_head = ClsHead(**default_cls_kwargs)
        self._cls_head.compile(
            optimizer=tf.keras.optimizers.Adam(compile_kwargs["learning_rate"]),
            loss=tf.keras.losses.BinaryCrossentropy(
                label_smoothing=compile_kwargs["label_smoothing"]
            ),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
        )

    def train(
        self,
        task: IMDBClassification,
        *,
        log_dir: Optional[str] = None,
        save_best_path: Optional[str] = None,
        early_stopping: bool = False,
    ) -> None:
        all_datasets = [task.test, task.train]
        if task.unsupervised is not None:
            all_datasets.append(task.unsupervised)

        pv_train = datasets.combine.concatenate_datasets(all_datasets).shuffle()
        pv_train = GensimCorpus(pv_train)

        for module in self._pv.modules:
            module.build_vocab(pv_train)
            module.train(
                pv_train,
                total_examples=module.corpus_count,
                epochs=module.epochs,
            )
        cls_train = self._feature_dataset(task.train, training=True)

        callbacks = []
        if log_dir is not None:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir))

        cls_val = None
        if task.validation is not None:
            cls_val = self._feature_dataset(task.validation, training=True)

            if early_stopping:
                callbacks.append(
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        min_delta=0,
                        patience=3,
                        verbose=0,
                        mode="min",
                        restore_best_weights=True,
                    )
                )

            if save_best_path is not None:
                callbacks.append(
                    tf.keras.callbacks.ModelCheckpoint(
                        self._cls_head_dirpath(save_best_path),
                        monitor="val_loss",
                        save_best_only=True,
                        save_weights_only=False,
                        mode="min",
                        save_freq="epoch",
                    )
                )

        self._cls_head.fit(
            cls_train,
            epochs=self._cls_epochs,
            validation_data=cls_val,
            callbacks=callbacks,
        )

        if save_best_path is not None:
            self._pv.save(self._pv_dirpath(save_best_path))
            if task.validation is None:
                self._cls_head.save(self._cls_head_dirpath(save_best_path))

    def predict(self, inputs: Dataset) -> Iterable[np.ndarray]:
        tf_ds = self._feature_dataset(inputs, training=False)

        return self._cls_head.predict(tf_ds)

    def save(self, dir_path: str) -> None:
        self._pv.save(self._pv_dirpath(dir_path))
        self._cls_head.save(
            self._cls_head_dirpath(dir_path), overwrite=True, include_optimizer=True
        )

    def load(self, dir_path: str) -> None:
        self._pv.load(self._pv_dirpath(dir_path))
        new_cls_head = tf.keras.models.load_model(self._cls_head_dirpath(dir_path))
        assert new_cls_head is not None
        self._cls_head = new_cls_head

    @classmethod
    def _pv_dirpath(cls, dir_path: str) -> str:
        new_dir = os.path.join(dir_path, "paragraph_vector")
        os.makedirs(new_dir, exist_ok=True)
        return new_dir

    @classmethod
    def _cls_head_dirpath(cls, dir_path: str) -> str:
        new_dir = os.path.join(dir_path, "cls_head")
        os.makedirs(new_dir, exist_ok=True)
        return new_dir

    def _feature_dataset(self, data: Dataset, *, training: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices(
            [tf.convert_to_tensor(self._pv.get_vector(doc["id"])) for doc in data]
        )

        if training:
            labels_ds = data.to_tf_dataset(1, columns=["label"]).unbatch()
            ds = tf.data.Dataset.zip((ds, labels_ds))

        ds = ds.shuffle(25000) if training else ds
        ds = ds.batch(self._batch_size)

        return ds
