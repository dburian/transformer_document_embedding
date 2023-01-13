"""Doc2Vec's implementation from the `gensim` package.


"""

import functools
import logging
import os
import random
from typing import Any, Iterable, Iterator, Mapping, Optional

import numpy as np
import tensorflow as tf
from gensim.models import doc2vec
from gensim.models.callbacks import CallbackAny2Vec

from transformer_document_embedding.models.experimental_model import \
    ExperimentalModel

DataType = Iterable[Mapping[str, Any]]


class EmbeddingDifferencesCallback(CallbackAny2Vec):
    def __init__(
        self,
        *,
        log_dir: str,
        dm: bool,
        min_doc_id: int,
        max_doc_id: int,
        num_samples: int,
        seed: Optional[int] = None,
    ) -> None:
        self._tb_writer = tf.summary.create_file_writer(log_dir)
        self._epoch = 0
        self._scalar_name = f"doc2vec_{'dm' if dm else 'dbow'}_embed_diff"
        if seed:
            random.seed(seed)
        self._sample_doc_ids = [
            random.randint(min_doc_id, max_doc_id) for _ in range(num_samples)
        ]
        self._last_embed = None

    def on_epoch_end(self, model: doc2vec.Doc2Vec) -> None:
        new_embed = np.stack([model.dv[id] for id in self._sample_doc_ids], axis=0)
        new_embed = new_embed / np.reshape(np.linalg.norm(new_embed, axis=1), (-1, 1))
        if self._last_embed is not None:
            with self._tb_writer.as_default():
                distances = 1 - np.sum(new_embed * self._last_embed, axis=1)
                mean_distance = np.mean(distances)
                std_distance = np.std(distances)
                logging.info(
                    f"{self._scalar_name}, Epoch {self._epoch}:"
                    f" {mean_distance:.5f}+-{std_distance:.5f}"
                )
                tf.summary.scalar(
                    f"{self._scalar_name}_mean", mean_distance, self._epoch
                )
                tf.summary.scalar(f"{self._scalar_name}_std", std_distance, self._epoch)

        self._epoch += 1
        self._last_embed = new_embed


class Doc2Vec(ExperimentalModel):
    """Implementation of Doc2Vec model using the `gensim` package."""

    class GensimCorpus:
        def __init__(self, dataset: DataType) -> None:
            self._dataset = dataset

        def __iter__(self) -> Iterable[doc2vec.TaggedDocument]:
            for doc in self._dataset:
                yield doc2vec.TaggedDocument(
                    Doc2Vec._preprocess_text(doc["text"]), [doc["id"]]
                )

    def __init__(self, *, log_dir: str, **gensim_kwargs) -> None:
        """Initializes the gensim model with `gensim_kwargs`."""
        self._log_dir = log_dir
        self._dm = doc2vec.Doc2Vec(
            dm_tag_count=1,
            **gensim_kwargs,
        )
        self._dbow = doc2vec.Doc2Vec(
            dm=0,
            **gensim_kwargs,
        )

    def train(
        self,
        training_data: DataType,
        num_eval_samples: Optional[int] = None,
        **train_kwargs,
    ) -> None:
        dm_callback, dbow_callback = None, None

        def reduce_extremes(
            extremes: tuple[float, float], doc: dict[str, Any]
        ) -> tuple[float, float]:
            return (
                min(extremes[0], doc["id"]),
                max(extremes[1], doc["id"]),
            )

        min_doc_id, max_doc_id = functools.reduce(
            reduce_extremes, training_data, (float("inf"), float("-inf"))
        )

        num_samples = (
            num_eval_samples
            if num_eval_samples is not None
            else min(5000, int(0.05 * (max_doc_id - min_doc_id)))
        )
        dm_callback = EmbeddingDifferencesCallback(
            log_dir=self._log_dir,
            dm=True,
            min_doc_id=min_doc_id,
            max_doc_id=max_doc_id,
            num_samples=num_samples,
        )
        dbow_callback = EmbeddingDifferencesCallback(
            log_dir=self._log_dir,
            dm=False,
            min_doc_id=min_doc_id,
            max_doc_id=max_doc_id,
            num_samples=num_samples,
        )

        gensim_corpus = Doc2Vec.GensimCorpus(training_data)
        self._dm.build_vocab(gensim_corpus)
        self._dbow.build_vocab(gensim_corpus)

        self._dm.train(
            gensim_corpus,
            total_examples=self._dm.corpus_count,
            callbacks=[dm_callback] if dm_callback is not None else [],
            **train_kwargs,
        )
        self._dbow.train(
            gensim_corpus,
            total_examples=self._dbow.corpus_count,
            callbacks=[dbow_callback] if dbow_callback is not None else [],
            **train_kwargs,
        )

    def predict(self, inputs: DataType) -> Iterator[np.ndarray]:
        for doc in inputs:
            doc_id = doc["id"]
            dm_vector, dbow_vector = None, None
            if doc_id in self._dm.dv and doc_id in self._dbow.dv:
                dm_vector, dbow_vector = self._dm.dv[doc_id], self._dbow.dv[doc_id]
            else:
                words = Doc2Vec._preprocess_text(doc["text"])
                dm_vector = self._dm.infer_vector(words)
                dbow_vector = self._dbow.infer_vector(words)

            yield np.concatenate([dm_vector, dbow_vector])

    def save(self, dir_path: str) -> None:
        self._dm.save(Doc2Vec._get_model_path(dir_path))
        self._dbow.save(Doc2Vec._get_model_path(dir_path, dm=False))

    def load(self, dir_path: str) -> None:
        self._dm.load(Doc2Vec._get_model_path(dir_path))
        self._dbow.load(Doc2Vec._get_model_path(dir_path, dm=False))

    @staticmethod
    def _get_model_path(dir_path: str, dm: bool = True) -> str:
        return os.path.join(dir_path, "dm" if dm else "dbow")

    @staticmethod
    def _preprocess_text(text: str) -> list[str]:
        words = text.split()
        while len(words) < 9:
            words.insert(0, "NULL")

        return words
