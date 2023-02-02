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
                # Disregarding linearly decreasing learning rate
                distances /= model.alpha
                mean_distance = np.mean(distances)
                std_distance = np.std(distances)
                logging.info(
                    "%s, Epoch %s: %.5f+-%.5f",
                    self._scalar_name,
                    self._epoch,
                    mean_distance,
                    std_distance,
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

    def __init__(
        self,
        *,
        log_dir: str,
        use_dm: bool = True,
        use_dbow: bool = True,
        dm_kwargs: Optional[dict[str, Any]] = None,
        dbow_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initializes the gensim model with `gensim_kwargs`."""
        self._log_dir = log_dir
        if dm_kwargs is None:
            dm_kwargs = {}
        if dbow_kwargs is None:
            dbow_kwargs = {}

        self._dm = (
            doc2vec.Doc2Vec(
                dm_tag_count=1,
                **dm_kwargs,
            )
            if use_dm
            else None
        )
        self._dbow = (
            doc2vec.Doc2Vec(
                dm=0,
                **dbow_kwargs,
            )
            if use_dbow
            else None
        )
        self._with_dm = lambda f: f(self._dm) if use_dm else lambda: None
        self._with_dbow = lambda f: f(self._dbow) if use_dbow else lambda: None

    def train(
        self,
        training_data: DataType,
        num_eval_samples: Optional[int] = None,
    ) -> None:
        dm_callback, dbow_callback = self._create_training_cbs(
            training_data, num_eval_samples
        )

        gensim_corpus = Doc2Vec.GensimCorpus(training_data)

        self._with_dm(lambda dm: dm.build_vocab(gensim_corpus))
        self._with_dbow(lambda dbow: dbow.build_vocab(gensim_corpus))

        self._with_dm(
            lambda dm: dm.train(
                gensim_corpus,
                total_examples=dm.corpus_count,
                callbacks=[dm_callback],
                epochs=dm.epochs,
            )
        )
        self._with_dbow(
            lambda dbow: dbow.train(
                gensim_corpus,
                total_examples=dbow.corpus_count,
                callbacks=[dbow_callback],
                epochs=dbow.epochs,
            )
        )

    def predict(self, inputs: DataType) -> Iterator[np.ndarray]:
        for doc in inputs:
            doc_id = doc["id"]
            embedding = []
            if (self._dm is None or doc_id in self._dm.dv) and (
                self._dbow is None or doc_id in self._dbow.dv
            ):
                if self._dm is not None:
                    embedding.append(self._dm.dv[doc_id])
                if self._dbow is not None:
                    embedding.append(self._dbow.dv[doc_id])

            else:
                words = Doc2Vec._preprocess_text(doc["text"])
                if self._dm is not None:
                    embedding.append(self._dm.infer_vector(words))
                if self._dbow is not None:
                    embedding.append(self._dbow.infer_vector(words))

            yield np.concatenate(embedding)

    def save(self, dir_path: str) -> None:
        self._with_dm(lambda dm: dm.save(Doc2Vec._get_model_path(dir_path)))
        self._with_dbow(
            lambda dbow: dbow.save(Doc2Vec._get_model_path(dir_path, dm=False))
        )

    def load(self, dir_path: str) -> None:
        self._with_dm(lambda dm: dm.load(Doc2Vec._get_model_path(dir_path)))
        self._with_dbow(
            lambda dbow: dbow.load(Doc2Vec._get_model_path(dir_path, dm=False))
        )

    def _create_training_cbs(
        self, training_data: DataType, num_eval_samples: Optional[int] = None
    ) -> list[CallbackAny2Vec]:
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
        min_doc_id, max_doc_id = int(min_doc_id), int(max_doc_id)

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

        return [dm_callback, dbow_callback]

    @staticmethod
    def _get_model_path(
        dir_path: str,
        # pylint disable=invalid-name
        dm: bool = True,
    ) -> str:
        return os.path.join(dir_path, "dm" if dm else "dbow")

    @staticmethod
    def _preprocess_text(
        text: str,
        # window: int,
    ) -> list[str]:
        words = text.split()
        return words
        # while len(words) < window * 2:
        #     words.insert(0, "NULL")

        # return words
