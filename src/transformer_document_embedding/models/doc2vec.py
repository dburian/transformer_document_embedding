import os
from typing import Any, Iterable, Iterator

import numpy as np
from gensim.models import doc2vec

from transformer_document_embedding.models.experimental_model import \
    ExperimentalModel

DataType = Iterable[dict[str, Any]]


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
        self._dm = doc2vec.Doc2Vec(dm_tag_count=1, **gensim_kwargs)
        self._dbow = doc2vec.Doc2Vec(dm=0, **gensim_kwargs)

    def train(self, training_data: DataType, **train_kwargs) -> None:
        gensim_corpus = Doc2Vec.GensimCorpus(training_data)
        self._dm.build_vocab(gensim_corpus)
        self._dbow.build_vocab(gensim_corpus)

        self._dm.train(
            gensim_corpus, total_examples=self._dm.corpus_count, **train_kwargs
        )
        self._dbow.train(
            gensim_corpus, total_examples=self._dbow.corpus_count, **train_kwargs
        )

    def predict(self, testing_inputs: DataType) -> Iterator[np.ndarray]:
        for doc in testing_inputs:
            doc_id = doc["id"]
            dm_vector, dbow_vector = None, None
            if doc_id in self._dm and doc_id in self._dbow:
                dm_vector, dbow_vector = self._dm.dv[doc_id], self._dbow.dv[doc_id]
            else:
                words = Doc2Vec._preprocess_text(doc["text"])
                dm_vector = self._dm.infer_vector(words)
                dbow_vector = self._dbow.infer_vector(words)

            yield np.concatenate([dm_vector, dbow_vector])

    def save(self, save_dir: str) -> None:
        self._dm.save(Doc2Vec._get_model_path(save_dir))
        self._dbow.save(Doc2Vec._get_model_path(save_dir, dm=False))

    def load(self, save_dir: str) -> None:
        self._dm.load(Doc2Vec._get_model_path(save_dir))
        self._dbow.load(Doc2Vec._get_model_path(save_dir, dm=False))

    @staticmethod
    def _get_model_path(dir_path: str, dm: bool = True) -> str:
        return os.path.join(dir_path, "dm" if dm else "dbow")

    @staticmethod
    def _preprocess_text(text: str) -> list[str]:
        words = text.split()
        while len(words) < 9:
            words.insert(0, "NULL")

        return words
