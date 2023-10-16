from typing import Iterable, Optional

import numpy as np
from datasets import Dataset
from gensim.corpora import Dictionary
from gensim.matutils import sparse2full
from gensim.models import TfidfModel

from transformer_document_embedding.baselines.experimental_model import (
    Baseline,
)
from transformer_document_embedding.tasks.experimental_task import ExperimentalTask


class TFIDF(Baseline):
    def __init__(
        self,
        word_filter_no_below: int = 10,
        word_filter_no_above: float = 0.8,
        smartirs: str = "lfn",
    ) -> None:
        self._word_filter_no_below = word_filter_no_below
        self._word_filter_no_above = word_filter_no_above

        self._smartirs = smartirs

    def train(
        self,
        task: ExperimentalTask,
        *,
        log_dir: Optional[str] = None,
        save_best_path: Optional[str] = None,
        early_stopping: bool = False,
    ) -> None:
        pass

    def predict(self, inputs: Dataset) -> Iterable[np.ndarray]:
        words_dataset = inputs.map(
            lambda doc: {"words": doc["text"].lower().split()},
            remove_columns=["id", "text"],
        )
        word_dict = self._get_word_dictionary(words_dataset)

        model = TfidfModel(
            smartirs=self._smartirs,
            dictionary=word_dict,
        )
        embed_dim = len(word_dict.keys())

        for article_words in words_dataset["words"]:
            sparse_vec = model[word_dict.doc2bow(article_words)]
            yield sparse2full(sparse_vec, embed_dim)

    def _get_word_dictionary(
        self,
        dataset: Dataset,
        *,
        no_below: Optional[int] = None,
        no_above: Optional[float] = None,
    ) -> Dictionary:
        no_below = no_below if no_below is not None else self._word_filter_no_below
        no_above = no_above if no_above is not None else self._word_filter_no_above

        gensim_dict = Dictionary(dataset["words"])

        gensim_dict.filter_extremes(no_below=no_below, no_above=no_above)

        return gensim_dict

    def save(self, dir_path: str) -> None:
        pass

    def load(self, dir_path: str) -> None:
        pass
