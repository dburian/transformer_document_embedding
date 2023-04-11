import os
from typing import Any, Iterable

import numpy as np
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset

from transformer_document_embedding.tasks.hf_task import HFTask


class WikipediaWines(HFTask):
    # TODO: This is super unknown. How do we split the dataset into splits?
    def __init__(self, datasets_dir: str, **kwargs) -> None:
        # Dataset already contains ids
        kwargs["add_ids"] = False
        super().__init__(os.path.join(datasets_dir, "wikipedia_wines.py"), **kwargs)

    def test(self) -> Dataset:
        test = self._get_split("test")
        assert test is not None
        return test.remove_columns("target_ids")

    def _retrieve_dataset(self) -> DatasetDict:
        return DatasetDict(
            {
                "articles": load_dataset(self._path, "articles"),
                "sims": load_dataset(self._path, "sims"),
            }
        )

    def _construct_splits(self, dataset: DatasetDict) -> DatasetDict:
        articles = dataset["articles"]
        sims = dataset["sims"]

        def article_to_train(article: dict[str, Any]) -> dict[str, Any]:
            return article

        return super()._construct_splits(
            DatasetDict(
                {
                    "test": sims,
                    "unsupervised": articles,
                    "train": articles.map(article_to_train),
                }
            )
        )

    def evaluate(self, pred_batches: Iterable[np.ndarray]) -> dict[str, float]:
        # TODO: pred_batches are embeddings
        pass
