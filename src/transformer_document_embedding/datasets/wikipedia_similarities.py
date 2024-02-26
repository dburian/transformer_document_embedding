import copy
import math
import random
from typing import Any

from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from transformer_document_embedding.datasets import col
from transformer_document_embedding.datasets.document_dataset import (
    DocumentDataset,
    EvaluationKind,
)


class WikipediaSimilarities(DocumentDataset):
    AVAILABLE_DATASETS = ["wine", "game"]

    def __init__(
        self,
        dataset: str,
        path: str,
        **kwargs,
    ) -> None:
        assert (
            dataset in self.AVAILABLE_DATASETS
        ), f"`dataset` must be one of: {', '.join(self.AVAILABLE_DATASETS)}"

        super().__init__(add_ids=False, **kwargs)

        self._path = path

        assert self._validation_source is None or self._validation_source == "test", (
            "Constructing validation set from training data does not make sense for"
            " this dataset."
        )
        assert self._data_size_limit is None, (
            "Truncating retrieval datasets is not allowed as it dramatically changes"
            "the meaning of final scores."
        )

        self._dataset = dataset

    @property
    def evaluation_kind(self) -> EvaluationKind:
        return EvaluationKind.RETRIEVAL

    def _retrieve_dataset(self) -> DatasetDict:
        articles = load_dataset(self._path, f"{self._dataset}_articles", split="train")
        sims = load_dataset(self._path, f"{self._dataset}_sims", split="train")

        return DatasetDict({"articles": articles, "sims": sims})

    def _create_splits(self, dataset: DatasetDict) -> DatasetDict:
        def _add_text(article: dict[str, Any]) -> dict[str, Any]:
            sections_text = [
                f"{title} {text}".strip()
                for title, text in zip(
                    article["section_titles"],
                    article["section_texts"],
                    strict=True,
                )
            ]

            text = " ".join(sections_text)
            return {col.TEXT: f"{article['title']} {text}"}

        train = (
            dataset["articles"]
            .map(_add_text)
            .remove_columns(["section_texts", "section_titles", "title"])
        )

        articles_by_id = {}
        for article in train:
            articles_by_id[article[col.ID]] = article

        similarities_by_id = {}
        for article in dataset["sims"]:
            similarities_by_id[article["source_id"]] = article["target_ids"]

        def _add_label(
            article: dict[str, Any], *, source_ids: set[int]
        ) -> dict[str, Any]:
            labels = []

            if article[col.ID] in source_ids:
                target_ids = similarities_by_id[article[col.ID]]
                labels = [copy.deepcopy(articles_by_id[id]) for id in target_ids]

            return {col.LABEL: labels}

        test_source_ids = set(dataset["sims"]["source_id"])
        splits = {"train": train}
        test_sims_total = len(dataset["sims"])
        if self._validation_fraction is not None and self._validation_source == "test":
            val_sims_len = math.floor(test_sims_total * self._validation_fraction)

            val_source_ids = set(random.sample(list(test_source_ids), k=val_sims_len))
            test_source_ids -= val_source_ids

            splits["validation"] = train.map(
                _add_label, fn_kwargs={"source_ids": val_source_ids}
            )

        splits["test"] = train.map(
            _add_label, fn_kwargs={"source_ids": test_source_ids}
        )

        return self._shorten_splits(DatasetDict(splits))
