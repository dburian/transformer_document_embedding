import copy
import math
import os
import random
from typing import Any, Iterable, Optional

import faiss
import numpy as np
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset

from transformer_document_embedding.tasks.hf_task import HFTask
from transformer_document_embedding.utils.evaluation import (
    evaluate_ir_metrics,
    smart_unbatch,
)

DATASET_DIR = "./data"
AVAILABLE_DATASETS = ["wine", "game"]


class WikipediaSimilarities(HFTask):
    def __init__(self, dataset: str, datasets_dir: str = DATASET_DIR, **kwargs) -> None:
        assert (
            dataset in AVAILABLE_DATASETS
        ), f"`dataset` must be one of: {', '.join(AVAILABLE_DATASETS)}"

        # Dataset already contains ids
        kwargs["add_ids"] = False
        super().__init__(
            os.path.join(datasets_dir, "wikipedia_similarities.py"), **kwargs
        )

        assert self._validation_source is None or self._validation_source == "test", (
            "Constructing validation set from training data does not make sense for"
            " this dataset."
        )

        self._dataset = dataset
        self._test_sims_total = None

    def _retrieve_dataset(self) -> DatasetDict:
        articles = load_dataset(self._path, f"{self._dataset}_articles", split="train")
        sims = load_dataset(self._path, f"{self._dataset}_sims", split="train")

        return DatasetDict({"articles": articles, "sims": sims})

    def _create_splits(self, dataset: DatasetDict) -> DatasetDict:
        def _add_text(article: dict[str, Any]) -> dict[str, Any]:
            sections_text = [
                f"{title} {text}".strip()
                for title, text in zip(
                    article["section_titles"], article["section_texts"]
                )
            ]

            text = " ".join(sections_text)
            return {"text": f"{article['title']} {text}"}

        train = (
            dataset["articles"]
            .map(_add_text)
            .remove_columns(["section_texts", "section_titles", "title"])
        )

        articles_by_id = {}
        for article in train:
            articles_by_id[article["id"]] = article

        similarities_by_id = {}
        for article in dataset["sims"]:
            similarities_by_id[article["source_id"]] = article["target_ids"]

        def _add_label(
            article: dict[str, Any], *, source_ids: set[int]
        ) -> dict[str, Any]:
            labels = []

            if article["id"] in source_ids:
                target_ids = similarities_by_id[article["id"]]
                labels = [copy.deepcopy(articles_by_id[id]) for id in target_ids]

            return {"label": labels}

        test_source_ids = set(dataset["sims"]["source_id"])
        splits = {"train": train}
        self._test_sims_total = len(dataset["sims"])
        if self._validation_fraction is not None and self._validation_source == "test":
            self._test_sims_total = len(test_source_ids)
            val_sims_len = math.floor(self._test_sims_total * self._validation_fraction)

            val_source_ids = set(random.sample(list(test_source_ids), k=val_sims_len))
            test_source_ids -= val_source_ids

            splits["validation"] = train.map(
                _add_label, fn_kwargs={"source_ids": val_source_ids}
            )

        splits["test"] = train.map(
            _add_label, fn_kwargs={"source_ids": test_source_ids}
        )

        if self._data_size_limit is not None:
            for name, split in splits.items():
                if len(split) > self._data_size_limit:
                    splits[name] = split.select(range(self._data_size_limit))

        return DatasetDict(splits)

    def evaluate(self, pred_batches: Iterable[np.ndarray]) -> dict[str, float]:
        preds = smart_unbatch(pred_batches, 1)

        true_pred_ids_iter = get_nearest_ids_from_faiss(
            self.splits["test"],
            preds,
            k=1000,
        )

        print(self._test_sims_total)
        return evaluate_ir_metrics(
            true_pred_ids_iter,
            hits_thresholds=[10, 100],
            iterable_length=self._test_sims_total,
            verbose=True,
        )


def get_nearest_ids_from_faiss(
    true_dataset: Dataset,
    embeddings: Iterable[np.ndarray],
    *,
    k: Optional[int] = None,
) -> Iterable[tuple[list[int], list[int]]]:
    embed_column_name = "embedding"
    faiss_dataset = true_dataset.add_column(
        name=embed_column_name,
        column=map(lambda vec: vec / np.linalg.norm(vec), embeddings),
    )
    faiss_dataset.add_faiss_index(
        embed_column_name, metric_type=faiss.METRIC_INNER_PRODUCT
    )

    if k is None:
        k = len(faiss_dataset)

    for article in faiss_dataset:
        if len(article["label"]) == 0:
            continue

        nearest_targets = faiss_dataset.get_nearest_examples(
            embed_column_name,
            np.array(article[embed_column_name]),
            k=k + 1,  # We're later removing the first hit, which is the query itself.
        )

        true_ids = [target_article["id"] for target_article in article["label"]]
        pred_ids = nearest_targets.examples["id"][1:]

        yield true_ids, pred_ids
