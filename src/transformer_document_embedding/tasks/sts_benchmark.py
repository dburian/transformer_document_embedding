from __future__ import annotations
import csv
import logging

from scipy.stats import spearmanr
import os
from typing import TYPE_CHECKING

from datasets import Dataset, DatasetDict
from transformer_document_embedding.tasks.hf_task import HFTask
import numpy as np
import pandas as pd

from transformer_document_embedding.utils.evaluation import smart_unbatch

if TYPE_CHECKING:
    from typing import Optional, Iterable, Any


logger = logging.getLogger(__name__)

DEFAULT_SPLIT_FILENAMES = {
    "train": "sts-train.csv",
    "validation": "sts-dev.csv",
    "test": "sts-test.csv",
}
COL_NAMES = [
    "genre",
    "file",
    "year",
    "not_unique_id",
    "score",  # Similarity float score <0,5>
    "sent1",
    "sent2",
    "opt_field1",
    "opt_field2",
]


class STSBenchmark(HFTask):
    def __init__(
        self,
        dir_path: str,
        split_filenames: dict[str, str] = DEFAULT_SPLIT_FILENAMES,
        data_size_limit: Optional[int] = None,
        validation_source_fraction: Optional[float] = None,
        validation_source: Optional[str] = None,
    ) -> None:
        super().__init__(
            data_size_limit=data_size_limit,
            add_ids=False,
            validation_source_fraction=validation_source_fraction,
            validation_source=validation_source,
        )
        self._dir_path = dir_path
        self._split_filenames = split_filenames

    def _retrieve_dataset(self) -> DatasetDict:
        def generate_texts(df: pd.DataFrame) -> Iterable[dict[str, Any]]:
            global_id = 0
            for idx, row in df.iterrows():
                for sent in ["sent1", "sent2"]:
                    yield {
                        "text": row[sent],
                        "pair_id": idx,
                        "score": row["score"],
                        "id": global_id,
                    }
                    global_id += 1

        splits = {}
        for split, filename in self._split_filenames.items():
            # TODO: Not really necessary to use pandas
            df = pd.read_csv(
                os.path.join(self._dir_path, filename),
                names=COL_NAMES,
                sep="\t",
                quoting=csv.QUOTE_NONE,
            )[["score", "sent1", "sent2"]]
            assert isinstance(df, pd.DataFrame)

            splits[split] = Dataset.from_generator(
                generate_texts, gen_kwargs={"df": df}
            )

        return DatasetDict(splits)

    def evaluate(
        self, split: Dataset, pred_batches: Iterable[np.ndarray]
    ) -> dict[str, float]:
        trues = []
        preds = []

        def sim(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))

        embedding_cache = {}
        for embedding, doc in zip(smart_unbatch(pred_batches, 1), split, strict=True):
            pair_id = doc["pair_id"]
            other_embedding = embedding_cache.get(pair_id, None)

            if other_embedding is None:
                embedding_cache[pair_id] = embedding
                continue

            trues.append(doc["score"] / 5)
            preds.append(sim(other_embedding, embedding))
            del embedding_cache[pair_id]

        if len(embedding_cache.keys()) > 0:
            logger.warn(
                "Not all sentences from given split were evaluated."
                f"{len(embedding_cache.keys())} sentences didn't have paired sentences."
            )

        correlation_results = spearmanr(np.array(trues), np.array(preds))

        return {
            "spearman_rank_correlation_statistic": correlation_results.statistic,
            "spearman_rank_correlation_p_value": correlation_results.pvalue,
        }
