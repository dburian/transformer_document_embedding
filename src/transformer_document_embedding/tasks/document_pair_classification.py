from __future__ import annotations
import torch

import os
import csv
from typing import TYPE_CHECKING

from torcheval.metrics import (
    BinaryAUPRC,
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)
from transformer_document_embedding.tasks.hf_task import HFTask


from datasets import DatasetDict, Dataset

from transformer_document_embedding.utils.evaluation import aggregate_batches

if TYPE_CHECKING:
    from typing import Optional, Union, Any, Iterable
    import numpy as np

KINDS_TO_FILES = {
    "pan": {
        "validation": "pla_dev.csv",
        "train": "pla_train.csv",
        "test": "pla_test.csv",
    },
    "s2orc": {
        "validation": "dev_ai2_gorc.csv",
        "train": "train_ai2_gorc.csv",
        "test": "test_ai2_gorc.csv",
    },
    "oc": {
        "validation": "dev_ai2_ab.csv",
        "train": "train_ai2_ab.csv",
        "test": "test_ai2_ab.csv",
    },
    "aan": {
        "validation": "dev_cite.csv",
        "train": "train_cite.csv",
        "test": "test_cite.csv",
    },
}


class DocumentPairClassification(HFTask):
    def __init__(
        self,
        path: str,
        kind: Optional[str] = None,
        data_size_limit: Optional[Union[int, dict]] = None,
    ) -> None:
        kind = kind if kind is not None else os.path.split(path)[-1].lower()

        assert kind in KINDS_TO_FILES, "`kind` must be one of {}".format(
            ",".join(KINDS_TO_FILES.keys())
        )

        super().__init__(
            data_size_limit=data_size_limit,
            add_ids=True,
            validation_source_fraction=None,
            validation_source=None,
        )

        filenames = KINDS_TO_FILES[kind]
        self._split_paths = {
            split: os.path.join(path, filenames[split])
            for split in ["train", "validation", "test"]
        }

    def _retrieve_dataset(self) -> DatasetDict:
        ds = {}
        for split_name, path in self._split_paths.items():
            ds[split_name] = Dataset.from_generator(
                self._read_csv, gen_kwargs={"path": path}
            )

        return DatasetDict(ds)

    @staticmethod
    def _read_csv(path: str) -> Iterable[dict[str, Any]]:
        with open(path, encoding="utf8", mode="r") as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for line in reader:
                yield {
                    "label": int(line[0]),
                    "text1": line[1].replace("\001", " "),
                    "text2": line[2],
                }

    def evaluate(
        self, split: Dataset, pred_batches: Iterable[np.ndarray]
    ) -> dict[str, float]:
        metrics = {
            "accuracy": BinaryAccuracy(),
            "recall": BinaryRecall(),
            "precision": BinaryPrecision(),
            "f1": BinaryF1Score(),
            "auprc": BinaryAUPRC(),
        }

        for metric in metrics.values():
            metric.reset()

        for pred_batch, true_batch in aggregate_batches(
            pred_batches, iter(split), lambda x: x["label"]
        ):
            for metric in metrics.values():
                metric.update(
                    torch.from_numpy(pred_batch),
                    torch.from_numpy(true_batch),
                )

        return {name: metric.compute().item() for name, metric in metrics.items()}