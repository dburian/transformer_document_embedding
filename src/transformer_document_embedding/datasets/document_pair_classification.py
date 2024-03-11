from __future__ import annotations

import os
import csv
from typing import TYPE_CHECKING

from transformer_document_embedding.datasets.document_dataset import (
    DocumentDataset,
    EvaluationKind,
)


from datasets import DatasetDict, Dataset


if TYPE_CHECKING:
    from typing import Optional, Any, Iterable

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


class DocumentPairClassification(DocumentDataset):
    def __init__(
        self,
        path: str,
        kind: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initializes document-pair classification tasks.

        Parameters
        ----------
        add_ids: bool, default=False
            Each document in a pair has an id. Pass `True` to also generate id per pair.
        """
        kind = kind if kind is not None else os.path.split(path)[-1].lower()

        assert kind in KINDS_TO_FILES, "`kind` must be one of {}".format(
            ",".join(KINDS_TO_FILES.keys())
        )

        super().__init__(
            **kwargs,
            validation_source_fraction=None,
            validation_source=None,
        )

        filenames = KINDS_TO_FILES[kind]
        self._split_paths = {
            split: os.path.join(path, filenames[split])
            for split in ["train", "validation", "test"]
        }

    @property
    def evaluation_kind(self) -> EvaluationKind:
        return EvaluationKind.PAIR_CLAS

    def _retrieve_dataset(self) -> DatasetDict:
        # Record ids of documents across splits
        ids_map = {}
        ds = {}
        for split_name, path in self._split_paths.items():
            ds[split_name] = Dataset.from_generator(
                self._read_csv, gen_kwargs={"path": path, "ids_map": ids_map}
            )

        return DatasetDict(ds)

    @staticmethod
    def _read_csv(path: str, ids_map: dict[int, int]) -> Iterable[dict[str, Any]]:
        def read_doc(text: str) -> tuple[str, int]:
            text = text.replace("\001", " ")
            # Store hash -> int mapping only to save on memory
            id = ids_map.setdefault(hash(text), len(ids_map))

            return text, id

        with open(path, encoding="utf8", mode="r") as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for line in reader:
                text0, id0 = read_doc(line[1])
                text1, id1 = read_doc(line[2])

                yield {
                    "label": int(line[0]),
                    "text_0": text0,
                    "id_0": id0,
                    "text_1": text1,
                    "id_1": id1,
                }
