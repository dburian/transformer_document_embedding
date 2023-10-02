from __future__ import annotations
import math
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset

from transformer_document_embedding.tasks.experimental_task import ExperimentalTask
from typing import cast

if TYPE_CHECKING:
    from typing import Any, Optional
    from datasets.arrow_dataset import Dataset


class HFTask(ExperimentalTask):
    def __init__(
        self,
        path,
        *,
        data_size_limit: Optional[int] = None,
        add_ids: bool = False,
        validation_source_fraction: Optional[float] = None,
        validation_source: Optional[str] = None,
    ) -> None:
        self._path = path
        self._data_size_limit = data_size_limit

        self._add_ids = add_ids
        self._validation_fraction = validation_source_fraction
        self._validation_source = validation_source
        self._splits = None

    @property
    def train(self) -> Dataset:
        return self.splits["train"]

    @property
    def test(self) -> Dataset:
        test = self.splits["test"]
        return test.remove_columns("label")

    @property
    def unsupervised(self) -> Optional[Dataset]:
        return self.splits.get("unsupervised", None)

    @property
    def validation(self) -> Optional[Dataset]:
        return self.splits.get("validation", None)

    @property
    def splits(self) -> DatasetDict:
        if self._splits is None:
            dataset = self._retrieve_dataset()
            self._splits = self._create_splits(dataset)

        return self._splits

    def _retrieve_dataset(self) -> DatasetDict:
        """Obtains the dataset. By default using the load_dataset function."""
        return cast(DatasetDict, load_dataset(self._path))

    def _create_splits(self, dataset: DatasetDict) -> DatasetDict:
        """Creates splits."""
        if self._add_ids:
            begin_id = 0

            def map_fn(_, idx: int) -> dict[str, Any]:
                return {"id": idx + begin_id}

            for name, split in dataset.items():
                dataset[name] = split.map(map_fn, with_indices=True)
                begin_id += len(dataset[name])

        if (
            "validation" not in dataset
            and self._validation_source is not None
            and self._validation_fraction is not None
        ):
            validation_source = dataset[self._validation_source]

            source_len = len(validation_source)
            val_len = math.floor(source_len * self._validation_fraction)

            all_indices = np.arange(source_len)
            np.random.shuffle(all_indices)

            val_indices = all_indices[:val_len]
            new_source_indicies = all_indices[val_len:]

            dataset["validation"] = validation_source.select(val_indices)
            dataset[self._validation_source] = validation_source.select(
                new_source_indicies
            )

        if self._data_size_limit is not None:
            for name, split in dataset.items():
                if len(split) > self._data_size_limit:
                    dataset[name] = split.select(range(self._data_size_limit))

        return dataset

    @abstractmethod
    def evaluate(self, pred_batches) -> dict[str, float]:
        raise NotImplementedError()
