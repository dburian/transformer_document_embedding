import math
from abc import abstractmethod
from typing import Any, Optional, cast

import numpy as np
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset

from transformer_document_embedding.tasks.experimental_task import \
    ExperimentalTask


class HFTask(ExperimentalTask):
    def __init__(
        self,
        path,
        *,
        data_size_limit: Optional[int] = None,
        add_ids: bool = False,
        validation_train_fraction: Optional[float] = None,
        test_as_validation: bool = False,
    ) -> None:
        assert validation_train_fraction is None or not test_as_validation, (
            f"{HFTask.__init__.__name__} validation split can be constructed from test"
            " or train split, but not both."
        )
        self._path = path
        self._data_size_limit = data_size_limit

        self._add_ids = add_ids
        self._validation_train_fraction = validation_train_fraction
        self._test_as_validation = test_as_validation
        self._dataset = None

    @property
    def train(self) -> Dataset:
        train = self._get_split("train")
        assert train is not None
        return train

    @property
    def test(self) -> Dataset:
        test = self._get_split("test")
        assert test is not None
        return test.remove_columns("label")

    @property
    def unsupervised(self) -> Optional[Dataset]:
        return self._get_split("unsupervised")

    @property
    def validation(self) -> Optional[Dataset]:
        return self._get_split("validation")

    def _get_split(self, split: str) -> Optional[Dataset]:
        if split not in self.dataset:
            return None

        return self.dataset[split]

    @property
    def dataset(self) -> DatasetDict:
        if self._dataset is None:
            self._dataset = self._construct_splits()

        return self._dataset

    def _retrieve_dataset(self) -> DatasetDict:
        """Obtains the dataset. By default using the load_dataset function."""
        return cast(DatasetDict, load_dataset(self._path))

    def _construct_splits(self) -> DatasetDict:
        """Creates splits."""
        dataset = self._retrieve_dataset()

        if self._add_ids:
            begin_id = 0

            def map_fn(_, idx: int) -> dict[str, Any]:
                return {"id": idx + begin_id}

            for name, split in dataset.items():
                dataset[name] = split.map(map_fn, with_indices=True)
                begin_id += len(dataset[name])

        if "validation" not in dataset:
            if self._test_as_validation:
                dataset["validation"] = dataset["test"]
            elif self._validation_train_fraction is not None:
                train_len = len(dataset["train"])
                val_len = math.floor(train_len * self._validation_train_fraction)

                all_indices = np.arange(train_len)
                np.random.shuffle(all_indices)

                val_indices = all_indices[:val_len]
                train_indices = all_indices[val_len:]

                dataset["validation"] = dataset["train"].select(val_indices)
                dataset["train"] = dataset["train"].select(train_indices)

        if self._data_size_limit is not None:
            for name, split in dataset.items():
                if len(split) > self._data_size_limit:
                    dataset[name] = split.select(range(self._data_size_limit))

        return dataset

    @abstractmethod
    def evaluate(self, pred_batches) -> dict[str, float]:
        raise NotImplementedError()
