from __future__ import annotations
from abc import abstractmethod
from enum import Enum
import pprint
from typing import TYPE_CHECKING, Optional, Union
import numpy as np
import math
from datasets import DatasetDict
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from typing import Any


class EvaluationKind(Enum):
    RETRIEVAL = "retrieval"
    BIN_CLAS = "binary_classification"
    PAIR_BIN_CLAS = "pair_binary_classification"
    SENT_EVAL = "sent_eval"
    NONE = "none"


class DocumentDataset:
    """Abstract class for all datasets for training/evaluating document embeddings.

    Wraps splits implemented as HuggingFace `Dataset` with some handy add
    features. Mainly adding ids to documents, and outsourcing validation split
    from other splits."""

    def __init__(
        self,
        *,
        data_size_limit: Optional[Union[int, dict]] = None,
        add_ids: bool = False,
        validation_source_fraction: Optional[float] = None,
        validation_source: Optional[str] = None,
        splits: Optional[dict[str, str]] = None,
    ) -> None:
        self._data_size_limit = data_size_limit

        self._add_ids = add_ids
        self._validation_fraction = validation_source_fraction
        self._validation_source = validation_source
        self._splits_transform = splits
        self._splits = None

    @property
    def splits(self) -> DatasetDict:
        """Returns dictionary of all available splits."""
        if self._splits is None:
            dataset = self._retrieve_dataset()
            self._splits = self._create_splits(dataset)

        return self._splits

    @property
    def evaluation_kind(self) -> EvaluationKind:
        """Returns identifier of the evaluation method of this dataset."""
        raise NotImplementedError()

    @abstractmethod
    def _retrieve_dataset(self) -> DatasetDict:
        """Obtains the dataset. By default using the load_dataset function."""
        raise NotImplementedError()

    def _shorten_splits(self, dataset: DatasetDict) -> DatasetDict:
        if self._data_size_limit is None:
            return dataset

        if not isinstance(self._data_size_limit, dict):
            self._data_size_limit = {
                split_name: self._data_size_limit for split_name in dataset.keys()
            }

        logger.info(
            "Shortening splits of %s to:\n%s",
            self.__class__.__name__,
            pprint.pformat(self._data_size_limit, indent=1),
        )
        for name, split in dataset.items():
            limit = self._data_size_limit.get(name, None)
            if limit is not None and len(split) > limit:
                dataset[name] = split.shuffle().select(range(limit))

        return dataset

    def _transform_splits(self, dataset: DatasetDict) -> DatasetDict:
        if self._splits_transform is None:
            return dataset

        split_config_str = ""
        new_splits = {}
        for new_name, old_name in self._splits_transform.items():
            new_splits[new_name] = dataset[old_name]
            split_config_str += f"{new_name}: (previously '{old_name}')\n"
            split_config_str += pprint.pformat(new_splits[new_name])
            split_config_str += "\n"

        logger.info(
            "Transforming splits of %s:\n%s",
            self.__class__.__name__,
            split_config_str,
        )
        return DatasetDict(new_splits)

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

        dataset = self._shorten_splits(dataset)
        dataset = self._transform_splits(dataset)

        return dataset
