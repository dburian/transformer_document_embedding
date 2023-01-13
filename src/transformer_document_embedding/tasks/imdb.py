from typing import Iterable

import numpy as np
import tensorflow as tf
from datasets.arrow_dataset import Dataset
from datasets.combine import concatenate_datasets
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from datasets.load import load_dataset

from transformer_document_embedding.tasks.experimental_task import \
    ExperimentalTask

IMDBData = Dataset | IterableDataset | DatasetDict | IterableDatasetDict


class IMDBClassification(ExperimentalTask):
    """Classification task done using the IMDB dataset.

    The dataset is specified as `datasets.Dataset` with 'train', 'test' and
    'unsupervised' splits.
    """

    def __init__(self, *, data_size_limit: int = -1) -> None:
        self._train = None
        self._test = None
        self._unsuper = None
        self._all_train = None
        self._test_inputs = None
        self._data_size_limit = data_size_limit

    @property
    def train(self) -> IMDBData:
        """
        Returns datasets.Dataset of both train and unsupervised training
        documents. Each document is dictionary with keys:
            - 'text' (str) - text of the document,
            - 'label' (int) - 1/0 sentiment class index,
            - 'id' (int) - document id unique among all the documents in the dataset.
        """
        if self._all_train is None:
            self._train = load_dataset(
                "imdb", split=f"train[:{self._data_size_limit}]"
            ).map(
                lambda _, idx: {"id": idx},
                with_indices=True,
            )
            self._unsuper = load_dataset(
                "imdb", split=f"unsupervised[:{self._data_size_limit}]"
            ).map(
                lambda _, idx: {"id": idx + len(self._train)},
                with_indices=True,
            )

            self._all_train = concatenate_datasets([self._train, self._unsuper])

        return self._all_train

    @property
    def test(self) -> IMDBData:
        """
        Returns datasets.Dataset of testing documents. Each document is
        dictionary with keys:
            - 'text' (str) - text of the document,
            - 'id' (int) - document id unique among all the documents in the dataset.
        """
        if self._test_inputs is None:
            if self._train is None or self._unsuper is None:
                self.train

            id_offset = len(self._train) + len(self._unsuper)
            self._test = load_dataset(
                "imdb", split=f"test[:{self._data_size_limit}]"
            ).map(
                lambda _, idx: {"id": idx + id_offset},
                with_indices=True,
            )
            self._test_inputs = self._test.remove_columns("label")

        return self._test_inputs

    def evaluate(
        self, test_predictions: Iterable[np.ndarray], batch_size: int = 100
    ) -> dict[str, float]:
        metrics = [
            tf.keras.metrics.BinaryCrossentropy(),
            tf.keras.metrics.BinaryAccuracy(),
        ]

        for met in metrics:
            met.reset_state()

        def update_metrics(y_true, y_pred) -> None:
            for met in metrics:
                met.update_state(y_true, y_pred)

        batch_true, batch_pred = [], []
        for test_doc, y_pred in zip(self._test, test_predictions):
            batch_true.append(test_doc["label"])
            batch_pred.append(y_pred)

            if len(batch_true) == batch_size:
                update_metrics(batch_true, batch_pred)
                batch_true, batch_pred = [], []

        if len(batch_true) > 0:
            update_metrics(batch_true, batch_pred)

        return {met.name: met.result().numpy() for met in metrics}


Task = IMDBClassification
