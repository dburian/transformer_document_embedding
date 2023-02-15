from typing import Iterable, Optional

import numpy as np
import tensorflow as tf
from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset

from transformer_document_embedding.tasks.experimental_task import \
    ExperimentalTask

IMDBData = Dataset


class IMDBClassification(ExperimentalTask):
    """Classification task done using the IMDB dataset.

    The dataset is specified as `datasets.Dataset` with 'train', 'test' and
    'unsupervised' splits.
    """

    def __init__(self, *, data_size_limit: int = -1) -> None:
        self._train: Optional[IMDBData] = None
        self._test: Optional[IMDBData] = None
        self._unsupervised: Optional[IMDBData] = None
        self._test_inputs: Optional[IMDBData] = None
        self._data_size_limit = data_size_limit

    @property
    def train(self) -> IMDBData:
        """
        Returns datasets.Dataset of both train and unsupervised training
        documents. Each document is dictionary with keys:
            - 'text' (str) - text of the document,
            - 'label' (int) - 1/0 sentiment class index,
        """
        if self._train is None:
            downloaded = load_dataset(
                "imdb", split=f"train[:{self._data_size_limit}]"
            ).map(
                lambda _, idx: {"id": idx},
                with_indices=True,
            )
            assert isinstance(
                downloaded, IMDBData
            ), f"{IMDBData.__name__}.train: cannot download train split."
            self._train = downloaded

        return self._train

    @property
    def unsupervised(self) -> IMDBData:
        if self._unsupervised is None:
            downloaded = load_dataset(
                "imdb", split=f"unsupervised[:{self._data_size_limit}]"
            ).map(
                lambda _, idx: {"id": idx + len(self.train)},
                with_indices=True,
            )
            assert isinstance(
                downloaded, IMDBData
            ), f"{IMDBData.__name__}.unsupervised: cannot download unsupervised split."
            self._unsupervised = downloaded

        return self._unsupervised

    @property
    def test(self) -> IMDBData:
        """
        Returns datasets.Dataset of testing documents. Each document is
        dictionary with keys:
            - 'text' (str) - text of the document,
        """
        if self._test_inputs is None:
            id_offset = len(self.train) + len(self.unsupervised)
            downloaded = load_dataset(
                "imdb", split=f"test[:{self._data_size_limit}]"
            ).map(
                lambda _, idx: {"id": idx + id_offset},
                with_indices=True,
            )
            assert isinstance(
                downloaded, IMDBData
            ), f"{IMDBData.__name__}.test: cannot download test split"
            self._test = downloaded

            self._test_inputs = self._test.remove_columns("label")

        return self._test_inputs

    def evaluate(self, pred_batches: Iterable[np.ndarray]) -> dict[str, float]:
        assert (
            self._test is not None
        ), f"{self.__class__}.evaluate() called before generating test data."

        metrics = [
            tf.keras.metrics.BinaryCrossentropy(),
            tf.keras.metrics.BinaryAccuracy(),
        ]

        for met in metrics:
            met.reset_state()

        def update_metrics(y_true, y_pred) -> None:
            for met in metrics:
                met.update_state(y_true, y_pred)

        true_iter = iter(self._test)
        for pred_batch in pred_batches:
            true_batch = []
            while len(true_batch) < len(pred_batch):
                true_batch.append(next(true_iter)["label"])

            update_metrics(true_batch, pred_batch)

        return {met.name: met.result().numpy() for met in metrics}


Task = IMDBClassification
