from __future__ import annotations

from datasets import load_dataset
from transformer_document_embedding.datasets.document_dataset import (
    EvaluationKind,
    DocumentDataset,
)
from datasets import DatasetDict


class IMDB(DocumentDataset):
    """Binary classification dataset of IMDB reviews."""

    def __init__(self, **kwargs) -> None:
        super().__init__(add_ids=True, **kwargs)
        self._path = "imdb"

    @property
    def evaluation_kind(self) -> EvaluationKind:
        return EvaluationKind.CLAS

    def _retrieve_dataset(self) -> DatasetDict:
        dataset_dict = load_dataset(self._path)
        assert isinstance(dataset_dict, DatasetDict)
        return dataset_dict
