from __future__ import annotations

from datasets import load_from_disk


from transformer_document_embedding.datasets.document_dataset import (
    DocumentDataset,
    EvaluationKind,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import DatasetDict
    from typing import Optional


class ExplicitDocumentDataset(DocumentDataset):
    def __init__(
        self,
        evaluation_kind: EvaluationKind,
        path: Optional[str] = None,
        splits: Optional[DatasetDict] = None,
        **kwargs,
    ) -> None:
        if splits is None and path is None:
            raise ValueError("splits or path must be given")

        super().__init__(**kwargs)

        if splits is not None:
            self._splits = splits
        self._path = path

        self._eval_kind = evaluation_kind

    @property
    def splits(self) -> DatasetDict:
        return self._splits

    def _retrieve_dataset(self) -> DatasetDict:
        assert self._path is not None
        ds = load_from_disk(self._path)
        return ds

    @property
    def evaluation_kind(self) -> EvaluationKind:
        return self._eval_kind
