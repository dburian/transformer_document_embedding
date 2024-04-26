from __future__ import annotations

from datasets import load_from_disk


from transformer_document_embedding.datasets.document_dataset import (
    DocumentDataset,
    EvaluationKind,
)
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from datasets import DatasetDict
    from typing import Optional


class ExplicitDocumentDataset(DocumentDataset):
    def __init__(
        self,
        evaluation_kind: Union[EvaluationKind, str],
        path: Optional[str] = None,
        splits: Optional[DatasetDict] = None,
        **kwargs,
    ) -> None:
        if splits is None and path is None:
            raise ValueError("splits or path must be given")

        super().__init__(**kwargs)

        self._splits = splits
        self._path = path

        self._eval_kind = (
            EvaluationKind(evaluation_kind)
            if isinstance(evaluation_kind, str)
            else evaluation_kind
        )

    def _retrieve_dataset(self) -> DatasetDict:
        assert self._path is not None
        ds = load_from_disk(self._path)
        return ds

    @property
    def evaluation_kind(self) -> EvaluationKind:
        return self._eval_kind
