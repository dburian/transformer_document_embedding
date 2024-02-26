from __future__ import annotations


from transformer_document_embedding.datasets.document_dataset import (
    DocumentDataset,
    EvaluationKind,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import DatasetDict


class ExplicitDocumentDataset(DocumentDataset):
    def __init__(
        self,
        splits: DatasetDict,
        evaluation_kind: EvaluationKind,
    ) -> None:
        super().__init__(
            data_size_limit=None,
            add_ids=False,
            validation_source_fraction=None,
            validation_source=None,
        )

        self._splits = splits
        self._eval_kind = evaluation_kind

    @property
    def splits(self) -> DatasetDict:
        return self._splits

    @property
    def evaluation_kind(self) -> EvaluationKind:
        return self._eval_kind
