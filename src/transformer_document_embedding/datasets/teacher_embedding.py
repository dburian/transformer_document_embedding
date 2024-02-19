from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import Dataset, DatasetDict, load_from_disk
from transformer_document_embedding.datasets import col
from transformer_document_embedding.datasets.document_dataset import (
    DocumentDataset,
    EvaluationKind,
)

if TYPE_CHECKING:
    from typing import Optional


class TeacherEmbedding(DocumentDataset):
    """Task to train embeddings using already generated teacher models.

    See `scripts/generate_embeddings.py` to create such dataset.
    """

    EVALUATION_KIND = EvaluationKind.NONE

    def __init__(
        self,
        path: str,
        contextual_embedding_col: Optional[str] = None,
        structural_embedding_col: Optional[str] = None,
        data_size_limit: Optional[int] = None,
        validation_source_fraction: Optional[float] = None,
        validation_source: Optional[str] = None,
    ) -> None:
        super().__init__(
            data_size_limit=data_size_limit,
            add_ids=False,
            validation_source_fraction=validation_source_fraction,
            validation_source=validation_source,
        )
        self._path_to_dataset = path

        self.contextual_embedding_col = contextual_embedding_col
        self.structural_embedding_col = structural_embedding_col

    @property
    def test(self) -> Dataset:
        return Dataset.from_dict({})

    def _retrieve_dataset(self) -> DatasetDict:
        dataset = load_from_disk(self._path_to_dataset)
        assert isinstance(dataset, DatasetDict)

        renames = {}
        if self.contextual_embedding_col is not None:
            renames[self.contextual_embedding_col] = col.CONTEXTUAL_EMBED

        if self.structural_embedding_col is not None:
            renames[self.structural_embedding_col] = col.STRUCTURAL_EMBED

        return dataset if len(renames) == 0 else dataset.rename_columns(renames)
