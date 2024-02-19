from __future__ import annotations
from typing import TYPE_CHECKING

from datasets import DatasetDict, load_dataset
from transformer_document_embedding.datasets.document_dataset import DocumentDataset

if TYPE_CHECKING:
    from typing import Optional


class BookCorpus(DocumentDataset):
    def __init__(
        self,
        *,
        data_size_limit: Optional[int] = None,
        validation_source_fraction: Optional[float] = None,
        validation_source: Optional[str] = None,
    ) -> None:
        super().__init__(
            data_size_limit=data_size_limit,
            add_ids=True,
            validation_source_fraction=validation_source_fraction,
            validation_source=validation_source,
        )

    def _retrieve_dataset(self) -> DatasetDict:
        ds = load_dataset("bookcorpus")
        assert isinstance(ds, DatasetDict)
        return ds
