from __future__ import annotations
from typing import Optional

from datasets import load_dataset
from transformer_document_embedding.datasets.document_dataset import DocumentDataset


from datasets import DatasetDict


class Wikipedia(DocumentDataset):
    def __init__(
        self,
        debug: bool = False,
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

        self._debug = debug

    def _retrieve_dataset(self) -> DatasetDict:
        ds = load_dataset(
            "wikipedia",
            "20220301.simple" if self._debug else "20220301.en",
        )
        assert isinstance(ds, DatasetDict)
        return ds
