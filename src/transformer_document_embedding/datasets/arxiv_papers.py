from __future__ import annotations
from typing import TYPE_CHECKING

from datasets import DatasetDict, load_dataset, load_from_disk

from transformer_document_embedding.datasets.document_dataset import (
    DocumentDataset,
    EvaluationKind,
)


if TYPE_CHECKING:
    from typing import Optional, Union


class ArxivPapers(DocumentDataset):
    def __init__(
        self,
        path: Optional[str] = None,
        data_size_limit: Optional[Union[int, dict]] = None,
        add_ids: bool = False,
        validation_source_fraction: Optional[float] = None,
        validation_source: Optional[str] = None,
        splits: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(
            data_size_limit=data_size_limit,
            add_ids=add_ids,
            validation_source_fraction=validation_source_fraction,
            validation_source=validation_source,
            splits=splits,
        )

        self._load_path = path

    @property
    def evaluation_kind(self) -> EvaluationKind:
        return EvaluationKind.CLAS

    def _retrieve_dataset(self) -> DatasetDict:
        ds = (
            load_from_disk(self._load_path)
            if self._load_path is not None
            else load_dataset("ccdv/arxiv-classification", "no_ref")
        )
        assert isinstance(ds, DatasetDict)

        return ds
