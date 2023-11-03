from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import Dataset, DatasetDict, load_from_disk
from transformer_document_embedding.tasks.hf_task import HFTask

if TYPE_CHECKING:
    from typing import Optional, Iterable
    import numpy as np


class TeacherEmbedding(HFTask):
    """Task to train embeddings using already generated teacher models.

    See `scripts/generate_embeddings.py` to create such dataset.
    """

    def __init__(
        self,
        path: str,
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

    def test(self) -> Dataset:
        return Dataset.from_dict({})

    def _retrieve_dataset(self) -> DatasetDict:
        dataset = load_from_disk(self._path_to_dataset)
        assert isinstance(dataset, DatasetDict)

        return dataset

    def evaluate(
        self,
        split: Dataset,
        pred_batches: Iterable[np.ndarray],
    ) -> dict[str, float]:
        return {"ok": 1.0}
