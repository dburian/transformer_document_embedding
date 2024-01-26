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

    BREADTH_COL = "breadth_embedding"
    DEPTH_COL = "depth_embedding"

    def __init__(
        self,
        path: str,
        breadth_embedding_col: Optional[str] = None,
        depth_embedding_col: Optional[str] = None,
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

        self.breadth_embedding_col = breadth_embedding_col
        self.depth_embedding_col = depth_embedding_col

    @property
    def test(self) -> Dataset:
        return Dataset.from_dict({})

    def _retrieve_dataset(self) -> DatasetDict:
        dataset = load_from_disk(self._path_to_dataset)
        assert isinstance(dataset, DatasetDict)

        renames = {}
        if self.breadth_embedding_col is not None:
            renames[self.breadth_embedding_col] = self.BREADTH_COL

        if self.depth_embedding_col is not None:
            renames[self.depth_embedding_col] = self.DEPTH_COL

        return dataset if len(renames) == 0 else dataset.rename_columns(renames)

    def evaluate(
        self,
        split: Dataset,
        pred_batches: Iterable[np.ndarray],
    ) -> dict[str, float]:
        return {"ok": 1.0}
