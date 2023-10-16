from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import DatasetDict, load_dataset
from transformer_document_embedding.tasks.hf_task import HFTask

if TYPE_CHECKING:
    from typing import Optional


class C4(HFTask):
    DEFAULT_TRAIN_FILES = "en/c4-train.000[012]*-of-01024.json.gz"  # ~30% of train
    DEFAULT_VALIDATION_FILES = "en/c4-validation.*.json.gz"  # all validation

    def __init__(
        self,
        path: str = "allenai/c4",
        config: str = "en",
        data_files: Optional[dict[str, str]] = None,
        *,
        data_size_limit: Optional[int] = None,
    ) -> None:
        super().__init__(
            data_size_limit=data_size_limit,
            add_ids=True,
        )
        self._path = path
        self._config = config

        if data_files is None:
            data_files = {
                "train": self.DEFAULT_TRAIN_FILES,
                "validation": self.DEFAULT_VALIDATION_FILES,
            }
        self._data_files = data_files

    @property
    def test(self) -> None:
        return None

    def _retrieve_dataset(self) -> DatasetDict:
        dataset_dict = load_dataset(
            self._path,
            self._config,
            data_files=self._data_files,
        )
        assert isinstance(dataset_dict, DatasetDict)
        return dataset_dict

    def evaluate(self, _) -> dict[str, float]:
        raise NotImplementedError()
