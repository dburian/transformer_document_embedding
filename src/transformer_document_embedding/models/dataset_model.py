from __future__ import annotations
from typing import Iterator, TYPE_CHECKING
from transformer_document_embedding.datasets import col


from transformer_document_embedding.models.embedding_model import EmbeddingModel

if TYPE_CHECKING:
    from datasets import Dataset
    import torch


class DatasetModel(EmbeddingModel):
    """Model that simply reads pre-generated embeddings present in the dataset."""

    def __init__(self, **kwargs) -> None:
        super().__init__()

        for name, value in kwargs.items():
            setattr(self, name, value)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @embedding_dim.setter
    def embedding_dim(self, dim: int) -> None:
        self._embedding_dim = dim

    def predict_embeddings(self, dataset: Dataset, **_) -> Iterator[torch.Tensor]:
        batch_size = getattr(self, "batch_size", None)
        if batch_size is None:
            batch_size = 16

        for batch in dataset.with_format("torch", columns=[col.EMBEDDING]).iter(
            batch_size
        ):
            yield batch[col.EMBEDDING]

    def load_weights(self, path: str, *, strict: bool = False) -> None:
        pass

    def save_weights(self, path: str) -> None:
        pass
