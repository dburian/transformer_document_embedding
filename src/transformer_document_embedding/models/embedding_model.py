from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset
    from typing import Iterator
    import torch


class EmbeddingModel:
    """Defines the minimal interface for embedding models."""

    @property
    def embedding_dim(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def predict_embeddings(self, dataset: Dataset) -> Iterator[torch.Tensor]:
        """Returns batched embeddings of a dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset with `col.TEXT` and optionally `col.ID` fields.

        Returns
        -------
        Iterator[torch.Tensor]
            Batched embeddings of non-shuffled `dataset`.
        """
        pass

    @abstractmethod
    def save_weights(self, path: str) -> None:
        """Saves the model's weights into given path."""
        raise NotImplementedError()

    @abstractmethod
    def load_weights(self, path: str, *, strict: bool = False) -> None:
        """Loads the model's weights from given path.

        Parameters
        ----------
        path: str
            Path where to save the model's weights.
        strict: bool, default = True
            If true, loading will fail if loaded parameters will not correspond
            to what is expected.
        """
        raise NotImplementedError()
