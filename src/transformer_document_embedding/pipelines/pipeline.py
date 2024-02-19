from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Any, Optional
    from transformer_document_embedding.models.embedding_model import EmbeddingModel
    from transformer_document_embedding.datasets.document_dataset import DocumentDataset
    import torch


class Pipeline:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()


class TrainPipeline(Pipeline):
    @abstractmethod
    def __call__(
        self,
        model: EmbeddingModel,
        head: Optional[torch.nn.Module],
        dataset: DocumentDataset,
        log_dir: Optional[str],
    ) -> None:
        raise NotImplementedError()


class EvalPipeline(Pipeline):
    @abstractmethod
    def __call__(
        self,
        model: EmbeddingModel,
        head: Optional[torch.nn.Module],
        dataset: DocumentDataset,
    ) -> dict[str, float]:
        raise NotImplementedError()


class DoNothingTrainPipeline(TrainPipeline):
    def __call__(
        self,
        model: EmbeddingModel,
        head: Optional[torch.nn.Module],
        dataset: DocumentDataset,
        log_dir: Optional[str],
    ) -> None:
        pass
