from __future__ import annotations
from typing import TYPE_CHECKING

from transformer_document_embedding.scripts.config_specs import PipelineSpec
from transformer_document_embedding.pipelines.pipeline import TrainPipeline

if TYPE_CHECKING:
    from typing import Any, Optional
    from transformer_document_embedding.models.embedding_model import EmbeddingModel
    from transformer_document_embedding.datasets.document_dataset import DocumentDataset
    import torch


class ConcatTrainPipeline(TrainPipeline):
    def __init__(self, pipelines: list[dict[str, Any]]) -> None:
        # Deferring import until now, to avoid circular import
        from transformer_document_embedding.pipelines.train_factory import train_factory

        self._pipelines = []
        for spec in pipelines:
            spec = PipelineSpec.from_dict(spec)
            self._pipelines.append(train_factory(spec))

    def __call__(
        self,
        model: EmbeddingModel,
        head: Optional[torch.nn.Module],
        dataset: DocumentDataset,
        log_dir: Optional[str],
    ) -> None:
        for pipeline in self._pipelines:
            pipeline(model, head, dataset, log_dir)
