from __future__ import annotations
from typing import TYPE_CHECKING, Type
from transformer_document_embedding.pipelines.gensim.cls_head_finetune import (
    PVClassificationHeadTrain,
    PVPairClassificationHeadTrain,
)

from transformer_document_embedding.pipelines.gensim.train_pv import (
    TrainPVPipeline,
    TrainPairPVPipeline,
)
from transformer_document_embedding.pipelines.torch.classification import (
    TorchClassifiactionPipeline,
    TorchTrainPairClassificationPipeline,
)
from transformer_document_embedding.pipelines.torch.student import StudentTrainPipeline
from transformer_document_embedding.pipelines.train_meta import ConcatTrainPipeline


if TYPE_CHECKING:
    from .pipeline import TrainPipeline
    from transformer_document_embedding.scripts.config_specs import PipelineSpec


pipelines_registered: dict[str, Type[TrainPipeline]] = {
    "concat": ConcatTrainPipeline,
    "pv": TrainPVPipeline,
    "pv_pair": TrainPairPVPipeline,
    "pv_cls_head": PVClassificationHeadTrain,
    "pv_pair_cls_head": PVPairClassificationHeadTrain,
    "torch_cls": TorchClassifiactionPipeline,
    "torch_pair_cls": TorchTrainPairClassificationPipeline,
    "student": StudentTrainPipeline,
}


def train_factory(pipeline_spec: PipelineSpec) -> TrainPipeline:
    if pipeline_spec.kind in pipelines_registered:
        return pipelines_registered[pipeline_spec.kind](**pipeline_spec.kwargs)

    raise NotImplementedError()
