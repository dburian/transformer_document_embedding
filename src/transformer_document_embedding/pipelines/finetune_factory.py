from __future__ import annotations

from typing import TYPE_CHECKING, Any

from transformer_document_embedding.pipelines.classification_finetune import (
    ClassificationFinetune,
    PairClassificationFinetune,
)
from transformer_document_embedding.datasets.document_dataset import EvaluationKind
from transformer_document_embedding.pipelines.pipeline import DoNothingTrainPipeline

if TYPE_CHECKING:
    from transformer_document_embedding.pipelines.pipeline import TrainPipeline

_finetune_pipelines = {
    EvaluationKind.CLAS: ClassificationFinetune,
    EvaluationKind.PAIR_CLAS: PairClassificationFinetune,
    EvaluationKind.RETRIEVAL: DoNothingTrainPipeline,
    # For models that require training we could define different finetuning
    # pipelines that would add a 'prepare()' method to the model, which would
    # be then used in the evaluation pipeline.
    EvaluationKind.SENT_EVAL: DoNothingTrainPipeline,
}


def finetune_factory(
    evaluation_kind: EvaluationKind, pipeline_kwargs: dict[str, Any]
) -> TrainPipeline:
    return _finetune_pipelines[evaluation_kind](**pipeline_kwargs)
