from __future__ import annotations

from typing import TYPE_CHECKING, Any

from transformer_document_embedding.pipelines.classification_finetune import (
    BinaryClassificationFinetune,
    PairBinaryClassificationFinetune,
)
from transformer_document_embedding.datasets.document_dataset import EvaluationKind
from transformer_document_embedding.pipelines.pipeline import DoNothingTrainPipeline

if TYPE_CHECKING:
    from transformer_document_embedding.pipelines.pipeline import TrainPipeline

_finetune_pipelines = {
    EvaluationKind.BIN_CLAS: BinaryClassificationFinetune,
    EvaluationKind.PAIR_BIN_CLAS: PairBinaryClassificationFinetune,
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
