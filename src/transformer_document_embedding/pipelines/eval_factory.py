from __future__ import annotations
from typing import TYPE_CHECKING


from transformer_document_embedding.pipelines.classification_eval import (
    ClassificationEval,
    PairClassificationEval,
)
from transformer_document_embedding.datasets.document_dataset import (
    EvaluationKind,
)
import logging

from transformer_document_embedding.pipelines.retrieval_eval import RetrievalEval
from transformer_document_embedding.pipelines.sent_eval_eval import SentEvalEval

if TYPE_CHECKING:
    from transformer_document_embedding.pipelines.pipeline import EvalPipeline


logger = logging.getLogger(__name__)


_eval_pipelines = {
    EvaluationKind.CLAS: ClassificationEval,
    EvaluationKind.PAIR_CLAS: PairClassificationEval,
    EvaluationKind.RETRIEVAL: RetrievalEval,
    EvaluationKind.SENT_EVAL: SentEvalEval,
}


def eval_factory(ds_kind: EvaluationKind) -> EvalPipeline:
    return _eval_pipelines[ds_kind]()
