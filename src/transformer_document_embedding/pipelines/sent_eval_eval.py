from __future__ import annotations
from functools import partial
from typing import Any, Optional, TYPE_CHECKING
from datasets import Dataset

import senteval
from transformer_document_embedding.datasets import col

from transformer_document_embedding.pipelines.pipeline import EvalPipeline
import numpy as np

if TYPE_CHECKING:
    from senteval.utils import dotdict
    from transformer_document_embedding.datasets.sent_eval import SentEval
    import torch
    from transformer_document_embedding.models.embedding_model import EmbeddingModel


class SentEvalEval(EvalPipeline):
    """Evaluation of Sent-Eval.

    Implemented in evaluation mode only -- no training is done. This prohibits
    to evaluate sent-eval on models that require training (such as TF-IDF).
    """

    def _reduce_results(self, all_results: dict[str, Any]) -> dict[str, float]:
        """Leaves only some metrics."""
        reduced_results = {}
        for task, task_results in all_results.items():
            if task.startswith("STS") and task != "STSBenchmark":
                reduced_results[f"{task}_spearman"] = task_results["all"]["spearman"][
                    "wmean"
                ]
                reduced_results[f"{task}_pearson"] = task_results["all"]["pearson"][
                    "wmean"
                ]
            elif task in ["STSBenchmark", "SICKRelatedness", "SICKEntailment"]:
                reduced_results[f"{task}_spearman"] = task_results["spearman"]
                reduced_results[f"{task}_pearson"] = task_results["pearson"]
            else:
                # All classification tasks
                reduced_results[f"{task}_accuracy"] = task_results["acc"]

        for metric_name, score in reduced_results.items():
            reduced_results[metric_name] = float(score)
        return reduced_results

    def _words_to_dataset(self, word_batch: list[list[str]]) -> Dataset:
        sentences = [" ".join(words) if len(words) > 0 else "." for words in word_batch]
        return Dataset.from_dict({col.TEXT: sentences})

    def _batcher(
        self, params: dotdict, batch: list[list[str]], model: EmbeddingModel
    ) -> np.ndarray:
        ds = self._words_to_dataset(batch)
        pred_batches = [
            batch.numpy(force=True) for batch in model.predict_embeddings(ds)
        ]
        return np.vstack(pred_batches)

    def __call__(
        self,
        model: EmbeddingModel,
        _: Optional[torch.nn.Module],
        dataset: SentEval,
    ) -> dict[str, float]:
        batcher = partial(self._batcher, model=model)
        se = senteval.SE(dataset.params, batcher)
        results = se.eval(dataset.tasks)

        return self._reduce_results(results)
