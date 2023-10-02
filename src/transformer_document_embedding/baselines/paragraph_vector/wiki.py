from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any, Iterable, Optional

from gensim.models.callbacks import CallbackAny2Vec

import transformer_document_embedding.tasks.wikipedia_similarities as wiki_sims_task
from transformer_document_embedding.baselines.experimental_model import (
    ExperimentalModel,
)
from transformer_document_embedding.models.paragraph_vector import ParagraphVector
from transformer_document_embedding.utils.evaluation import evaluate_ir_metrics
from transformer_document_embedding.utils.gensim.data import GensimCorpus

if TYPE_CHECKING:
    from transformer_document_embedding.tasks.experimental_task import ExperimentalTask
    from datasets.arrow_dataset import Dataset
    from gensim.models import Doc2Vec
    import numpy as np


logger = logging.getLogger(__name__)


class EvaluateIRMetrics(CallbackAny2Vec):
    """Callback to save model after each epoch."""

    def __init__(
        self,
        *,
        baseline: ParagraphVectorWikipediaSimilarities,
        val_dataset: Dataset,
        eval_every: int,
        log_dir: str,
        save_best_path: Optional[str] = None,
        hits_thresholds: list[int] = None,
        decisive_metric: str = "mean_percentile_rank",
        higher_is_better: bool = False,
    ):
        if hits_thresholds is None:
            hits_thresholds = [10, 100]
        self._baseline = baseline
        self._true_dataset = val_dataset
        self._eval_every = eval_every

        self._save_best_path = save_best_path
        self._hits_thresholds = hits_thresholds
        self._decisive_metric = decisive_metric
        self._higher_is_better = higher_is_better
        self._log_dir = log_dir

        self._epoch = 0
        self._best_score = float("inf")
        if self._higher_is_better:
            self._best_score *= -1

    def on_epoch_end(self, model: Doc2Vec) -> None:
        self._epoch += 1
        if self._epoch % self._eval_every == 0:
            prefix = "dm" if model.dm else "dbow"
            self.evaluate(prefix)

    def _is_best(self, score: float) -> bool:
        if self._higher_is_better:
            return score > self._best_score
        else:
            return score < self._best_score

    def evaluate(self, log_name_prefix: str) -> None:
        logger.info("Evaluating %s model.", log_name_prefix)

        pred_embeddings = self._baseline.predict(self._true_dataset)
        true_pred_ids_iter = wiki_sims_task.get_nearest_ids_from_faiss(
            self._true_dataset, pred_embeddings
        )

        metrics = evaluate_ir_metrics(
            true_pred_ids_iter, hits_thresholds=self._hits_thresholds
        )

        score = metrics[self._decisive_metric]
        if self._save_best_path is not None and self._is_best(score):
            logger.info(
                "Saving best %s model to %s.",
                ParagraphVectorWikipediaSimilarities.__name__,
                self._save_best_path,
            )

            self._best_score = score
            self._baseline.save(self._save_best_path)

        import tensorflow as tf

        writer = tf.summary.create_file_writer(self._log_dir)
        with writer.as_default():
            for name, score in metrics.items():
                tf.summary.scalar(f"{log_name_prefix}_{name}", score, self._epoch)


class ParagraphVectorWikipediaSimilarities(ExperimentalModel):
    def __init__(
        self,
        dm_kwargs: Optional[dict[str, Any]] = None,
        dbow_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self._pv = ParagraphVector(dm_kwargs=dm_kwargs, dbow_kwargs=dbow_kwargs)

    def train(
        self,
        task: ExperimentalTask,
        *,
        log_dir: Optional[str] = None,
        save_best_path: Optional[str] = None,
        early_stopping: bool = False,
    ) -> None:
        train_data = GensimCorpus(task.train.shuffle())
        callbacks = []
        if log_dir is not None and task.validation is not None:
            callbacks.append(
                EvaluateIRMetrics(
                    baseline=self,
                    val_dataset=task.validation,
                    eval_every=10,
                    save_best_path=save_best_path,
                    log_dir=log_dir,
                )
            )

        for module in self._pv.modules:
            module.build_vocab(train_data)
            module.train(
                train_data,
                total_examples=module.corpus_count,
                epochs=module.epochs,
                callbacks=callbacks,
            )

        if save_best_path and log_dir is None:
            # We should save the model, but we don't know validation metrics.
            self._pv.save(save_best_path)

    def predict(self, inputs: Dataset) -> Iterable[np.ndarray]:
        for doc in inputs:
            yield self._pv.get_vector(doc["id"])

    def save(self, dir_path: str) -> None:
        self._pv.save(dir_path)

    def load(self, dir_path: str) -> None:
        self._pv.load(dir_path)
