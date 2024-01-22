from __future__ import annotations

import os
import logging
from typing import TYPE_CHECKING
from datasets import concatenate_datasets

from gensim.models.callbacks import CallbackAny2Vec

import transformer_document_embedding.tasks.wikipedia_similarities as wiki_sims_task
from ..experimental_model import ExperimentalModel
from .paragraph_vector import ParagraphVector
from transformer_document_embedding.utils.evaluation import evaluate_ir_metrics
from transformer_document_embedding.utils.gensim.data import GensimCorpus

if TYPE_CHECKING:
    from transformer_document_embedding.tasks.experimental_task import ExperimentalTask
    from datasets.arrow_dataset import Dataset
    from gensim.models import Doc2Vec
    from typing import Any, Iterable, Optional
    import numpy as np


logger = logging.getLogger(__name__)


class EvaluateIRMetrics(CallbackAny2Vec):
    """Callback to to evaluate IR metrics."""

    def __init__(
        self,
        *,
        model: ParagraphVectorEmbed,
        val_dataset: Dataset,
        eval_every: int,
        log_dir: str,
        save_best_path: Optional[str] = None,
        hits_thresholds: Optional[list[int]] = None,
        decisive_metric: str = "mean_percentile_rank",
        higher_is_better: bool = False,
    ):
        if hits_thresholds is None:
            hits_thresholds = [10, 100]
        self._model = model
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

        pred_embeddings = self._model.predict(self._true_dataset)
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
                ParagraphVectorEmbed.__name__,
                self._save_best_path,
            )

            self._best_score = score
            self._model.save(self._save_best_path)

        import tensorflow as tf

        writer = tf.summary.create_file_writer(self._log_dir)
        with writer.as_default():
            for name, score in metrics.items():
                tf.summary.scalar(f"{log_name_prefix}_{name}", score, self._epoch)


class CheckpointSave(CallbackAny2Vec):
    """Callback to periodically save the model."""

    def __init__(
        self,
        epoch_checkpoints: list[int],
        save_dir: str,
        paragraph_vector: ParagraphVector,
    ) -> None:
        self._epoch_checkpoints = epoch_checkpoints
        self._save_dir = save_dir
        self._epoch = 0
        self._pv = paragraph_vector

        os.makedirs(self._save_dir, exist_ok=True)

    def on_epoch_end(self, _: Doc2Vec) -> None:
        if self._epoch in self._epoch_checkpoints:
            model_path = os.path.join(self._save_dir, f"after_epoch_{self._epoch}")
            os.makedirs(model_path, exist_ok=True)
            self._pv.save(model_path)

        self._epoch += 1


def compute_alpha(
    total_epochs: int,
    cur_epoch: int,
    start_alpha: float = 0.025,
    end_alpha: float = 1e-4,
) -> float:
    progress = cur_epoch / total_epochs
    next_alpha = start_alpha - (start_alpha - end_alpha) * progress
    next_alpha = max(end_alpha, next_alpha)
    return next_alpha


class ParagraphVectorEmbed(ExperimentalModel):
    def __init__(
        self,
        dm_kwargs: Optional[dict[str, Any]] = None,
        dbow_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self._pv = ParagraphVector(dm_kwargs=dm_kwargs, dbow_kwargs=dbow_kwargs)

    def train(
        self,
        task: ExperimentalTask,
        starts_at_epoch: Optional[int],
        save_at_epochs: Optional[list[int]],
        save_best: bool,
        log_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        all_datasets = [task.train]
        if task.validation is not None:
            all_datasets.append(task.validation)

        train_data = GensimCorpus(concatenate_datasets(all_datasets).shuffle())
        callbacks = []
        # if log_dir is not None and task.validation is not None:
        #     callbacks.append(
        #         EvaluateIRMetrics(
        #             model=self,
        #             val_dataset=task.validation,
        #             eval_every=10,
        #             save_best_path=model_dir
        #             if save_best and model_dir is not None
        #             else None,
        #             log_dir=log_dir,
        #         )
        #     )

        if log_dir is not None and save_at_epochs is not None:
            callbacks.append(
                CheckpointSave(
                    epoch_checkpoints=save_at_epochs,
                    save_dir=os.path.join(log_dir, "checkpoints"),
                    paragraph_vector=self._pv,
                )
            )

        for module in self._pv.modules:
            if starts_at_epoch is None:
                module.build_vocab(train_data)

            train_kwargs: dict[str, Any] = {
                "callbacks": callbacks,
            }

            if starts_at_epoch is not None:
                train_kwargs["start_alpha"] = compute_alpha(
                    total_epochs=module.epochs,
                    cur_epoch=starts_at_epoch,
                )
                train_kwargs["end_alpha"] = 1e-4

            module.train(
                train_data,
                total_examples=module.corpus_count,
                epochs=module.epochs,
                **train_kwargs,
            )

        if save_best and log_dir is not None and log_dir is None:
            # We should save the model, but we don't know validation metrics.
            self.save(os.path.join(log_dir, "model"))

    def predict(self, inputs: Dataset) -> Iterable[np.ndarray]:
        for doc in inputs:
            yield self._pv.get_vector(doc["id"])

    def save(self, dir_path: str) -> None:
        self._pv.save(dir_path)

    def load(self, dir_path: str) -> None:
        self._pv.load(dir_path)
