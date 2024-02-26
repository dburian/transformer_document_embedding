from __future__ import annotations
import os
from typing import TYPE_CHECKING, Any
from transformer_document_embedding.datasets.document_dataset import EvaluationKind
from transformer_document_embedding.pipelines.eval_factory import eval_factory
from transformer_document_embedding.pipelines.train_factory import train_factory

from transformer_document_embedding.scripts.utils import save_results
from transformer_document_embedding.utils.net_helpers import (
    load_model_weights,
    save_model_weights,
)
import logging

if TYPE_CHECKING:
    from transformer_document_embedding.models.embedding_model import EmbeddingModel
    from transformer_document_embedding.datasets.document_dataset import DocumentDataset
    import torch
    from transformer_document_embedding.scripts.config_specs import ExperimentSpec
    from typing import Optional

logger = logging.getLogger(__name__)


def load_train_save(
    config: ExperimentSpec,
    load_model_weights_path: Optional[str],
    load_head_weights_path: Optional[str],
    save_trained_model: bool,
    save_trained_head: bool,
    exp_path: str,
) -> tuple[EmbeddingModel, Optional[torch.nn.Module], DocumentDataset]:
    dataset = config.dataset.initialize()
    model = config.model.initialize()

    if load_model_weights_path is not None:
        model.load_weights(load_model_weights_path)

    head = None if config.head is None else config.head.initialize()

    if head is not None and load_head_weights_path is not None:
        load_model_weights(head, load_head_weights_path, strict=True)

    if config.train_pipeline is None:
        return model, head, dataset

    training_pipeline = train_factory(config.train_pipeline)
    training_pipeline(model, head, dataset, exp_path)

    if save_trained_model:
        save_path = os.path.join(exp_path, "trained_model")
        logger.info("Saving model to '%s'.", save_path)
        model.save_weights(save_path)

    if head is not None and save_trained_head:
        save_path = os.path.join(exp_path, "trained_head")
        logger.info("Saving head to '%s'.", save_path)
        save_model_weights(head, save_path)

    return model, head, dataset


def evaluate(
    model,
    head: Optional[torch.nn.Module],
    dataset: DocumentDataset,
    exp_path: str,
) -> dict[str, Any]:
    if dataset.evaluation_kind == EvaluationKind.NONE:
        return {}

    eval_pipeline = eval_factory(dataset.evaluation_kind)
    results = eval_pipeline(model, head, dataset)
    save_results(results, exp_path)
    return results
