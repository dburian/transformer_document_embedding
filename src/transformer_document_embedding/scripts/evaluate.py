"""Evaluates previously trained embedding models.

For each dataset a correct head is selected and fine-tuned. The whole model is
evaluated and the results are aggregated into single file.
"""
from __future__ import annotations
from dataclasses import fields
import os
import logging
import pprint
from typing import Iterator, Optional, TYPE_CHECKING

import coolname
from datasets import DatasetDict
from sklearn.model_selection import StratifiedKFold
from transformer_document_embedding.datasets import col
from transformer_document_embedding.datasets.explicit_document_dataset import (
    ExplicitDocumentDataset,
)
from transformer_document_embedding.pipelines.classification_finetune import (
    get_head_features,
)
from transformer_document_embedding.scripts.common import evaluate
from transformer_document_embedding.scripts.config_specs import (
    EmbeddingModelSpec,
    EvaluationInstanceSpec,
    EvaluationsSpec,
)

from transformer_document_embedding.scripts.utils import (
    load_yaml,
    save_config,
    save_results,
)
import argparse
from transformer_document_embedding.pipelines.finetune_factory import finetune_factory
from transformer_document_embedding.utils.net_helpers import save_model_weights

import numpy as np

if TYPE_CHECKING:
    from transformer_document_embedding.datasets.document_dataset import DocumentDataset
    from transformer_document_embedding.models.embedding_model import EmbeddingModel


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to yaml evaluation config.",
        required=True,
    )

    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        action="extend",
        nargs="+",
        help="Paths to root directories of experiments with learned embedding models.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Format string that for split model's path gives model's name.",
        default="{whole_path}",
    )
    parser.add_argument(
        "--output_base_path",
        type=str,
        default="./evaluations",
        help="Path to directory containing all evaluation results.",
    )

    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="_".join(coolname.generate(2)),
        help="Name of the evaluation. If no name is given, one is generated.",
    )

    parser.add_argument(
        "--save_trained_model",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Whether to save trained model.",
    )

    parser.add_argument(
        "--save_trained_head",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Whether to save trained head.",
    )

    return parser.parse_args()


def iterate_cross_validation_splits(
    base_document_dataset: DocumentDataset, split_name: str, num_folds: int
) -> Iterator[DocumentDataset]:
    cv_dataset = base_document_dataset.splits[split_name]
    labels = cv_dataset.with_format("np")[col.LABEL]
    feats_placeholder = np.zeros(len(labels))

    fold_gen = StratifiedKFold(n_splits=num_folds)
    for train_indices, test_indices in fold_gen.split(feats_placeholder, labels):
        fold = DatasetDict(
            train=cv_dataset.select(train_indices),
            test=cv_dataset.select(test_indices),
        )

        yield ExplicitDocumentDataset(fold, base_document_dataset.evaluation_kind)


def cross_validate_single_dataset(
    config: EvaluationInstanceSpec,
    model: EmbeddingModel,
    exp_path: str,
) -> dict[str, float]:
    assert config.cross_validate is not None
    logger.info(
        "Cross validating '%s' using '%d' folds",
        config.cross_validate.split,
        config.cross_validate.num_folds,
    )

    base_dataset = config.dataset.initialize()

    # Add features for the cross-validated split now to avoid re-generating it
    # for each fold
    base_dataset.splits[config.cross_validate.split] = get_head_features(
        base_dataset.evaluation_kind,
        base_dataset.splits[config.cross_validate.split],
        model,
    )

    fold_results = []
    for fold in iterate_cross_validation_splits(
        base_dataset, config.cross_validate.split, config.cross_validate.num_folds
    ):
        head = None if config.head is None else config.head.initialize(model)

        training_pipeline = finetune_factory(
            fold.evaluation_kind, config.finetune_pipeline_kwargs
        )
        training_pipeline(model, head, fold, exp_path)

        fold_results.append(
            evaluate(model, head, fold, exp_path, write_results_to_disk=False)
        )

    results = {}
    for metric_name in set(key for res in fold_results for key in res.keys()):
        metric_scores = [res[metric_name] for res in fold_results if metric_name in res]
        results[f"{metric_name}_mean"] = np.mean(metric_scores).item()
        results[f"{metric_name}_std"] = np.std(metric_scores).item()

    save_results(results, exp_path)

    return results


def evaluate_single_dataset(
    config: EvaluationInstanceSpec,
    args: argparse.Namespace,
    model: EmbeddingModel,
    exp_path: str,
) -> dict[str, float]:
    save_config(config, exp_path)

    if config.cross_validate is not None:
        return cross_validate_single_dataset(config, model, exp_path)

    dataset = config.dataset.initialize()
    head = None if config.head is None else config.head.initialize(model)

    training_pipeline = finetune_factory(
        dataset.evaluation_kind, config.finetune_pipeline_kwargs
    )
    training_pipeline(model, head, dataset, exp_path)

    if args.save_trained_model:
        save_path = os.path.join(exp_path, "trained_model")
        model.save_weights(save_path)

    if head is not None and args.save_trained_head:
        save_model_weights(head, os.path.join(exp_path, "trained_head"))

    return evaluate(model, head, dataset, exp_path)


def evaluate_model(
    eval_config: EvaluationsSpec,
    model_config: EmbeddingModelSpec,
    args: argparse.Namespace,
    model_weights_path: str,
    model_name: str,
) -> None:
    model_eval_path = os.path.join(args.output_base_path, args.name, model_name)

    model = model_config.initialize()
    logger.info("Loading weights for '%s' from '%s'.", model_name, model_weights_path)
    model.load_weights(model_weights_path, strict=False)

    results = {}
    for eval_name, eval_spec in eval_config.evaluations.items():
        eval_path = os.path.join(model_eval_path, eval_name)
        if os.path.isdir(eval_path):
            eval_results_path = os.path.join(eval_path, "results.yaml")
            if os.path.isfile(eval_results_path):
                logger.info(
                    "Evaluation of '%s' already exists. Loading results from '%s'.",
                    eval_name,
                    eval_results_path,
                )
                results[eval_name] = load_yaml(eval_results_path)
                continue

        os.makedirs(eval_path, exist_ok=True)

        config = EvaluationInstanceSpec(
            model=model_config,
            **{
                field.name: getattr(eval_spec, field.name)
                for field in fields(eval_spec)
            },
        )

        logging.info(
            "Starting evaluation of '%s' on '%s' with config:\n%s",
            model_name,
            eval_name,
            pprint.pformat(config, indent=1),
        )
        results[eval_name] = evaluate_single_dataset(
            config,
            args,
            model,
            eval_path,
        )

    save_results(results, model_eval_path)


def find_config(path: str) -> Optional[str]:
    folders = path.split(os.path.sep)
    while len(folders) > 0:
        config_path = os.path.join(*folders, "config.yaml")
        if os.path.isfile(config_path):
            return config_path

        folders = folders[:-1]

    return None


def main() -> None:
    args = parse_args()
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    eval_config = EvaluationsSpec.from_dict(load_yaml(args.config))

    for model_path in args.model_path:
        model_name = args.model_name.format(
            *model_path.split(os.path.sep),
            whole_path="-".join(model_path.split(os.path.sep)),
        )
        config_path = find_config(model_path)
        if config_path is None:
            logger.error(
                "Cannot find `config.yaml` file for model saved at '%s'. Skipping...",
                model_path,
            )
            continue

        logger.info("Evaluating model with config '%s'", config_path)
        model_config = EmbeddingModelSpec.from_dict(load_yaml(config_path)["model"])

        evaluate_model(
            eval_config,
            model_config,
            args,
            model_weights_path=model_path,
            model_name=model_name,
        )


if __name__ == "__main__":
    main()
