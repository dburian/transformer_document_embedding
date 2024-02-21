"""Evaluates previously trained embedding models.

For each dataset a correct head is selected and fine-tuned. The whole model is
evaluated and the results are aggregated into single file.
"""
from __future__ import annotations
import os
import logging
import pprint
from typing import Optional, TYPE_CHECKING

import coolname
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

if TYPE_CHECKING:
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


def eval_single_dataset(
    config: EvaluationInstanceSpec,
    args: argparse.Namespace,
    model: EmbeddingModel,
    exp_path: str,
) -> dict[str, float]:
    save_config(config, exp_path)

    dataset = config.dataset.initialize()
    head = None if config.head is None else config.head.initialize()

    training_pipeline = finetune_factory(
        dataset.EVALUATION_KIND, config.finetune_pipeline_kwargs
    )
    training_pipeline(model, head, dataset, exp_path)

    if args.save_trained_model:
        save_path = os.path.join(exp_path, "trained_model")
        model.save_weights(save_path)

    if head is not None and args.save_trained_head:
        save_model_weights(head, os.path.join(exp_path, "trained_head"))

    return evaluate(model, head, dataset, exp_path)


def find_config(path: str) -> Optional[str]:
    folders = path.split(os.path.sep)
    while len(folders) > 0:
        config_path = os.path.join(*folders, "config.yaml")
        if os.path.isfile(config_path):
            return config_path

        folders = folders[:-1]

    return None


def get_unique_names(paths: list[str]) -> list[str]:
    for num_levels in range(1, max(len(os.path.split(path)) for path in paths)):
        names = ["-".join(path.split(os.path.sep)[-num_levels:]) for path in paths]

        if len(names) == len(set(names)):
            return names

    return ["-".join(os.path.split(path)) for path in paths]


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
    model.load_weights(model_weights_path)

    results = {}
    for eval_name, eval_spec in eval_config.evaluations.items():
        eval_path = os.path.join(model_eval_path, eval_name)
        os.makedirs(eval_path, exist_ok=True)

        config = EvaluationInstanceSpec(
            model=model_config,
            dataset=eval_spec.dataset,
            head=eval_spec.head,
            finetune_pipeline_kwargs=eval_spec.finetune_pipeline_kwargs,
        )

        logging.info(
            "Starting evaluation of '%s' on '%s' with config:\n%s",
            model_name,
            eval_name,
            pprint.pformat(config, indent=1),
        )
        results[eval_name] = eval_single_dataset(
            config,
            args,
            model,
            eval_path,
        )

    save_results(results, model_eval_path)


def main() -> None:
    args = parse_args()
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    eval_config = EvaluationsSpec.from_dict(load_yaml(args.config))

    unique_names = get_unique_names([os.path.dirname(path) for path in args.model_path])

    for model_path, model_name in zip(args.model_path, unique_names, strict=True):
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
