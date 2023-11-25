"""Evaluate pre-trained model on a number of tasks at once."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import logging
import os

from tqdm.auto import tqdm
import yaml


from typing import TYPE_CHECKING
from transformer_document_embedding.scripts.pipelines import add_common_args

from transformer_document_embedding.scripts.config_specs import (
    ExperimentSpec,
    EvaluationSpec,
)
from transformer_document_embedding.scripts.utils import (
    init_type,
    load_yaml,
    log_results,
)

if TYPE_CHECKING:
    from transformer_document_embedding.tasks.experimental_task import ExperimentalTask
    from transformer_document_embedding.models.experimental_model import (
        ExperimentalModel,
    )


EVALUATIONS_OUTPUT_BASE_PATH = "./evaluations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    add_common_args(parser, output_base_path=EVALUATIONS_OUTPUT_BASE_PATH)

    parser.add_argument(
        "--experiment_config",
        "-e",
        nargs="+",
        action="extend",
        type=str,
        help="Paths to configuration of the saved model to evaluate.",
    )

    parser.add_argument(
        "--model_dir",
        default="model",
        type=str,
        help="Relative path from configuration to directory with saved model",
    )

    return parser.parse_args()


def evaluate_single(
    config: EvaluationSpec,
    model: ExperimentalModel,
    model_name: str,
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    exp_path = os.path.join(
        args.output_base_path,
        args.name,
        model_name,
    )
    logging.info(
        "Evaluating model '%s' on '%s'.",
        model_name,
        ",".join((task_spec.module for task_spec in config.tasks)),
    )

    metrics = []
    for task_spec in tqdm(
        config.tasks, desc="Tasks evaluated", total=len(config.tasks)
    ):
        task: ExperimentalTask = init_type(task_spec)

        test_predictions = model.predict(task.test)
        task_metrics = task.evaluate(task.test, test_predictions)
        logging.info("Evaluation done. Results:\n%s", metrics)

        metrics.append(task_metrics)

        test_log_path = os.path.join(exp_path, task_spec.name)
        log_results(test_log_path, task_metrics)

    with open(
        os.path.join(exp_path, "results.yaml"), mode="w", encoding="utf8"
    ) as results_file:
        results_with_spec = [
            asdict(task_spec) | {"results": task_metrics}
            for task_spec, task_metrics in zip(config.tasks, metrics, strict=True)
        ]
        yaml.dump(results_with_spec, results_file)

    return metrics


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    tasks_config = load_yaml(args.config)

    for model_config_path in args.experiment_config:
        model_config = ExperimentSpec.from_dict(load_yaml(model_config_path)).model

        model: ExperimentalModel = init_type(model_config)
        model_config_dir = os.path.dirname(model_config_path)
        model_load_dir = os.path.join(model_config_dir, args.model_dir)
        model.load(model_load_dir)

        eval_config = EvaluationSpec.from_dict(
            {
                **tasks_config,
                "model": asdict(model_config),
                "evaluated_model_path": model_load_dir,
            }
        )
        model_name = os.path.basename(model_config_dir)

        evaluate_single(eval_config, model, model_name, args)


if __name__ == "__main__":
    main()
