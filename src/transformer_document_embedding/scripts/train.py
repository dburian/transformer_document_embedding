"""Trains and optionally saves a model on a task."""
from __future__ import annotations

import argparse
import logging
import os
from transformer_document_embedding.scripts.config_specs import ExperimentSpec


from transformer_document_embedding.scripts.pipelines import (
    InitializeModelAndTask,
    TrainingPipeline,
    add_common_args,
)
from transformer_document_embedding.scripts.utils import (
    log_results,
    load_yaml,
    save_results,
)


training_pipeline = TrainingPipeline(train=True)
initialization_pipeline = InitializeModelAndTask()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    add_common_args(parser)
    training_pipeline.add_args(parser)
    initialization_pipeline.add_args(parser)

    return parser.parse_args()


def train(
    config: ExperimentSpec,
    args: argparse.Namespace,
) -> dict[str, float]:
    exp_path = os.path.join(
        args.output_base_path, config.task.module, config.model.module, args.name
    )
    os.makedirs(exp_path, exist_ok=True)

    model, task = initialization_pipeline.run(args.name, exp_path, config)

    training_pipeline.run(args, model, task, exp_path, config)

    logging.info("Evaluating on test data...")
    test_predictions = model.predict(task.test)
    results = task.evaluate(task.test, test_predictions)
    logging.info("Evaluation done. Results:\n%s", results)

    save_results(results, exp_path)

    test_log_path = os.path.join(exp_path, "test")
    log_results(test_log_path, results)

    return results


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    config = ExperimentSpec.from_dict(load_yaml(args.config))

    train(config, args)


if __name__ == "__main__":
    main()
