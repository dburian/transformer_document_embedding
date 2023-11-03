"""Runs experiment for given model and task.

# TODO: Document this module once the interface settles...
"""
from __future__ import annotations

import argparse
import logging
import os
import pprint

from transformer_document_embedding.scripts.args import (
    add_common_args,
)


from transformer_document_embedding.experiments.config import ExperimentConfig
from transformer_document_embedding.experiments.result import save_csv_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    add_common_args(parser)

    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help=(
            "Path from which to load the fitted model instead of fitting it (which is"
            " the default behaviour)."
        ),
    )

    parser.add_argument(
        "--save_trained",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to save trained model.",
    )

    return parser.parse_args()


def log_results(log_path: str, results: dict[str, float]) -> None:
    import tensorflow as tf

    with tf.summary.create_file_writer(log_path).as_default():
        for name, res in results.items():
            tf.summary.scalar(name, res, step=1)

        tf.summary.flush()


def evaluate_best(
    config: ExperimentConfig,
    args: argparse.Namespace,
) -> None:
    logging.info(
        "Starting experiment '%s', with config:\n%s",
        config.name,
        pprint.pformat(config.values, indent=1),
    )
    config.save()

    model = config.get_model_type()(**config.values["model"].get("kwargs", {}))
    task = config.get_task_type()(**config.values["task"].get("kwargs", {}))

    if args.load_model_path is not None:
        logging.info("Loading model from %s.", args.load_model_path)
        model.load(args.load_model_path)
    else:
        logging.info("Training model...")

        model.train(
            task,
            log_dir=config.experiment_path,
            model_dir=config.model_path,
            **config.values["model"].get("train_kwargs", {}),
        )
        logging.info("Training done.")

        if args.save_trained:
            trained_path = os.path.join(config.model_path, "trained")
            logging.info(
                "Saving trained model to %s.",
                trained_path,
            )
            os.makedirs(trained_path, exist_ok=True)
            model.save(trained_path)

    logging.info("Evaluating on test data...")
    test_predictions = model.predict(task.test)
    results = task.evaluate(task.test, test_predictions)
    logging.info("Evaluation done. Results:\n%s", results)

    save_csv_results(results, config.experiment_path)

    test_log_path = os.path.join(config.experiment_path, "test")
    log_results(test_log_path, results)

    return results


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    for exp_file in args.config:
        config = ExperimentConfig.from_yaml(exp_file, args.output_base_path, args.name)

        evaluate_best(config, args)


if __name__ == "__main__":
    main()
