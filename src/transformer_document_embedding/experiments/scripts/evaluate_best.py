"""Runs experiment for given model and task.

# TODO: Document this module once the interface settles...
"""
import argparse
import logging
import os
import pprint
from typing import Optional

import tensorflow as tf

import transformer_document_embedding as tde

EXPERIMENTS_DIR = "./results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        action="extend",
        help=(
            "Experiment configurations to be run described by YAML files. Required"
            " syntax to be found here:"
            " github.com/dburian/transformer_document_embedding/"
            "blob/master/log/experiment_files.md."
        ),
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--output_base_path",
        type=str,
        default=EXPERIMENTS_DIR,
        help=(
            "Path to directory containing all experiment results. Default:"
            f" '{EXPERIMENTS_DIR}'."
        ),
    )
    parser.add_argument(
        "--save_best",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to save the best model. This is either checkpoint if validation"
            " data is available, or trained model."
        ),
    )
    parser.add_argument(
        "--early_stop",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to stop training when the validation loss stops decreasing.",
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help=(
            "Path from which to load the fitted model instead of fitting it (which is"
            " the default behaviour)."
        ),
    )

    args = parser.parse_args()
    return args


def evaluate_best(
    config: tde.experiments.ExperimentConfig,
    save_best: bool,
    early_stopping: bool,
    load_model_path: Optional[str],
) -> None:
    logging.info(
        "Starting experiment with config:\n%s",
        pprint.pformat(config.values, indent=1),
    )
    model = config.get_model_type()(**config.values["model"].get("kwargs", {}))
    task = config.get_task_type()(**config.values["task"].get("kwargs", {}))

    if load_model_path is not None:
        logging.info("Loading model from %s.", load_model_path)
        model.load(load_model_path)
    else:
        logging.info("Training model...")

        save_best_path = None
        if save_best:
            logging.info("Saving best checkpoint of model to %s", config.model_path)
            save_best_path = config.model_path

        model.train(
            task,
            log_dir=config.experiment_path,
            save_best_path=save_best_path,
            early_stopping=early_stopping,
        )
        logging.info("Training done.")

    if not save_best:
        logging.info("Saving trained model to %s.", config.model_path)
        model.save(config.model_path)

    logging.info("Evaluating on test data...")
    test_predictions = model.predict(task.test)
    results = task.evaluate(test_predictions)
    logging.info("Evaluation done. Results:\n%s", results)

    tde.experiments.save_csv_results(results, config.experiment_path)

    test_log_path = os.path.join(config.experiment_path, "test")
    with tf.summary.create_file_writer(test_log_path).as_default():
        for name, res in results.items():
            tf.summary.scalar(name, res, step=1)

        tf.summary.flush()

    config.save()
    return results


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    for exp_file in args.config:
        config = tde.experiments.ExperimentConfig.from_yaml(
            exp_file, args.output_base_path
        )

        evaluate_best(
            config,
            args.save_best,
            args.early_stop,
            args.load_model_path,
        )


if __name__ == "__main__":
    main()
