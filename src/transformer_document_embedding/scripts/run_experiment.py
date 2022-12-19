"""Runs experiment for given model and task.

Both model and tasks are defined as module paths which get automatically
prefixed with 'transformer_document_embedding.models.' for model and
'transformer_document_embedding.tasks.' for task.

Experiment results vary and are saved under ./results.

Example usage:
    run-experiment -m imdb_doc2vec -t imdb

"""
import argparse
import importlib
import logging
import os
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

MODEL_PACKAGE_PREFIX = "transformer_document_embedding.models"
TASK_PACKAGE_PREFIX = "transformer_document_embedding.tasks"
EXPERIMENTS_DIR = "/home/dburian/docs/transformer_document_embedding/results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model's module within " + MODEL_PACKAGE_PREFIX + ".",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        help="Task's module within " + TASK_PACKAGE_PREFIX + ".",
        required=True,
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        default=None,
        help=(
            "Where to save experiment results. By default derived from 'task' and"
            " 'model'."
        ),
    )
    parser.add_argument(
        "--save_model",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save model after training.",
    )
    parser.add_argument(
        "--load_model",
        type=boo,
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether to load model from `--model_save_path` instead of training new"
            " one."
        ),
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=None,
        help=(
            "Path where to save the model once fitted. By default derived from"
            " 'experiment_path'."
        ),
    )

    args = parser.parse_args()

    if args.experiment_path is None:
        args.experiment_path = os.path.join(
            EXPERIMENTS_DIR,
            *args.task.split("."),
            *args.model.split("."),
            f'{datetime.now().strftime("%Y-%m-%d_%H%M%S")}',
        )

    if args.model_save_path is None:
        args.model_save_path = os.path.join(args.experiment_path, "model_save")

    os.makedirs(args.experiment_path, exist_ok=True)
    os.makedirs(args.model_save_path, exist_ok=True)

    return args


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    model_pkg = importlib.import_module(MODEL_PACKAGE_PREFIX + "." + args.model)
    task_pkg = importlib.import_module(TASK_PACKAGE_PREFIX + "." + args.task)

    model = model_pkg.Model(log_dir=args.experiment_path)
    task = task_pkg.Task()

    if args.load_model:
        model.load(args.model_save_path)
    else:
        print("Training")
        model.train(task.train)

    if args.save_model:
        print("Saving.")
        model.save(args.model_save_path)

    print("Evaluating.")
    test_predictions = model.predict(task.test)
    results = task.evaluate(test_predictions)

    print("Saving results")
    # utils.save_results(results, args.experiment_path)

    print("Finished")


if __name__ == "__main__":
    main()
