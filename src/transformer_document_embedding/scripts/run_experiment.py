import argparse
import importlib
import os
from datetime import datetime

MODEL_PACKAGE_PREFIX = "transformer_document_embedding.models"
TASK_PACKAGE_PREFIX = "transformer_document_embedding.tasks"
EXPERIMENTS_DIR = "/home/dburian/docs/transformer_document_embedding/results"


def main(args: argparse.Namespace) -> None:
    model = importlib.import_module(MODEL_PACKAGE_PREFIX + "." + args.model).Model
    task = importlib.import_module(TASK_PACKAGE_PREFIX + "." + args.task).Task
    print(args)

    print("Fitting")
    model.fit(task.train)

    model.save()


if __name__ == "__main__":
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

    main(args)
