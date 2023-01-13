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
from typing import Any, Optional

import pkg_resources
import yaml

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import transformer_document_embedding as tde

MODEL_MODULE_PREFIX = "transformer_document_embedding.models"
TASK_MODULE_PREFIX = "transformer_document_embedding.tasks"
EXPERIMENTS_DIR = "./results"
CONF_REQUIRED_FIELDS = [
    (["tde_version"], "version of transformer_document_embedding package"),
    (
        ["model", "module"],
        f"model's module path (gets prepended with {MODEL_MODULE_PREFIX})",
    ),
    (
        ["task", "module"],
        f"task's module path (gets prepended with {TASK_MODULE_PREFIX})",
    ),
]

ExpConfig = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        action="extend",
        help=(
            "Experiments to be run described by YAML files. Required syntax to be found"
            " here:"
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
        "--save_model",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save model after training.",
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


def parse_experiment_file(file_path: str) -> ExpConfig:
    def check_field_exist(exp: ExpConfig, field_path: list[str]) -> bool:
        field = exp
        for breadcrumb in field_path:
            if breadcrumb not in field:
                return False
            field = field[breadcrumb]
        return True

    with open(file_path, mode="r", encoding="utf8") as file:
        experiment = yaml.safe_load(file)

        for field_path, desc in CONF_REQUIRED_FIELDS:
            if not check_field_exist(experiment, field_path):
                logging.error(
                    "Parsing of experiment file %s failed. Missing required field:"
                    " %s - %s.",
                    file_path,
                    ".".join(field_path),
                    desc,
                )

        tde_version = pkg_resources.get_distribution(
            "transformer_document_embedding"
        ).version
        if experiment["tde_version"] != tde_version:
            logging.warning(
                "Experiment %s designed for different version of"
                " transformer_document_embedding, results might differ. Experiment's"
                " version: %s, current version: %s",
                file_path,
                experiment["version"],
                tde_version,
            )

        return experiment


def create_experiment_dirs(exp: ExpConfig, base_path: str) -> tuple[str, str]:
    exp_path = os.path.join(
        base_path,
        exp["task"]["module"],
        exp["model"]["module"],
        f'{datetime.now().strftime("%Y-%m-%d_%H%M%S")}',
    )
    model_path = os.path.join(exp_path, "model")

    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    return exp_path, model_path


def run_single(
    config: ExpConfig,
    output_base_path: str,
    save_model: bool,
    load_model_path: Optional[str],
) -> None:
    # pylint: disable=invalid-name
    Model = importlib.import_module(
        MODEL_MODULE_PREFIX + "." + config["model"]["module"]
    ).Model
    # pylint: disable=invalid-name
    Task = importlib.import_module(
        TASK_MODULE_PREFIX + "." + config["task"]["module"]
    ).Task

    exp_path, model_path = create_experiment_dirs(config, output_base_path)

    model = Model(log_dir=exp_path, **config["model"].get("kwargs", {}))
    task = Task(**config["task"].get("kwargs", {}))

    if load_model_path is not None:
        logging.info("Loading model from %s.", load_model_path)
        model.load(load_model_path)
    else:
        logging.info("Training model...")
        model.train(task.train)
        logging.info("Training done.")

    if save_model:
        logging.info("Saving trained model to %s", model_path)
        model.save(model_path)

    logging.info("Evaluating on test data...")
    test_predictions = model.predict(task.test)
    results = task.evaluate(test_predictions)
    logging.info("Evaluation done. Results:\n%s", results)

    tde.evaluation.save_results(results, exp_path)

    config_out_file = os.path.join(exp_path, "config.yaml")
    logging.info("Saving experiment config to %s.", config_out_file)
    with open(config_out_file, mode="w", encoding="utf8") as config_file:
        yaml.dump(config, config_file)


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    for exp_file in args.experiment:
        experiment = parse_experiment_file(exp_file)
        run_single(
            experiment,
            args.output_base_path,
            args.save_model,
            args.load_model_path,
        )


if __name__ == "__main__":
    main()
