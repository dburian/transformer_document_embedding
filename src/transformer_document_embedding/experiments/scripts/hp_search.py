"""Runs experiment for given model and task.

# TODO: Document this module once the interface settles...
"""
import argparse
import logging
import pprint

import transformer_document_embedding as tde
from transformer_document_embedding.experiments.search import GridSearch, OneSearch

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
        "--grid_config",
        "--gc",
        type=str,
        default=None,
        help=(
            "Run grid search over all arguments specified in grid search YAML"
            " file for each experiment. Syntax documented here: TODO."
        ),
    )
    parser.add_argument(
        "--one_config",
        "--oc",
        type=str,
        default=None,
        help=(
            "Run one-search by trying all values sequentially together with all"
            " experiment configs."
        ),
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
        "--early_stop",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to stop training when the validation loss stops decreasing.",
    )

    args = parser.parse_args()
    return args


def run_single(
    config: tde.experiments.ExperimentConfig,
    early_stopping: bool,
) -> None:
    logging.info(
        "Starting experiment with config:\n%s",
        pprint.pformat(config.values, indent=1),
    )
    model = config.get_model_type()(**config.values["model"].get("kwargs", {}))
    task = config.get_task_type()(**config.values["task"].get("kwargs", {}))

    logging.info("Training model...")

    model.train(
        task,
        log_dir=config.experiment_path,
        early_stopping=early_stopping,
    )
    logging.info("Training done.")

    config.log_hparams()
    config.save()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    for search_cls, config_path in [
        (GridSearch, args.grid_config),
        (OneSearch, args.one_config),
    ]:
        if config_path is None:
            continue

        search = search_cls.from_yaml(config_path)
        for exp_file in args.config:
            config_path = tde.experiments.ExperimentConfig.from_yaml(
                exp_file, args.output_base_path
            )

            for experiment_instance in search.based_on(config_path):
                run_single(experiment_instance, args.early_stop)


if __name__ == "__main__":
    main()
