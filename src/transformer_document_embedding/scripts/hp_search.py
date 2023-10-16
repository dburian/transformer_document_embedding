"""Runs experiment for given model and task.

# TODO: Document this module once the interface settles...
"""
from __future__ import annotations
import argparse
import logging
import pprint

from transformer_document_embedding.experiments.config import ExperimentConfig
from transformer_document_embedding.experiments.search import GridSearch, OneSearch
from transformer_document_embedding.scripts.args import add_common_args

EXPERIMENTS_DIR = "./results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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

    add_common_args(parser)

    return parser.parse_args()


def run_single(config: ExperimentConfig) -> None:
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
        **config.values["model"].get("train_kwargs", {}),
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
            exp_config = ExperimentConfig.from_yaml(exp_file, args.output_base_path)

            for experiment_instance in search.based_on(exp_config):
                run_single(experiment_instance)


if __name__ == "__main__":
    main()
