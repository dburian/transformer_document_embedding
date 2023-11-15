"""Runs experiment for given model and task.

# TODO: Document this module once the interface settles...
"""
from __future__ import annotations
import argparse
import logging

from transformer_document_embedding.experiments.config import (
    HPSearchExperimentConfig,
    load_config_values,
)
from transformer_document_embedding.experiments.search import GridSearch, OneSearch
from transformer_document_embedding.scripts.args import add_common_args
from transformer_document_embedding.scripts.pipelines import (
    InitializeModelAndTask,
    TrainingPipeline,
)

HS_BASE_PATH = "./hp_searches"

training_pipeline = TrainingPipeline(train=True)
initialization_pipeline = InitializeModelAndTask()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--grid_config",
        "--gc",
        type=str,
        default=None,
        help="Path to yaml file defining parameters to grid search.",
    )
    parser.add_argument(
        "--one_config",
        "--oc",
        type=str,
        default=None,
        help="Path to yaml file defining parameters to 'one' search.",
    )

    add_common_args(parser, output_base_path=HS_BASE_PATH)
    training_pipeline.add_args(parser)
    initialization_pipeline.add_args(parser)

    return parser.parse_args()


def run_single(config: HPSearchExperimentConfig, args: argparse.Namespace) -> None:
    model, task = initialization_pipeline.run(config)

    training_pipeline.run(args, model, task, config)

    config.log_hparams()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    assert args.grid_config is None or args.one_config is None, (
        "Cannot simultaneously run grid search and one search. "
        "Please issue two commands instead."
    )

    for search_cls, config_path in [
        (GridSearch, args.grid_config),
        (OneSearch, args.one_config),
    ]:
        if config_path is None:
            continue

        search = search_cls.from_yaml(
            config_path, args.output_base_path, name=args.name
        )
        base_config_values = load_config_values(args.config)

        for experiment_instance in search.based_on(base_config_values):
            run_single(experiment_instance, args)


if __name__ == "__main__":
    main()
