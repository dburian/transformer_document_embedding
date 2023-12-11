"""Runs hyperparameter search on a model and a task."""

from __future__ import annotations
import argparse
from copy import deepcopy
import re
from itertools import product
import logging
from transformer_document_embedding.scripts.config_specs import ExperimentSpec


from transformer_document_embedding.scripts.utils import (
    load_yaml,
)
from transformer_document_embedding.scripts.pipelines import (
    InitializeModelAndTask,
    TrainingPipeline,
    add_common_args,
)

import os
from typing import Any, Iterable


HS_BASE_PATH = "./hp_searches"

training_pipeline = TrainingPipeline(train=True)
initialization_pipeline = InitializeModelAndTask()


def log_hparams(
    flattened_hparams: dict[str, Any], trial_id: str, output_path: str
) -> None:
    import tensorflow as tf
    from tensorboard.plugins.hparams import api as hp

    with tf.summary.create_file_writer(output_path).as_default():
        # tf is unable to log NoneTypes
        for key, value in flattened_hparams.items():
            if value is None:
                flattened_hparams[key] = "None"
            if isinstance(value, list) or isinstance(value, dict):
                flattened_hparams[key] = str(value)

        hp.hparams(flattened_hparams, trial_id)

        tf.summary.flush()


def deep_update_with_flatten(
    self: dict[Any, Any], flatten: dict[Any, Any]
) -> dict[Any, Any]:
    for key, new_value in flatten.items():
        crumbs = key.split(".")
        dct = self
        for crumb in crumbs[:-1]:
            if crumb not in dct:
                dct[crumb] = {}
            dct = dct[crumb]

        dct[crumbs[-1]] = new_value

    return self


def grid_search(
    hparams: dict[str, Any], reference_values: dict[str, Any]
) -> Iterable[tuple[dict[str, Any], ExperimentSpec]]:
    options_per_key = (
        [(key, value) for value in hparams[key]] for key in hparams.keys()
    )
    # product yields a single hparam setting as a tuple of (key, value) tuples
    for flattened_hparams in map(dict, product(*options_per_key)):
        new_values = deep_update_with_flatten(
            deepcopy(reference_values), flattened_hparams
        )
        yield flattened_hparams, ExperimentSpec.from_dict(new_values)


def one_search(
    hparams: dict[str, Any], reference_values: dict[str, Any]
) -> Iterable[tuple[dict[str, Any], ExperimentSpec]]:
    options = ({key: value} for key in hparams.keys() for value in hparams[key])
    for flattened_hparams in options:
        new_values = deep_update_with_flatten(
            deepcopy(reference_values), flattened_hparams
        )
        yield flattened_hparams, ExperimentSpec.from_dict(new_values)


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


def generate_name_from_dict(dct: dict[str, Any]) -> str:
    name_parts = []
    for key, value in dct.items():
        short_key = re.sub("([a-zA-Z])[a-zA-Z]+", r"\1", key)
        value_str = value
        if isinstance(value, list):
            value_str = f"[{','.join(map(str, value))}]"
        elif isinstance(value, dict):
            value_str = f"{{{generate_name_from_dict(value)}}}"
        name_parts.append(f"{short_key}={value_str}")

    return "-".join(name_parts)


def search_single(
    config: ExperimentSpec,
    flattened_hparams: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    exp_name = generate_name_from_dict(flattened_hparams)
    exp_path = os.path.join(args.output_base_path, args.name, exp_name)
    os.makedirs(exp_path, exist_ok=True)

    model, task = initialization_pipeline.run(exp_name, exp_path, config)

    training_pipeline.run(args, model, task, exp_path, config)

    trial_id = os.path.join(args.name, exp_name.replace("/", "-"))
    log_hparams(flattened_hparams, trial_id, exp_path)


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    assert args.grid_config is None or args.one_config is None, (
        "Cannot simultaneously run grid search and one search. "
        "Please issue two commands instead."
    )

    reference_config = load_yaml(args.config)
    for search_fn, config_path in [
        (grid_search, args.grid_config),
        (one_search, args.one_config),
    ]:
        if config_path is None:
            continue

        hparams = load_yaml(config_path)

        for hparam_instance, experiment_spec in search_fn(hparams, reference_config):
            search_single(experiment_spec, hparam_instance, args)


if __name__ == "__main__":
    main()
