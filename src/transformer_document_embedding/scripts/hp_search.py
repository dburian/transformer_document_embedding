"""a __doc__"""
from __future__ import annotations
from copy import deepcopy
from itertools import product
import re
import os
import logging
import pprint
from typing import TYPE_CHECKING, Iterable

import coolname
from transformer_document_embedding.scripts.common import evaluate, load_train_save
from transformer_document_embedding.scripts.config_specs import ExperimentSpec

from transformer_document_embedding.scripts.utils import (
    load_yaml,
    save_config,
)
import argparse

if TYPE_CHECKING:
    from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to yaml experiment file.",
        required=True,
    )
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

    parser.add_argument(
        "--output_base_path",
        type=str,
        default="./hp_searches",
        help="Path to directory containing all experiment results.",
    )

    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="_".join(coolname.generate(2)),
        help="Name of the experiment. If no name is given, one is generated.",
    )

    parser.add_argument(
        "--save_trained_model",
        type=bool,
        action=argparse.BooleanOptionalAction,
        # default=self._save_trained,
        help="Whether to save trained model.",
    )

    parser.add_argument(
        "--save_trained_head",
        type=bool,
        action=argparse.BooleanOptionalAction,
        # default=self._save_trained,
        help="Whether to save trained head.",
    )

    parser.add_argument(
        "--load_model_weights_path",
        type=str,
        default=None,
        help="Path from where to load model's weights.",
    )

    parser.add_argument(
        "--load_head_weights_path",
        type=str,
        default=None,
        help="Path from where to load head's weights.",
    )

    return parser.parse_args()


def generate_filename(input: Any) -> str:
    if isinstance(input, list):
        return f"[{','.join(map(generate_filename, input))}]"

    if isinstance(input, dict):
        name_parts = []
        for key, value in input.items():
            short_key = re.sub("([a-zA-Z])[a-zA-Z]+", r"\1", key)
            value_str = generate_filename(value)
            name_parts.append(f"{short_key}={value_str}")

        return "-".join(name_parts)

    return str(input)


def search_single(
    config: ExperimentSpec, flattened_hparams: dict[str, Any], args: argparse.Namespace
) -> dict[str, float]:
    exp_name = generate_filename(flattened_hparams)
    exp_path = os.path.join(args.output_base_path, args.name, exp_name)
    os.makedirs(exp_path, exist_ok=True)

    logging.info(
        "Starting experiment '%s' with config:\n%s",
        args.name,
        pprint.pformat(config, indent=1),
    )
    save_config(config, exp_path)

    model, head, dataset = load_train_save(
        config,
        load_model_weights_path=args.load_model_weights_path,
        load_head_weights_path=args.load_head_weights_path,
        save_trained_head=args.save_trained_head,
        save_trained_model=args.save_trained_model,
        exp_path=exp_path,
    )

    return evaluate(
        model, head, dataset, exp_path, evaluation_kwargs=config.evaluation_kwargs
    )


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


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    assert args.grid_config is None or args.one_config is None, (
        "Cannot simultaneously run grid search and one search. "
        "Please run two commands instead."
    )

    reference_config = load_yaml(args.config)
    for search_fn, search_config_path in [
        (grid_search, args.grid_config),
        (one_search, args.one_config),
    ]:
        if search_config_path is None:
            continue

        hparams = load_yaml(search_config_path)

        for hparam_instance, experiment_spec in search_fn(hparams, reference_config):
            search_single(experiment_spec, hparam_instance, args)


if __name__ == "__main__":
    main()
