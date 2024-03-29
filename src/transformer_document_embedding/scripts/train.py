"""a __doc__"""
from __future__ import annotations
import os
import logging
import pprint

import coolname
from datasets import disable_caching
from transformer_document_embedding.scripts.common import evaluate, load_train_save

from transformer_document_embedding.scripts.utils import (
    load_yaml,
    save_config,
)
import argparse
from transformer_document_embedding.scripts.config_specs import ExperimentSpec


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
        "--output_base_path",
        type=str,
        default="./results",
        help=(
            "Path to directory containing all experiment results. Default:"
            # f" '{output_base_path}'."
        ),
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

    parser.add_argument(
        "--disable_hf_caching",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to disable HuggingFace datasets caching.",
    )

    return parser.parse_args()


def train(config: ExperimentSpec, args: argparse.Namespace) -> dict[str, float]:
    exp_path = os.path.join(
        args.output_base_path, config.dataset.module, config.model.module, args.name
    )
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
        save_trained_model=args.save_trained_model,
        save_trained_head=args.save_trained_head,
        exp_path=exp_path,
    )

    return evaluate(
        model, head, dataset, exp_path, evaluation_kwargs=config.evaluation_kwargs
    )


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    if args.disable_hf_caching:
        disable_caching()

    config = ExperimentSpec.from_dict(load_yaml(args.config))
    train(config, args)


if __name__ == "__main__":
    main()
