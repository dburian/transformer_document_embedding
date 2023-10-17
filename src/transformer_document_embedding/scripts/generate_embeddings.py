from __future__ import annotations

from typing import TYPE_CHECKING

import os
import argparse
import pprint

import logging

from transformer_document_embedding.experiments.config import ExperimentConfig
from transformer_document_embedding.scripts.args import add_common_args
from transformer_document_embedding.utils.evaluation import smart_unbatch

if TYPE_CHECKING:
    from typing import Iterable
    import numpy as np
    from datasets import Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to train the model before generating embeddings.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split for which the embeddings should be generated.",
    )

    add_common_args(parser)
    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help=(
            "Path from which to load the fitted model instead of fitting it (which is"
            " the default behaviour)."
        ),
    )

    parser.add_argument(
        "--save_trained",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to save trained model.",
    )
    parser.add_argument(
        "--embed_fragment_length",
        type=int,
        default=10e5,
        help="Number of embeddings stored in one file.",
    )

    return parser.parse_args()


def save_embeddings(
    embed_iter: Iterable[np.ndarray],
    split: Dataset,
    args: argparse.Namespace,
    embedding_path: str,
) -> None:
    import pandas as pd
    from tqdm.auto import tqdm

    fragment_counter = 0
    df = []

    def save_fragment(df: pd.DataFrame) -> None:
        fragment_path = os.path.join(embedding_path, f"fragment_{fragment_counter}.csv")
        pd.DataFrame(df).to_csv(fragment_path, index=False)

    id_embed_iter = zip(split["id"], smart_unbatch(embed_iter, 1), strict=True)
    for id, embed in tqdm(
        id_embed_iter, desc="Generating embeddings", total=len(split)
    ):
        df.append({"id": id, "embed": embed})
        if len(df) == args.embed_fragment_length:
            save_fragment(pd.DataFrame(df))

            df = []
            fragment_counter += 1

    if len(df) > 0:
        save_fragment(pd.DataFrame(df))


def generate_embeddings(
    config: ExperimentConfig,
    args: argparse.Namespace,
) -> None:
    logging.info(
        "Starting experiment with config:\n%s",
        pprint.pformat(config.values, indent=1),
    )
    model = config.get_model_type()(**config.values["model"].get("kwargs", {}))
    task = config.get_task_type()(**config.values["task"].get("kwargs", {}))

    # TODO: Common training pipeline (in evaluate)
    if args.load_model_path is not None:
        logging.info("Loading model from %s.", args.load_model_path)
        model.load(args.load_model_path)

    if args.train:
        logging.info("Training model...")

        model.train(
            task,
            log_dir=config.experiment_path,
            model_dir=config.model_path,
            **config.values["model"].get("train_kwargs", {}),
        )
        logging.info("Training done.")

        if args.save_trained:
            trained_path = os.path.join(config.model_path, "trained")
            logging.info(
                "Saving trained model to %s.",
                trained_path,
            )
            os.makedirs(trained_path, exist_ok=True)
            model.save(trained_path)

    embed_path = os.path.join(config.experiment_path, "embeddings")
    os.makedirs(embed_path, exist_ok=True)
    logging.info("Generating embeddings for '%s' to '%s'", args.split, embed_path)
    split = getattr(task, args.split)
    embed_iter = model.predict(split)
    save_embeddings(
        embed_iter=embed_iter,
        split=split,
        args=args,
        embedding_path=embed_path,
    )


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    for config_file in args.config:
        exp_config = ExperimentConfig.from_yaml(config_file, args.output_base_path)

        generate_embeddings(config=exp_config, args=args)


if __name__ == "__main__":
    main()
