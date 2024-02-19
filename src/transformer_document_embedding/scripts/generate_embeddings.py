from __future__ import annotations


import os
import argparse

import logging
import pprint

from datasets import Dataset, DatasetDict, concatenate_datasets
import datasets.utils.logging as hf_logging
from transformer_document_embedding.datasets import col
from transformer_document_embedding.scripts.common import load_train_save
from transformer_document_embedding.scripts.config_specs import ExperimentSpec
from transformer_document_embedding.scripts.utils import (
    load_yaml,
    save_config,
)

import coolname

from transformer_document_embedding.pipelines.classification_eval import smart_unbatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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
        "--splits",
        type=str,
        default="all",
        help="Name of splits separated by commas for which the embeddings should be"
        "generated. 'all' means to generate embeddings for all splits.",
    )

    parser.add_argument(
        "--embedding_col_name",
        type=str,
        default=col.EMBEDDING,
        help="Column name for embedding of an input example that will be added to"
        "the dataset.",
    )

    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="1GB",
        help="Maximum size of generated shards.",
    )

    return parser.parse_args()


def generate_embeddings(
    config: ExperimentSpec,
    args: argparse.Namespace,
) -> None:
    exp_path = os.path.join(
        args.output_base_path, config.dataset.module, config.model.module, args.name
    )
    os.makedirs(exp_path, exist_ok=True)
    logging.info(
        "Generating embeddings '%s' with config:\n%s",
        args.name,
        pprint.pformat(config, indent=1),
    )
    save_config(config, exp_path)

    model, _, dataset = load_train_save(
        config,
        save_trained_model=args.save_trained_model,
        save_trained_head=args.save_trained_head,
        load_model_weights_path=args.load_model_weights_path,
        load_head_weights_path=args.load_head_weights_path,
        exp_path=exp_path,
    )

    # The gymnastics with generators, new dataset and concatenation is not
    # straightforward, but:
    # - `map` iterates over slices, which are not compatible with datasets so:
    #   `.map(lambda x: model.predict(x))` won't work since `predict` accepts
    #   only `Dataset`
    # - `map` can iterate over whole splits, but we risk that the split won't fit
    #   into memory
    # - `add_column` can accept generator, but it is not documented and thus it
    #   feels like a hacky solution
    def embed_generator(split: Dataset):
        for embed in smart_unbatch(model.predict_embeddings(split), 1):
            yield {args.embedding_col_name: embed}

    embeddings = DatasetDict()

    splits = args.splits.split(",") if args.splits != "all" else dataset.splits.keys()

    hf_logging.disable_progress_bar()
    for split_name in splits:
        logging.info("Generating embeddings for split '%s'", split_name)
        split = dataset.splits.get(split_name, None)
        if split is None:
            logging.warn(
                "Split '%s' doesn't exist for this dataset. Skipping...", split_name
            )
            continue

        split_embeddings = Dataset.from_generator(
            embed_generator, gen_kwargs={"split": split}
        )
        embeddings[split_name] = concatenate_datasets([split, split_embeddings], axis=1)
    hf_logging.enable_progress_bar()

    embed_dataset_path = os.path.join(exp_path, "embeddings")
    logging.info("Saving the embeddings dataset to '%s'", embed_dataset_path)
    embeddings.save_to_disk(embed_dataset_path, max_shard_size=args.max_shard_size)


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    config = ExperimentSpec.from_dict(load_yaml(args.config))
    generate_embeddings(config, args)


if __name__ == "__main__":
    main()
