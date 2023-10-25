from __future__ import annotations


import os
import argparse
import pprint

import logging


from datasets import Dataset, DatasetDict, concatenate_datasets
import datasets.utils.logging as hf_logging
from transformer_document_embedding.experiments.config import ExperimentConfig
from transformer_document_embedding.scripts.args import add_common_args

from transformer_document_embedding.utils.evaluation import smart_unbatch


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
        "--splits",
        type=str,
        default="all",
        help="Name of splits separated by commas for which the embeddings should be"
        "generated. 'all' means to generate embeddings for all splits.",
    )

    parser.add_argument(
        "--embedding_col_name",
        type=str,
        default="embedding",
        help="Column name for embedding of an input example that will be added to"
        "the dataset.",
    )

    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="500MB",
        help="Maximum size of generated shards.",
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
        for embed in smart_unbatch(model.predict(split), 1):
            yield {args.embedding_col_name: embed}

    embeddings = DatasetDict()

    # TODO: We're relying on HF Task interface. Do we really need the basic
    # task? Should the basic task also have `splits` property?
    splits = args.splits.split(",") if args.splits != "all" else task.splits.keys()

    hf_logging.disable_progress_bar()
    for split_name in splits:
        logging.info("Generating embeddings for split '%s'", split_name)
        split = task.splits.get(split_name, None)
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

    embed_dataset_path = os.path.join(config.experiment_path, "embeddings")
    logging.info("Saving the embeddings dataset to '%s'", embed_dataset_path)
    embeddings.save_to_disk(embed_dataset_path, max_shard_size=args.max_shard_size)


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
