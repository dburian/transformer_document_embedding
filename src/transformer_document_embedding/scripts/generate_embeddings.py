from __future__ import annotations


import os
import argparse

import logging


from datasets import Dataset, DatasetDict, concatenate_datasets
import datasets.utils.logging as hf_logging
from transformer_document_embedding.scripts.config_specs import ExperimentSpec
from transformer_document_embedding.scripts.pipelines import (
    InitializeModelAndTask,
    TrainingPipeline,
    add_common_args,
)
from transformer_document_embedding.scripts.utils import (
    load_yaml,
)
from transformer_document_embedding.utils.evaluation import smart_unbatch

training_pipeline = TrainingPipeline()
initialization_pipeline = InitializeModelAndTask()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    training_pipeline.add_args(parser)
    initialization_pipeline.add_args(parser)

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
        default="1GB",
        help="Maximum size of generated shards.",
    )

    add_common_args(parser)

    return parser.parse_args()


def generate_embeddings(
    config: ExperimentSpec,
    args: argparse.Namespace,
) -> None:
    exp_path = os.path.join(
        args.output_base_path,
        config.task.module,
        config.model.module,
        args.name,
    )
    os.makedirs(exp_path, exist_ok=True)

    model, task = initialization_pipeline.run(args.name, exp_path, config)

    training_pipeline.run(args, model, task, exp_path, config)

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
    splits = args.splits.split(",") if args.splits != "all" else task.split_names.keys()

    hf_logging.disable_progress_bar()
    for split_name in splits:
        logging.info("Generating embeddings for split '%s'", split_name)
        split = task.split_names.get(split_name, None)
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
