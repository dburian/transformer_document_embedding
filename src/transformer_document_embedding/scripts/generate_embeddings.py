from __future__ import annotations
from copy import deepcopy


import os
import argparse

import logging
import pprint

from datasets import Dataset, DatasetDict, concatenate_datasets
import datasets.utils.logging as hf_logging
from transformer_document_embedding.datasets import col
from transformer_document_embedding.scripts.evaluate import find_config
from transformer_document_embedding.scripts.utils import (
    load_yaml,
    save_config,
)

import coolname

from transformer_document_embedding.pipelines.classification_eval import smart_unbatch
from transformer_document_embedding.scripts.config_specs import ExperimentSpec


logger = logging.getLogger(__name__)


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
        "-m",
        "--model",
        type=str,
        action="extend",
        nargs="+",
        help="Path to model's weights.",
    )

    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="_".join(coolname.generate(2)),
        help="Name of the experiment. If no name is given, one is generated.",
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
        "--embedding_prediction_batch_size",
        type=int,
        default=8,
        help="Batch size when predicting embeddings.",
    )

    return parser.parse_args()


def generate_embeddings(
    config: ExperimentSpec,
    model_weights_path: str,
    args: argparse.Namespace,
) -> None:
    exp_path = os.path.join(
        args.output_base_path, config.dataset.module, config.model.module, args.name
    )
    os.makedirs(exp_path, exist_ok=True)
    logger.info(
        "Generating embeddings '%s' with config:\n%s",
        args.name,
        pprint.pformat(config, indent=1),
    )
    save_config(config, exp_path)

    dataset = config.dataset.initialize()
    model = config.model.initialize()

    logger.info("Loading model weights from '%s'.", model_weights_path)
    model.load_weights(model_weights_path)

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
        for embed in smart_unbatch(
            model.predict_embeddings(
                split,
                batch_size=args.embedding_prediction_batch_size,
            ),
            1,
        ):
            yield {args.embedding_col_name: embed}

    embeddings = DatasetDict()

    splits = args.splits.split(",") if args.splits != "all" else dataset.splits.keys()

    hf_logging.disable_progress_bar()
    for split_name in splits:
        logger.info("Generating embeddings for split '%s'", split_name)
        split = dataset.splits.get(split_name, None)
        if split is None:
            logger.warn(
                "Split '%s' doesn't exist for this dataset. Skipping...", split_name
            )
            continue

        split_embeddings = Dataset.from_generator(
            embed_generator, gen_kwargs={"split": split}
        )
        embeddings[split_name] = concatenate_datasets([split, split_embeddings], axis=1)
    hf_logging.enable_progress_bar()

    embed_dataset_path = os.path.join(exp_path, "embeddings")
    logger.info("Saving the embeddings dataset to '%s'", embed_dataset_path)
    embeddings.save_to_disk(embed_dataset_path)


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    config = load_yaml(args.config)
    config.pop("head", None)
    config.pop("train_pipeline", None)
    for model_weights_path in args.model:
        model_conf_path = find_config(model_weights_path)
        if model_conf_path is None:
            logger.error(
                "Cannot find config for a model's weights path '%s'", model_weights_path
            )
            continue
        model_conf = load_yaml(model_conf_path)
        instance_config = deepcopy(config)
        instance_config["model"] = model_conf["model"]

        generate_embeddings(
            ExperimentSpec.from_dict(instance_config), model_weights_path, args
        )


if __name__ == "__main__":
    main()
