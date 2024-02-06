from __future__ import annotations
import argparse
import os
from functools import partial
import logging
import pprint
from typing import Any, TYPE_CHECKING

from datasets import Dataset, disable_progress_bar, enable_progress_bar
import senteval
from transformer_document_embedding.models.experimental_model import ExperimentalModel

from transformer_document_embedding.scripts.pipelines import (
    InitializeModelAndTask,
    add_common_args,
)
from transformer_document_embedding.scripts.utils import (
    ExperimentSpec,
    init_type,
    load_yaml,
    save_config,
    save_results,
)
from transformer_document_embedding.tasks.experimental_task import ExperimentalTask


from transformer_document_embedding.tasks.sent_eval import SentEval
import numpy as np

from transformer_document_embedding.utils.evaluation import smart_unbatch

if TYPE_CHECKING:
    from senteval.utils import dotdict

initialization_pipeline = InitializeModelAndTask()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    add_common_args(parser)
    initialization_pipeline.add_args(parser)

    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="Path from which to load the fitted model before training.",
    )

    parser.add_argument(
        "--save_trained",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to save trained model.",
    )

    parser.add_argument(
        "--train",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to train the model before generating embeddings.",
    )

    return parser.parse_args()


def words_to_dataset(word_batch: list[list[str]]) -> Dataset:
    sentences = [" ".join(words) if len(words) > 0 else "." for words in word_batch]
    return Dataset.from_dict({"text": sentences})


def reduce_results(all_results: dict[str, Any]) -> dict[str, float]:
    """Leaves only comparative results."""
    reduced_results = {}
    for task, task_results in all_results.items():
        if task.startswith("STS") and task != "STSBenchmark":
            reduced_results[f"{task}_spearman"] = task_results["all"]["spearman"][
                "wmean"
            ]
            reduced_results[f"{task}_pearson"] = task_results["all"]["pearson"]["wmean"]
        elif task in ["STSBenchmark", "SICKRelatedness", "SICKEntailment"]:
            reduced_results[f"{task}_spearman"] = task_results["spearman"]
            reduced_results[f"{task}_pearson"] = task_results["pearson"]
        else:
            # All classification tasks
            reduced_results[f"{task}_accuracy"] = task_results["acc"]

    return reduced_results


def generic_prepare(
    params: dotdict,
    samples: list[list[str]],
    *,
    config: ExperimentSpec,
    args: argparse.Namespace,
) -> None:
    model: ExperimentalModel = init_type(config.model)
    task: SentEval = init_type(config.task)

    if args.load_model_path is not None:
        logging.info("Loading model from %s.", args.load_model_path)
        model.load_weights(args.load_model_path)

    ds = words_to_dataset(samples)
    if task.add_ids:
        disable_progress_bar()
        ds = ds.map(
            lambda _, ids: {"id": ids},
            with_indices=True,
            batched=True,
            keep_in_memory=True,
        )
        enable_progress_bar()
        params.sent_to_id = {sent["text"]: sent["id"] for sent in ds}

    if args.train:
        model.train(DummyTask(ds), **config.model.train_kwargs)

    params.model = model


def batcher(params: dotdict, batch: list[list[str]]) -> np.ndarray:
    ds = words_to_dataset(batch)
    if params.sent_to_id is not None:
        sent_to_id = params.sent_to_id
        disable_progress_bar()
        ds = ds.map(
            lambda sent: {"id": sent_to_id[sent["text"]]},
            keep_in_memory=True,
        )
        enable_progress_bar()

    assert isinstance(params.model, ExperimentalModel)
    pred_batches = params.model.predict(ds)
    return np.vstack(list(smart_unbatch(pred_batches, 1)))


def evaluate(config: ExperimentSpec, args: argparse.Namespace) -> None:
    exp_path = os.path.join(
        args.output_base_path, config.task.module, config.model.module, args.name
    )
    os.makedirs(exp_path, exist_ok=True)
    logging.info(
        "Starting experiment '%s' with config:\n%s",
        args.name,
        pprint.pformat(config, indent=1),
    )
    save_config(config, exp_path)

    task = init_type(config.task)
    assert isinstance(task, SentEval), "SentEval can be only run with `SentEval` task."

    prepare = partial(generic_prepare, config=config, args=args)

    logging.info("Evaluating: %s", ",".join(task.tasks))
    se = senteval.SE(task.params, batcher, prepare)

    results = se.eval(task.tasks)
    results = reduce_results(results)
    logging.info("Results: %s", pprint.pformat(results, indent=1))

    save_results(results, exp_path)


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    config = ExperimentSpec.from_dict(load_yaml(args.config))

    evaluate(config, args)


class DummyTask(ExperimentalTask):
    def __init__(self, train_ds: Dataset) -> None:
        self._splits = {"train": train_ds}

    @property
    def splits(self) -> dict[str, Any]:
        return self._splits

    def evaluate(self, *_) -> dict[str, float]:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
