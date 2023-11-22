from __future__ import annotations
import argparse
from functools import partial
import logging
import pprint
from typing import Any, Optional, TYPE_CHECKING

from datasets import Dataset, disable_progress_bar, enable_progress_bar
import senteval
from transformer_document_embedding.models.experimental_model import ExperimentalModel
from transformer_document_embedding.experiments.config import ExperimentConfig
from transformer_document_embedding.experiments.result import save_csv_results
from transformer_document_embedding.scripts.args import add_common_args

from transformer_document_embedding.scripts.pipelines import InitializeModelAndTask
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
    print("Reducing")
    pprint.pprint(all_results)
    reduced_results = {}
    for task, task_results in all_results.items():
        if task.startswith("STS") and task != "STSBenchmark":
            print(f"task: {task}, task_results: {task_results}")
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
    config: ExperimentConfig,
    args: argparse.Namespace,
) -> None:
    model = config.get_model_type()(**config.values["model"].get("kwargs", {}))
    task = config.get_task_type()(**config.values["task"].get("kwargs", {}))

    if args.load_model_path is not None:
        logging.info("Loading model from %s.", args.load_model_path)
        model.load(args.load_model_path)

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
        model.train(
            DummyTask(ds),
            **config.values["model"].get("train_kwargs", {}),
        )

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


def evaluate(config: ExperimentConfig, args: argparse.Namespace) -> None:
    _, task = initialization_pipeline.run(config)
    assert isinstance(task, SentEval), "SentEval can be only run with `SentEval` task."

    prepare = partial(generic_prepare, config=config, args=args)

    logging.info("Evaluating: %s", ",".join(task.tasks))
    se = senteval.SE(task.params, batcher, prepare)

    results = se.eval(task.tasks)
    results = reduce_results(results)
    logging.info("Results: %s", pprint.pformat(results, indent=1))

    save_csv_results(results, config.experiment_path)


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    config = ExperimentConfig.from_yaml(
        args.config, args.output_base_path, name=args.name
    )

    evaluate(config, args)


class DummyTask(ExperimentalTask):
    def __init__(self, train_ds: Dataset) -> None:
        self._train_ds = train_ds

    @property
    def train(self):
        return self._train_ds

    @property
    def validation(self) -> Optional[Any]:
        return None

    @property
    def test(self):
        return Dataset.from_dict({"text": []})

    def evaluate(self, *_) -> dict[str, float]:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
