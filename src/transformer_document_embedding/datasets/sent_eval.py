from __future__ import annotations
from typing import Any, Union

from datasets import DatasetDict

from transformer_document_embedding.datasets.document_dataset import (
    DocumentDataset,
    EvaluationKind,
)

import logging

PREDEFINED_PARAMS = {
    "debug": {
        # Predict all embeddings at once
        "batch_size": 1000000,
        "usepytorch": True,
        "kfold": 5,
        "classifier": {
            "nhid": 0,
            "optim": "rmsprop",
            "batch_size": 128,
            "tenacity": 3,
            "epoch_size": 2,
        },
    },
    "default": {
        # Predict all embeddings at once
        "batch_size": 1000000,
        "usepytorch": True,
        "kfold": 10,
        "classifier": {
            "nhid": 0,
            "optim": "adam",
            "batch_size": 64,
            "tenacity": 5,
            "epoch_size": 4,
        },
    },
}

AVAILABLE_TASKS = {
    "CR",
    "MR",
    "MPQA",
    "SUBJ",
    "SST2",
    "SST5",
    "TREC",
    "MRPC",
    "SNLI",
    "SICKEntailment",
    "SICKRelatedness",
    "STSBenchmark",
    "ImageCaptionRetrieval",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "Length",
    "WordContent",
    "Depth",
    "TopConstituents",
    "BigramShift",
    "Tense",
    "SubjNumber",
    "ObjNumber",
    "OddManOut",
    "CoordinationInversion",
}

logger = logging.getLogger(__name__)


class SentEval(DocumentDataset):
    def __init__(
        self,
        path: str,
        tasks: list[str],
        params: Union[dict[str, Any], str],
    ) -> None:
        super().__init__(
            data_size_limit=None,
            add_ids=False,
            validation_source_fraction=None,
            validation_source=None,
        )

        set_of_tasks = set(tasks)
        self.tasks = list(set_of_tasks & AVAILABLE_TASKS)
        duplicates = len(tasks) - len(set_of_tasks)

        if duplicates > 0:
            logger.warn("List of sent-eval tasks contained %d duplicates.", duplicates)

        if len(self.tasks) != len(tasks) - duplicates:
            unknown_tasks = set_of_tasks - AVAILABLE_TASKS
            logger.warn(
                "List of sent-eval tasks contained unknown tasks: '%s'",
                ",".join(unknown_tasks),
            )

        self.params = PREDEFINED_PARAMS[params] if isinstance(params, str) else params
        self.params["task_path"] = path

    @property
    def evaluation_kind(self) -> EvaluationKind:
        return EvaluationKind.SENT_EVAL

    def _retrieve_dataset(self) -> DatasetDict:
        # Empty dataset in case somebody would like to train on this
        return DatasetDict()
