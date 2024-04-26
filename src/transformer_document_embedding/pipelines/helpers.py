from __future__ import annotations
from typing import Iterable, Iterator, TYPE_CHECKING

from torcheval.metrics import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    Metric,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
)

if TYPE_CHECKING:
    import torch


def classification_metrics(num_classes: int, **metric_kwargs) -> dict[str, Metric]:
    if num_classes == 2:
        return {
            "binary_precision": BinaryPrecision(**metric_kwargs),
            "binary_accuracy": BinaryAccuracy(**metric_kwargs),
            "binary_f1": BinaryF1Score(**metric_kwargs),
            "binary_recall": BinaryRecall(**metric_kwargs),
        }

    micro_accuracy = MulticlassAccuracy(
        num_classes=num_classes, average="micro", **metric_kwargs
    )

    macro_accuracy = MulticlassAccuracy(
        num_classes=num_classes, average="macro", **metric_kwargs
    )

    # Currently not used as there is a bug. Waiting for PR (#166) to be merged.
    # macro_recall = MulticlassRecall(
    #     num_classes=num_classes, average="macro", **metric_kwargs
    # )

    macro_precision = MulticlassPrecision(
        num_classes=num_classes, average="macro", **metric_kwargs
    )

    macro_f1 = MulticlassF1Score(
        num_classes=num_classes, average="macro", **metric_kwargs
    )

    return {
        "micro_accuracy": micro_accuracy,
        "macro_accuracy": macro_accuracy,
        "macro_precision": macro_precision,
        "macro_f1": macro_f1,
        # "macro_recall": macro_recall,
    }


def smart_unbatch(
    iterable: Iterable[torch.Tensor],
    single_dim: int,
) -> Iterator[torch.Tensor]:
    for batch in iterable:
        if len(batch.shape) > single_dim:
            yield from smart_unbatch(batch, single_dim)
        else:
            yield batch
