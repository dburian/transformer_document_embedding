from __future__ import annotations

from torcheval.metrics import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    Metric,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


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

    macro_recall = MulticlassRecall(
        num_classes=num_classes, average="macro", **metric_kwargs
    )

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
        "macro_recall": macro_recall,
    }
