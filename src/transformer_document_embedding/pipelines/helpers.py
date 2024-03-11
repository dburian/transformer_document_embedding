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
    accuracy = (
        BinaryAccuracy(**metric_kwargs)
        if num_classes == 2
        else MulticlassAccuracy(
            num_classes=num_classes, average="micro", **metric_kwargs
        )
    )

    recall = (
        BinaryRecall(**metric_kwargs)
        if num_classes == 2
        else MulticlassRecall(num_classes=num_classes, average="micro", **metric_kwargs)
    )

    f1 = (
        BinaryF1Score(**metric_kwargs)
        if num_classes == 2
        else MulticlassF1Score(
            num_classes=num_classes, average="micro", **metric_kwargs
        )
    )

    precision = (
        BinaryPrecision(**metric_kwargs)
        if num_classes == 2
        else MulticlassPrecision(
            num_classes=num_classes, average="micro", **metric_kwargs
        )
    )
    return {
        "precision": precision,
        "accuracy": accuracy,
        "f1": f1,
        "recall": recall,
    }
