from __future__ import annotations

from torcheval.metrics import (
    Metric,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


def classification_metrics(num_classes: int, **metric_kwargs) -> dict[str, Metric]:
    accuracy = MulticlassAccuracy(
        num_classes=num_classes, average="micro", **metric_kwargs
    )

    recall = MulticlassRecall(num_classes=num_classes, average="micro", **metric_kwargs)

    precision = MulticlassPrecision(
        num_classes=num_classes, average="micro", **metric_kwargs
    )

    f1 = MulticlassF1Score(num_classes=num_classes, average="micro", **metric_kwargs)

    return {
        "precision": precision,
        "accuracy": accuracy,
        "f1": f1,
        "recall": recall,
    }
