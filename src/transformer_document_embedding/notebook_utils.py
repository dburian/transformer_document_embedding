from __future__ import annotations

import os
from typing import Any, Callable, Optional
import pandas as pd

from transformer_document_embedding.scripts.utils import load_yaml


def simplify_metric_name(metric_name: str) -> str:
    prepositions = ["binary_", "micro_"]
    for prep in prepositions:
        if metric_name.startswith(prep):
            return metric_name[len(prep) :]

    return metric_name


def load_validation_results(
    eval_dir: str,
    model_features: Optional[Callable[[str], dict[str, Any]]],
) -> pd.DataFrame:
    model_evals = []
    for model_dir in os.scandir(eval_dir):
        model_eval_results = load_yaml(os.path.join(model_dir.path, "results.yaml"))
        for task, metrics in model_eval_results.items():
            for metric_name, score in metrics.items():
                if metric_name.endswith("_std"):
                    # We handle st. deviation metric scores together with mean metric
                    continue

                std = 0
                if metric_name.endswith("_mean"):
                    metric_name = metric_name[: -len("_mean")]
                    std = metrics[f"{metric_name}_std"]

                model_evals.append(
                    {
                        "model": model_dir.name,
                        "task": task,
                        "metric": simplify_metric_name(metric_name),
                        "full_metric": metric_name,
                        "score": score,
                        "std": std,
                    }
                )

    evals = pd.DataFrame(model_evals)
    if model_features is not None:
        evals = pd.concat(
            [evals, pd.DataFrame(map(model_features, evals["model"]))], axis=1
        )

    # Task type
    evals["task_type"] = "classification"
    evals.loc[
        evals["task"].isin(["sims_games", "sims_wines"]), "task_type"
    ] = "retrieval"

    return evals


def add_normalized_score(
    evals: pd.DataFrame, extra_idx_cols: Optional[list[str]] = None
) -> pd.DataFrame:
    metric_maximums = evals.groupby(["task", "metric"])["score"].max()

    extra_idx_cols = [] if extra_idx_cols is None else extra_idx_cols
    by_metric = evals.set_index(["task", "metric", "model"] + extra_idx_cols)

    by_metric["normalized_score"] = by_metric["score"] / metric_maximums
    return by_metric.reset_index()
