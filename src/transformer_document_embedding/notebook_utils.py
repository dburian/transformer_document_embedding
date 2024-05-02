from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Optional, TYPE_CHECKING
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_document_embedding.scripts.config_specs import EmbeddingModelSpec
from transformer_document_embedding.scripts.evaluate import find_config

from transformer_document_embedding.scripts.utils import load_yaml
import logging

if TYPE_CHECKING:
    from transformer_document_embedding.models.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)

boxplot_kwargs = dict(
    showmeans=True,
    meanprops={
        "markeredgecolor": "black",
        "markerfacecolor": "black",
        "marker": "^",
        "markeredgewidth": 1,
    },
)


def simplify_metric_name(metric_name: str) -> str:
    prepositions = ["binary_", "micro_"]
    for prep in prepositions:
        if metric_name.startswith(prep):
            return metric_name[len(prep) :]

    return metric_name


def load_validation_results(
    eval_dir: str,
    model_features: Optional[Callable[[str], dict[str, Any]]] = None,
) -> pd.DataFrame:
    model_evals = []
    for model_dir in os.scandir(eval_dir):
        try:
            model_eval_results = load_yaml(os.path.join(model_dir.path, "results.yaml"))
        except FileNotFoundError:
            logger.warn("Results for model '%s' not found. Skipping...", model_dir.name)
            continue

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


def seaborn_defaults() -> None:
    plt.rc("figure", figsize=(15, 8))
    sns.set_theme("paper", "whitegrid", "colorblind")


def uncompressed_tables() -> None:
    pd.set_option("display.max_columns", 1000)
    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.max_colwidth", 500)


def model_order(
    evals: pd.DataFrame, by: str = "score", of: str = "model"
) -> Iterable[str]:
    return evals.groupby(of, observed=False)[by].mean().sort_values().index


def load_model_save(path_to_weights: str) -> EmbeddingModel:
    config_path = find_config(path_to_weights)
    if config_path is None:
        raise ValueError(
            "Cannot find `config.yaml` file for model saved at '%s'. Skipping..."
            % path_to_weights,
        )

    model_config = EmbeddingModelSpec.from_dict(load_yaml(config_path)["model"])

    model = model_config.initialize()
    model.load_weights(path_to_weights)
    return model
