# %%
import os
import pandas as pd
from transformer_document_embedding.scripts.utils import load_yaml
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# %%
plt.rc("figure", figsize=(15, 8))
sns.set_context("paper")
sns.set_style("whitegrid")
# deep, muted, bright, pastel, dark, colorblind
sns.color_palette("bright")
sns.set_palette(sns.color_palette("colorblind"))


# %%
pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_colwidth", 500)

# %%
from typing import Any, Callable, NamedTuple


def struct_features(model_str: str) -> dict[str, Any]:
    id_segments = model_str.split("-")

    FeatureReader = NamedTuple(
        "Feature", [("id", str), ("label", str), ("type", Callable[[str], Any])]
    )

    readers = [
        FeatureReader(
            "h.k.s_h_k.l_t",
            "loss_type",
            lambda lt: "max_marginals" if "max_marginals" in lt else lt,
        ),
        FeatureReader(
            "h.k.s_h_k.l_t",
            "mm_loss_type",
            lambda lt: lt[len("max_marginals_") :] if "max_marginals" in lt else np.nan,
        ),
        FeatureReader("h.k.s_h_k.m_m_l", "mm_lam", float),
    ]

    features = {"loss_type": model_str}
    for seg in id_segments:
        if "=" not in seg:
            continue
        feat_id, value = seg.split("=")
        for feat_reader in readers:
            if feat_reader.id != feat_id:
                continue

            features[feat_reader.label] = feat_reader.type(value)

    return features


# %%
metric_renames = {
    "micro_accuracy": "accuracy",
    "binary_accuracy": "accuracy",
    "macro_f1": "f1",
    "macro_precision": "precision",
    "macro_recall": "recall",
    "binary_recall": "recall",
    "binary_precision": "precision",
    "binary_f1": "f1",
}


def load_eval_dir(path):
    model_evals = []
    for model_dir in os.scandir(path):
        model_eval_results = load_yaml(os.path.join(model_dir.path, "results.yaml"))
        model_results = {
            (task, metric_renames.get(metric, metric)): score
            for task, metrics in model_eval_results.items()
            for metric, score in metrics.items()
        }
        model_results["model"] = model_dir.name
        model_evals.append(model_results)
    evals = pd.DataFrame(model_evals)

    # Handling CV results
    tmp_evals = evals.copy()
    imdb_cols = [col for col in tmp_evals.columns if col[0] == "imdb"]
    imdb_mean_cols = [col for col in imdb_cols if col[1].endswith("mean")]
    for _, mean_col in imdb_mean_cols:
        metric = mean_col[mean_col.find("_") + 1 : -len("_mean")]
        tmp_evals[("imdb", metric)] = list(
            zip(
                tmp_evals[("imdb", mean_col)],
                tmp_evals[("imdb", f"binary_{metric}_std")],
                strict=True,
            )
        )
    tmp_evals.drop(columns=imdb_cols, inplace=True)

    evals_long = tmp_evals.melt(id_vars=["model"])
    evals_long["task"] = evals_long["variable"].map(lambda tup: tup[0])
    evals_long["metric"] = evals_long["variable"].map(lambda tup: tup[1])
    evals_long["std"] = evals_long["value"].map(
        lambda val: 0 if not isinstance(val, tuple) else val[1]
    )
    evals_long["value"] = evals_long["value"].map(
        lambda val: val if not isinstance(val, tuple) else val[0]
    )
    evals_long.drop(columns=["variable"], inplace=True)

    # Normalized value
    metric_limits = evals_long.groupby(["task", "metric"])["value"].agg(["min", "max"])
    by_metric = evals_long.set_index(["task", "metric", "model"])
    by_metric["normalized_value"] = (by_metric["value"] - metric_limits["min"]) / (
        metric_limits["max"] - metric_limits["min"]
    )
    evals_long = by_metric.reset_index()

    # Task type
    evals_long["task_type"] = "classification"
    evals_long.loc[
        evals_long["task"].isin(["sims_games", "sims_wines"]), "task_type"
    ] = "retrieval"

    return evals_long


# %%
def add_model_features(df, features_func):
    features = pd.DataFrame(map(features_func, df["model"]))
    return pd.concat((features, df), axis=1)


# %%
evals = load_eval_dir("../evaluations/old_structural_eval/")
evals = add_model_features(evals, struct_features)

# %%
cls_evals = load_eval_dir("../evaluations/cls_structural_eval/")
cls_evals = add_model_features(cls_evals, struct_features)

# %%
evals.shape

# %%
evals[evals["value"].isna()]

# %% [markdown]
# # Exploration

# %% [markdown]
# ## Retrieval tasks

# %%
ret_evals = evals[evals["task_type"] == "retrieval"]


# %%
def permutation_testing(df, metric_blacklist=None, less_is_better=None):
    if less_is_better is None:
        less_is_better = ["mean_percentile_rank"]
    if metric_blacklist is None:
        metric_blacklist = ["auprc"]

    df = df.set_index(["task", "metric"]).sort_index()
    results = []
    for (task, metric), row in df.iterrows():
        if metric in metric_blacklist:
            continue

        for _, other_row in df.loc[(task, metric)].iterrows():
            if other_row["model"] == row["model"]:
                continue

            other_value = other_row["normalized_value"]
            value = row["normalized_value"]
            won_by = (
                (other_value - value)
                if metric in less_is_better
                else (value - other_value)
            )
            results.append(
                {
                    "other_model": other_row["model"],
                    "task": task,
                    "metric": metric,
                    "won_by": won_by,
                    "won": won_by > 0,
                    **row,
                }
            )

    return pd.DataFrame(results)


# %%
sns.barplot(
    ret_evals,
    y="model",
    x="normalized_value",
    hue="metric",
    errorbar=("pi", 100),
    order=ret_evals.groupby("model")["normalized_value"].mean().sort_values().index,
)

# %%
matches = permutation_testing(ret_evals)
res = matches.groupby("model")[["won", "won_by"]].mean().reset_index()

# %%
sns.barplot(res, y="model", x="won_by", order=res.sort_values("won_by")["model"])

# %%
sns.barplot(res, y="model", x="won", order=res.sort_values("won")["model"])

# %% [markdown]
# # Presentable

# %%
pres_evals = evals[evals["metric"] == "accuracy"]

# %%
cls_pres_evals = cls_evals[cls_evals["metric"] == "accuracy"]

# %%
sns.barplot(
    pres_evals,
    y="model",
    x="value",
    order=pres_evals.groupby("model")["value"].mean().sort_values().index,
)

# %%
sns.barplot(
    cls_pres_evals,
    y="model",
    x="value",
    order=cls_pres_evals.groupby("model")["value"].mean().sort_values().index,
)

# %%
sns.barplot(
    cls_pres_evals,
    y="model",
    x="normalized_value",
    order=cls_pres_evals.groupby("model")["normalized_value"]
    .mean()
    .sort_values()
    .index,
)

# %%
tmp = pd.concat(
    [
        pres_evals[
            pres_evals["model"]
            == "h.k.s_h_k.l_t=max_marginals_cos_dist-h.k.s_h_k.m_m_l=1"
        ],
        cls_pres_evals[
            cls_pres_evals["model"].isin(
                [
                    "h.k.s_h_k.l_t=max_marginals_mse-h.k.s_h_k.m_m_l=1",
                    "h.k.s_h_k.l_t=cos_dist",
                    "h.k.s_h_k.l_t=max_marginals_cos_dist-h.k.s_h_k.m_m_l=0.1",
                ]
            )
        ],
    ]
)

sns.barplot(tmp, x="task", y="value", hue="model")

# %% [markdown]
# ## Best max-marginals

# %%
mmarginals_evals = pres_evals[pres_evals["loss_type"] == "max_marginals"]

# %%
mmarginals_evals[
    (mmarginals_evals["mm_loss_type"] == "mse") & (mmarginals_evals["mm_lam"] == 1.5)
]

# %%
sns.barplot(
    mmarginals_evals,
    y="mm_loss_type",
    hue="mm_lam",
    x="normalized_value",
    errorbar=("pi", 50),
)

# %%
sns.barplot(
    mmarginals_evals[mmarginals_evals["mm_lam"] == 1.0],
    y="mm_loss_type",
    hue="task",
    x="normalized_value",
    errorbar=("pi", 50),
)

# %%
sns.barplot(
    mmarginals_evals,
    y="mm_loss_type",
    hue="task",
    x="normalized_value",
    errorbar=("pi", 50),
)

# %% [markdown]
# ## Basic + best max-marginals

# %%
basics = pres_evals[pres_evals["mm_lam"].isna()]
best_mm = mmarginals_evals[
    (mmarginals_evals["mm_loss_type"] == "cos_dist")
    & (mmarginals_evals["mm_lam"] == 1.0)
].copy()

# %%
best_mm["loss_type"] += (
    "_" + best_mm["mm_loss_type"] + "_" + best_mm["mm_lam"].astype(str) + "lam"
)

# %%
best_evals = pd.concat([basics, best_mm])
best_evals = best_evals.drop(columns=["mm_loss_type", "mm_lam"])

# %%
best_evals["model"].unique()

# %%
sns.barplot(
    best_evals,
    y="loss_type",
    x="normalized_value",
    order=best_evals.groupby("loss_type")["normalized_value"]
    .mean()
    .sort_values()
    .index,
    errorbar=("pi", 50),
)

# %%
sns.barplot(
    best_evals,
    y="loss_type",
    x="value",
    order=best_evals.groupby("loss_type")["value"].mean().sort_values().index,
    errorbar=("pi", 50),
)

# %%
best_evals.groupby("loss_type")["value"].agg(["mean", "std"]).sort_values("mean")

# %%
sns.barplot(
    best_evals,
    y="loss_type",
    x="normalized_value",
    hue="task",
    order=best_evals.groupby("loss_type")["normalized_value"]
    .mean()
    .sort_values()
    .index,
)

# %%
sns.barplot(
    best_evals,
    y="loss_type",
    x="value",
    hue="task",
    order=best_evals.groupby("loss_type")["value"].mean().sort_values().index,
)
