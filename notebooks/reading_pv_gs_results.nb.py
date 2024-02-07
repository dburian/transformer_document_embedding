# %% [markdown]
# # Reading results from Paragraph Vector (PV) grid searches

# %%
import os
import pandas as pd
from transformer_document_embedding.scripts.utils import load_yaml

# %%
pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 1000)

# %%
BASE_DIR = "../hp_searches/tmp_pv_results/"


def get_model_type(gs_dir: str) -> str:
    """gs_dir is of format 'pv_<task_name>_<model_type>_gs'"""
    return gs_dir.split("_")[2]


def get_task_name(gs_dir: str) -> str:
    """gs_dir is of format 'pv_<task_name>_<model_type>_gs'"""
    return gs_dir.split("_")[1]


exp_props_names = {
    "m.k.p_p": "pre_processing",
    "m.k.d_k.v_s": "dim",
    "m.k.d_k.m_c": "min_count",
}


def get_exp_props(exp_dir: str) -> dict[str, str]:
    props = {}
    for prop_id, value in map(lambda prop: prop.split("="), exp_dir.split("-")):
        prop_name = exp_props_names[prop_id]
        if prop_name == "min_count" and value != "2":
            value = "1%"
        props[prop_name] = value

    return props


# %%
results_df = []

# %%
for gs_dir in os.listdir(BASE_DIR):
    task_name = get_task_name(gs_dir)
    model_type = get_model_type(gs_dir)

    for exp_dir in os.listdir(os.path.join(BASE_DIR, gs_dir)):
        results = load_yaml(os.path.join(BASE_DIR, gs_dir, exp_dir, "results.yaml"))
        results_df.append(
            {
                "model": model_type,
                **get_exp_props(exp_dir),
                **{(task_name, key): float(res) for key, res in results.items()},
            }
        )

# %%
results_df = (
    pd.DataFrame(results_df)
    .set_index(["model", "pre_processing", "dim", "min_count"])
    .groupby(level=list(range(4)))
    .first()
)

# %%
results_df.columns = pd.MultiIndex.from_tuples(results_df.columns)

# %%
results_df

# %%
agg_results_classification = (
    results_df.drop(columns=["wines", "games"], level=0)
    .T.groupby(level=1)
    .agg(["mean", "median", "std"])
    .T.unstack()
)

# %%
agg_results_sims = (
    results_df.loc[:, (["wines", "games"], slice(None))]
    .T.groupby(level=1)
    .agg(["mean", "std"])
    .T.unstack()
)

# %%
agg_results = pd.concat([agg_results_classification, agg_results_sims], axis=1)

# %%
agg_results.sort_values(("accuracy", "mean"), ascending=False)

# %% [markdown]
# ---

# %% [markdown]
# ## Plotting

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

# %%
plt.rc("figure", figsize=(15, 7))

# %%
plot_df = results_df.reset_index()

# Get single ID column
plot_df[("id", "")] = plot_df.apply(
    lambda row: f"{row[('model', '')]}_{row[('pre_processing', '')]}_"
    f"{row[('dim', '')]}_{row[('min_count', '')]}",
    axis=1,
)

# Prepare to stack: swap column index levels, type of score is superior than task
plot_df = plot_df.set_index(["id", "model", "dim", "pre_processing", "min_count"])
plot_df.columns = plot_df.columns.swaplevel()

# Stack & reset index
plot_df = plot_df.stack().reset_index().rename(columns={"level_5": "task"})

# %%
plot_df.sort_values("task")

# %%
metrics = [
    "accuracy",
    "auprc",
    "f1",
    "precision",
    "recall",
    "hit_rate_at_10",
    "hit_rate_at_100",
    "mean_percentile_rank",
    "mean_reciprocal_rank",
]
classification_metrics = ["accuracy", "auprc", "f1", "precision", "recall"]
similarity_metrics = [
    "hit_rate_at_10",
    "hit_rate_at_100",
    "mean_percentile_rank",
    "mean_reciprocal_rank",
]

# %% [markdown]
# ## Studying effect of each hyperparameter

# %%
hue = "min_count"

# %%
fig, axes = plt.subplots(ceil(len(metrics) / 3), 3, figsize=(20, 12))
axes = axes.reshape(-1)
for metric, axis in zip(metrics, axes, strict=True):
    non_nan_tasks = plot_df[[metric, "task"]].dropna()["task"].unique()
    sns.lineplot(
        plot_df[plot_df["task"].isin(non_nan_tasks)],
        x="task",
        y=metric,
        hue=hue,
        ax=axis,
    )
    axis.set_title(metric)


# %% [markdown]
# ## Studying the $k$ best & worst


# %%
def get_sorted_by_metric(plot_df, metric, agg="median"):
    return plot_df.groupby("id")[metric].agg(agg).sort_values(ascending=False)


# %%
k = 5

# %%
fig, axes = plt.subplots(ceil(len(metrics) / 3), 3, figsize=(30, 20))
axes = axes.reshape(-1)
for metric, axis in zip(metrics, axes, strict=True):
    non_nan_tasks = plot_df[[metric, "task"]].dropna()["task"].unique()
    sorted_by_ids = get_sorted_by_metric(plot_df, metric)
    worst_ids = sorted_by_ids.index[:k]
    best_ids = sorted_by_ids.index[-k:]

    sns.lineplot(
        plot_df.loc[plot_df["id"].isin(best_ids) & plot_df["task"].isin(non_nan_tasks)],
        x="task",
        y=metric,
        hue="id",
        ax=axis,
    )
    sns.lineplot(
        plot_df.loc[
            plot_df["id"].isin(worst_ids) & plot_df["task"].isin(non_nan_tasks)
        ],
        x="task",
        y=metric,
        hue="id",
        ax=axis,
    )
    axis.set_title(metric)

# %%
k = 7

# %%
fig, axes = plt.subplots(
    ceil(len(metrics) / 3), 3, figsize=(28, 20), gridspec_kw={"wspace": 0.4}
)
axes = axes.reshape(-1)
for metric, axis in zip(metrics, axes, strict=True):
    non_nan_tasks = plot_df[[metric, "task"]].dropna()["task"].unique()
    sorted_by_ids = get_sorted_by_metric(plot_df, metric, agg="median")
    worst_ids = sorted_by_ids.index[-k:]
    best_ids = sorted_by_ids.index[:k]

    sns.barplot(
        plot_df.loc[plot_df["id"].isin(best_ids) | plot_df["id"].isin(worst_ids)],
        order=best_ids.tolist() + worst_ids.tolist(),
        x=metric,
        y="id",
        ax=axis,
    )
    axis.set_title(metric)

# %%
fig, axes = plt.subplots(
    ceil(len(metrics) / 3), 3, figsize=(28, 20), gridspec_kw={"wspace": 0.4}
)
axes = axes.reshape(-1)
for metric, axis in zip(metrics, axes, strict=True):
    dim_count = plot_df.copy()
    dim_count["dim_count"] = plot_df.apply(
        lambda row: f"{row['dim']}_{row['min_count']}", axis=1
    )

    non_nan_tasks = dim_count[[metric, "task"]].dropna()["task"].unique()
    sns.barplot(
        dim_count[dim_count["task"].isin(non_nan_tasks)],
        x="dim_count",
        y=metric,
        ax=axis,
    )
    axis.set_title(metric)

# %%
plot_df[plot_df["task"].isin(non_nan_tasks)].groupby(["min_count", "dim"])[
    metrics
].mean().reset_index()

# %% [markdown]
# ---
#
# ## Comparing PV vectors used in student training

# %%
relevant_df = plot_df[plot_df["id"].isin(["dbow_None_100_2", "dbow_None_768_2"])][
    similarity_metrics + ["id", "task"]
].dropna()

# %%
relevant_df[relevant_df["task"] == "games"]
