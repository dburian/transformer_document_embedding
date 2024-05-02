# %%
import os
import pandas as pd
from transformer_document_embedding.scripts.utils import load_yaml
from functools import partial
import re

# %%
pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_colwidth", 500)

# %%
from typing import Any


def pv_features(model_str: str) -> dict[str, Any]:
    feats = {
        "larger_train_split": False,
    }
    segments = model_str.split("-")
    if segments[0] == "big":
        feats["larger_train_split"] = True
        del segments[0]

    feats["train_epochs"] = int(segments[-1][-1]) + 1
    del segments[-1]

    docs_per_epoch = 150000 * (3 if feats["larger_train_split"] else 1)
    feats["train_iters"] = docs_per_epoch * feats["train_epochs"]

    prop_trans = {
        "m.k.t_p_p": ("pre_process", str),
        "m.k.m_c": ("min_count", int),
        "m.k.v_s": ("vector_size", int),
    }
    for seg in segments:
        prop, value = seg.split("=")
        true_prop, type_ = prop_trans[prop]
        feats[true_prop] = type_(value)

    return feats


# %%
def load_eval_dir(path, features_func=pv_features):
    model_evals = []
    for model_dir in os.scandir(path):
        model_eval_results = load_yaml(os.path.join(model_dir.path, "results.yaml"))
        model_results = {
            (task, metric): score
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
        metric = mean_col[: mean_col.find("_")]
        tmp_evals[("imdb", metric)] = list(
            zip(
                tmp_evals[("imdb", mean_col)],
                tmp_evals[("imdb", f"{metric}_std")],
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

    # Adjusted value
    metric_limits = evals_long.groupby(["task", "metric"])["value"].agg(["min", "max"])
    by_metric = evals_long.set_index(["task", "metric", "model"])
    by_metric["adjusted_value"] = (by_metric["value"] - metric_limits["min"]) / (
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
dbow_evals_long = load_eval_dir("../evaluations/pv_dbow_gs_eval_correct/")
dbow_evals_long = add_model_features(dbow_evals_long, pv_features)

# %%
dbow_evals_long

# %%
dm_evals_long = load_eval_dir("../evaluations/pv_dm_gs_eval_correct/")
dm_evals_long = add_model_features(dm_evals_long, pv_features)

# %% [markdown]
# # Funcs

# %%
cls_tasks = ["imdb", "pan", "oc", "s2orc", "aan"]
retrieval_tasks = ["sims_wines", "sims_games"]


def only_cls(df):
    return df[df["task_type"] == "classification"]


def only_retrieval(df):
    return df[df["task_type"] == "retrieval"]


def overall_permutation_testing(df, metric_blacklist=None, less_is_better=None):
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

            other_value = other_row["adjusted_value"]
            value = row["adjusted_value"]
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


def paired_permutation_testing(
    df,
    get_opponents,
    metric_blacklist=None,
    less_is_better=None,
):
    """Perm. testing with assigned opponents"""
    if less_is_better is None:
        less_is_better = ["mean_percentile_rank"]
    if metric_blacklist is None:
        metric_blacklist = ["auprc"]

    all_models = df["model"].unique()

    df = df.set_index(["model", "task", "metric"])
    results = []
    for (model, task, metric), row in df.iterrows():
        if metric in metric_blacklist:
            continue

        opponents = get_opponents(model)
        if isinstance(opponents, str):
            opponents = [opponents]

        for other in opponents:
            if other not in all_models:
                print(f"Opponent '{other}' of model '{model}' doesn't exist.")
                continue
            other_value = df.loc[(other, task, metric), "adjusted_value"]
            value = row["adjusted_value"]
            won_by = value - other_value
            if metric in less_is_better:
                won_by *= -1

            results.append(
                {
                    "model": model,
                    "task": task,
                    "metric": metric,
                    "won_by": won_by,
                    "won": won_by > 0,
                    **row,
                }
            )

    return pd.DataFrame(results)


def epoch_other_model(model):
    epoch = int(model[-1])
    other_epoch = (epoch + 5) % 10
    return f"{model[:-1]}{other_epoch}"


epoch_permutation_testing = partial(
    paired_permutation_testing,
    get_opponents=epoch_other_model,
)


def train_split_other_model(model):
    min_count = pv_features(model)["min_count"]
    if min_count != 2:
        other_min_count = {1500: 4500, 4500: 1500}[min_count]
        model = re.sub(f"={min_count}", f"={other_min_count}", model)

    train_split = model[: model.find("-")]
    if train_split == "big":
        return model[model.find("-") + 1 :]
    else:
        return f"big-{model}"


train_split_permutation_testing = partial(
    paired_permutation_testing,
    get_opponents=train_split_other_model,
)


def train_iters_opponents(model):
    other_epoch = epoch_other_model(model)
    other_train_split = train_split_other_model(model)
    other_both = epoch_other_model(other_train_split)
    return [
        other_epoch,
        other_train_split,
        other_both,
    ]


train_iters_permutation_testing = partial(
    paired_permutation_testing,
    get_opponents=train_iters_opponents,
)


# %% [markdown]
# ---
# # Plotting

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
plt.rc("figure", figsize=(15, 8))


# %% [markdown]
# ## DBOW

# %%
overall_matches = overall_permutation_testing(dbow_evals_long)

# %%
overall_res = overall_matches.groupby("model")[["won", "won_by"]].sum().reset_index()

# %%
overall_res = pd.concat(
    (overall_res, pd.DataFrame(map(pv_features, overall_res["model"]))), axis=1
)

# %%
_, ax = plt.subplots(figsize=(8, 15))
sns.barplot(
    overall_res,
    y="model",
    x="won",
    order=overall_res.sort_values("won", ascending=False)["model"],
    hue="train_epochs",
    ax=ax,
)


# %% [markdown]
# ### Epoch 5 vs 10

# %%
epoch_matches = epoch_permutation_testing(dbow_evals_long)

# %%
epoch_matches

# %%
sns.barplot(
    epoch_matches.groupby(["larger_train_split", "train_epochs"])["won"]
    .sum()
    .reset_index(),
    x="train_epochs",
    y="won",
    hue="larger_train_split",
)

# %%
sns.barplot(
    epoch_matches.groupby(["larger_train_split", "train_iters"])["won_by"]
    .sum()
    .reset_index(),
    x="train_iters",
    y="won_by",
    hue="larger_train_split",
)

# %% [markdown]
# #### By train. iters

# %%
train_iters_matches = train_iters_permutation_testing(dbow_evals_long)

# %%
train_iters_res = (
    train_iters_matches.groupby(["larger_train_split", "train_iters"])[
        ["won", "won_by"]
    ]
    .sum()
    .reset_index()
)

# %%
sns.barplot(
    train_iters_res,
    x="train_iters",
    hue="larger_train_split",
    y="won_by",
    order=train_iters_res.sort_values("won_by")["train_iters"],
)

# %%
train_iters_res = (
    train_iters_matches.groupby(
        ["larger_train_split", "train_iters", "task", "metric"]
    )[["won", "won_by"]]
    .sum()
    .reset_index()
)

# %%
sns.swarmplot(
    train_iters_res,
    x="metric",
    hue="train_iters",
    y="won_by",
    dodge=True,
    # order=train_iters_res.sort_values("won_by")["train_iters"],
)

# %%
sns.barplot(
    train_iters_res,
    x="metric",
    hue="train_iters",
    y="won_by",
    dodge=True,
    errorbar=("pi", 100),
    # order=train_iters_res.sort_values("won_by")["train_iters"],
)

# %%
successful_models = (
    train_iters_matches.groupby("model")["won_by"]
    .sum()
    .sort_values(ascending=False)
    .iloc[:36]
)
successful_models

# %%
suc_train_iter_res = (
    train_iters_matches[train_iters_matches["model"].isin(successful_models.index)]
    .groupby(["train_iters", "task", "metric"])[["won_by", "won"]]
    .sum()
    .reset_index()
)


# %%
sns.swarmplot(
    suc_train_iter_res,
    x="metric",
    hue="train_iters",
    y="won_by",
    dodge=True,
    # order=train_iters_res.sort_values("won_by")["train_iters"],
)

# %% [markdown]
# #### By metric

# %%
epoch_res = (
    epoch_matches.groupby(["larger_train_split", "train_epochs", "task", "metric"])[
        ["won", "won_by"]
    ]
    .sum()
    .reset_index()
)
epoch_res["big_epoch"] = (
    epoch_res["larger_train_split"].astype(str)
    + "_"
    + epoch_res["train_epochs"].astype(str)
)

# %%
sns.barplot(
    epoch_res,
    x="metric",
    y="won_by",
    hue="big_epoch",
    order=epoch_res.sort_values("won_by", ascending=True)["metric"],
    errorbar=("pi", 100),
)

# %%
sns.barplot(
    epoch_res[epoch_res["metric"] == "recall"], x="task", y="won_by", hue="big_epoch"
)


# %% [markdown]
# #### Is big better for epochs 10


# %%
evals_long_10 = dbow_evals_long[dbow_evals_long["train_epochs"] == 10]

# %%
evals_long_10.shape

# %%
train_matches = train_split_permutation_testing(evals_long_10)
train_res = train_matches.groupby("model")[["won", "won_by"]].sum().reset_index()
train_res = add_model_features(train_res, pv_features)

# %%
sns.barplot(
    train_res,
    x="won_by",
    y="model",
    hue="larger_train_split",
    order=train_res.sort_values("won_by", ascending=False)["model"],
)

# %% [markdown]
# #### Focusing on k best models

# %%
k = overall_res.shape[0] // 2
top_models = overall_res.sort_values("won_by", ascending=False)["model"].iloc[:k]

# %%
evals_tops = dbow_evals_long[dbow_evals_long["model"].isin(top_models)]
tops_matches = epoch_permutation_testing(evals_tops)
tops_res = (
    tops_matches.groupby(["larger_train_split", "train_epochs"])[["won", "won_by"]]
    .sum()
    .reset_index()
)

# %%
sns.barplot(
    tops_res, x="won_by", y="train_epochs", hue="larger_train_split", orient="h"
)

# %%
tops_res = (
    tops_matches.groupby(["larger_train_split", "train_epochs", "metric"])[
        ["won", "won_by"]
    ]
    .sum()
    .reset_index()
)
tops_res["big_epoch"] = (
    tops_res["larger_train_split"].astype(str)
    + "_"
    + tops_res["train_epochs"].astype(str)
)

# %%
sns.barplot(
    tops_res,
    x="metric",
    y="won_by",
    hue="big_epoch",
    order=epoch_res.sort_values("won_by", ascending=True)["metric"],
    errorbar=("pi", 100),
)


# %% [markdown]
# #### Small training set longer training vs larger training set less epochs?


# %%
def get_other_other(model):
    return epoch_other_model(train_split_other_model(model))


big_quick = dbow_evals_long["larger_train_split"] & (
    dbow_evals_long["train_epochs"] == 5
)
small_long = ~dbow_evals_long["larger_train_split"] & (
    dbow_evals_long["train_epochs"] == 10
)
matches = paired_permutation_testing(
    dbow_evals_long[big_quick | small_long], get_opponents=get_other_other
)
res = matches.groupby("train_iters")[["won", "won_by"]].sum().reset_index()

# %%
sns.barplot(res, x="train_iters", y="won_by")

# %% [markdown]
# ### Large vs small training set
#
# **Do they behave the same?**

# %%
evals_long_10_big = dbow_evals_long[
    (dbow_evals_long["train_epochs"] == 10) & dbow_evals_long["larger_train_split"]
]
evals_long_10_small = dbow_evals_long[
    (dbow_evals_long["train_epochs"] == 10) & ~dbow_evals_long["larger_train_split"]
]

# %%
big_matches = overall_permutation_testing(evals_long_10_big)
small_matches = overall_permutation_testing(evals_long_10_small)


# %%
def model_id(model):
    features = pv_features(model)

    return (
        f"tpp={features['pre_process']}-vs={features['vector_size']}-"
        f"mc={'small' if features['min_count'] == 2 else 'big'}"
    )


# %%
small_matches["model_id"] = [model_id(m) for m in small_matches["model"]]
small_matches["other_model_id"] = [model_id(m) for m in small_matches["other_model"]]

# %%
big_matches["model_id"] = [model_id(m) for m in big_matches["model"]]
big_matches["other_model_id"] = [model_id(m) for m in big_matches["other_model"]]

# %%
big_matches = big_matches.set_index(
    ["model_id", "other_model_id", "task", "metric"]
).sort_index()
small_matches = small_matches.set_index(
    ["model_id", "other_model_id", "task", "metric"]
).sort_index()

# %%
wons = big_matches["won"] == small_matches["won"]

# %%
sns.histplot(wons, discrete=True, stat="percent")

# %%
wons_by = (big_matches["won_by"] - small_matches["won_by"]).abs()

# %%
sns.histplot(wons_by, stat="percent", binwidth=0.2)

# %%
big_matches["won_by"].corr(small_matches["won_by"])

# %% [markdown]
# #### Does some model attribute cause better results in one or the other?

# %%
fig, (big_ax, small_ax) = plt.subplots(ncols=2, figsize=(20, 8))
sns.histplot(big_matches, x="won_by", hue="pre_process", element="step", ax=big_ax)
big_ax.set_title("big")
sns.histplot(small_matches, x="won_by", hue="pre_process", element="step", ax=small_ax)
small_ax.set_title("small")

# %%
fig, (big_ax, small_ax) = plt.subplots(ncols=2, figsize=(20, 8))
sns.histplot(big_matches, x="won_by", hue="min_count", element="step", ax=big_ax)
big_ax.set_title("big")
sns.histplot(small_matches, x="won_by", hue="min_count", element="step", ax=small_ax)
small_ax.set_title("small")

# %%
fig, (big_ax, small_ax) = plt.subplots(ncols=2, figsize=(20, 8))
sns.histplot(big_matches, x="won_by", hue="vector_size", element="step", ax=big_ax)
big_ax.set_title("big")
sns.histplot(small_matches, x="won_by", hue="vector_size", element="step", ax=small_ax)
small_ax.set_title("small")

# %% [markdown]
# #### Top k models from each

# %%
top_k_models_big = (
    big_matches.reset_index()
    .groupby("model_id")["won_by"]
    .sum()
    .reset_index()
    .sort_values("won_by", ascending=False)
    .iloc[:15]
)

# %%
top_k_models_small = (
    small_matches.reset_index()
    .groupby("model_id")["won_by"]
    .sum()
    .reset_index()
    .sort_values("won_by", ascending=False)
    .iloc[:15]
)

# %%
top_k_models_big = top_k_models_big.reset_index().drop(columns="index")
top_k_models_small = top_k_models_small.reset_index().drop(columns="index")

# %%
pd.concat([top_k_models_big, top_k_models_small], axis=1)

# %%
best_dbow = top_k_models_big.copy()

# %% [markdown]
# #### Which differ?

# %%
small_differ = small_matches[big_matches["won"] != small_matches["won"]].reset_index()

# %%
sns.histplot(small_differ, x="task_type", stat="percent")

# %%
sns.histplot(small_differ, x="value", stat="percent", y="metric")

# %%
small_differ["model_id"] = pd.Categorical(
    small_differ["model_id"], small_differ["model_id"].unique().sort()
)

# %%
small_differ["other_model_id"] = pd.Categorical(
    small_differ["other_model_id"], small_differ["other_model_id"].unique().sort()
)

# %%
ax = sns.histplot(small_differ, x="model_id", y="other_model_id", stat="percent")
ax.tick_params("x", rotation=90)

# %%
other_features = pd.DataFrame(map(pv_features, small_differ["other_model"]))
other_features.columns = [f"other_{col}" for col in other_features.columns]

# %%
small_differ = pd.concat([small_differ, other_features], axis=1)

# %%
sns.histplot(
    small_differ, x="pre_process", y="other_pre_process", stat="percent", cbar=True
)

# %%
sns.histplot(
    small_differ, x="min_count", y="other_min_count", stat="percent", cbar=True
)

# %%
sns.histplot(
    small_differ, x="vector_size", y="other_vector_size", stat="percent", cbar=True
)

# %% [markdown]
# ### The best model

# %% [markdown]
# #### Best `model_id`

# %%
all_matches = pd.concat([big_matches, small_matches], axis=0).reset_index()
all_res = all_matches.groupby(["model_id"])[["won", "won_by"]].sum().reset_index()

# %%
sns.barplot(
    all_res,
    x="won",
    y="model_id",
    order=all_res.sort_values("won", ascending=False)["model_id"],
)

# %%
sns.barplot(
    all_res,
    x="won_by",
    y="model_id",
    order=all_res.sort_values("won_by", ascending=False)["model_id"],
)

# %% [markdown]
# #### Best in large training split

# %%
big_res = big_matches.groupby("model")[["won", "won_by"]].sum().reset_index()

# %%
sns.barplot(
    big_res,
    y="model",
    x="won_by",
    order=big_res.sort_values("won_by", ascending=False)["model"],
)

# %% [markdown]
# #### Best in small training split

# %%
small_res = small_matches.groupby("model")[["won", "won_by"]].sum().reset_index()

# %%
sns.barplot(
    small_res,
    y="model",
    x="won_by",
    order=small_res.sort_values("won_by", ascending=False)["model"],
)

# %%
all_matches = pd.concat([big_matches, small_matches], axis=0).reset_index()

# %%
all_res = all_matches.groupby(["model"])[["won", "won_by"]].sum().reset_index()
all_res = pd.concat(
    [
        all_res,
        pd.DataFrame(map(pv_features, all_res["model"])),
        pd.DataFrame({"model_id": map(model_id, all_res["model"])}),
    ],
    axis=1,
)

# %%
sns.barplot(
    all_res,
    y="model_id",
    x="won_by",
    order=all_res[all_res["larger_train_split"]].sort_values("won_by", ascending=False)[
        "model_id"
    ],
    hue="larger_train_split",
)

# %%
sns.swarmplot(all_res, y="won", x="vector_size", hue="larger_train_split")

# %%
sns.swarmplot(all_res, y="won", x="pre_process", hue="larger_train_split")

# %%
sns.swarmplot(all_res, y="won", x="min_count", hue="larger_train_split")

# %%
small_epoch10 = dbow_evals_long[
    ~dbow_evals_long["larger_train_split"] & (dbow_evals_long["train_epochs"] == 10)
]

# %%
bests_matches = overall_permutation_testing(small_epoch10)

# %%
bests_res = bests_matches.groupby("model")[["won", "won_by"]].sum().reset_index()
bests_res = pd.concat(
    [bests_res, pd.DataFrame(map(pv_features, bests_res["model"]))], axis=1
)

# %%
sns.barplot(
    bests_res,
    x="won",
    y="model",
    order=bests_res.sort_values("won", ascending=False)["model"],
)

# %%
sns.barplot(
    bests_res,
    x="won_by",
    y="model",
    order=bests_res.sort_values("won_by", ascending=False)["model"],
)

# %% [markdown]
# #### Best in classification tasks

# %%
present_evals = dbow_evals_long[
    dbow_evals_long["larger_train_split"] & (dbow_evals_long["train_epochs"] == 10)
]
present_matches = overall_permutation_testing(
    present_evals, metric_blacklist=["auprc", "precision", "recall"]
)
present_res = (
    present_matches.groupby(["task_type", "model"])[["won", "won_by"]]
    .sum()
    .reset_index()
)

# %%
sns.barplot(
    present_res,
    x="won_by",
    y="model",
    hue="task_type",
    order=present_res[present_res["task_type"] == "classification"].sort_values(
        "won_by", ascending=False
    )["model"],
)

# %%
present_evals = dbow_evals_long[
    dbow_evals_long["larger_train_split"] & (dbow_evals_long["train_epochs"] == 10)
]
present_matches = overall_permutation_testing(
    present_evals, metric_blacklist=["auprc", "precision", "recall"]
)
present_res = (
    present_matches.groupby(["task_type", "model"])[["won", "won_by"]]
    .sum()
    .reset_index()
)

# %%
sns.barplot(
    present_res,
    x="won",
    y="model",
    hue="task_type",
    order=present_res[present_res["task_type"] == "classification"].sort_values(
        "won", ascending=False
    )["model"],
)

# %% [markdown]
# #### No permutation testing

# %%
agg_pres_evals = (
    dbow_evals_long[
        dbow_evals_long["larger_train_split"] & (dbow_evals_long["train_epochs"] == 10)
    ]
    .groupby(["model", "task_type", "metric"])["adjusted_value"]
    .mean()
    .reset_index()
)

# %%
agg_acc_pres_evals = agg_pres_evals[agg_pres_evals["metric"] == "accuracy"].sort_values(
    "adjusted_value", ascending=False
)

# %%
sns.barplot(
    agg_acc_pres_evals, y="model", x="adjusted_value"
)  # , order=agg_acc_pres_evals.sort)

# %%
dbow_evals_long[
    dbow_evals_long["model"] == "big-m.k.t_p_p=None-m.k.v_s=768-m.k.m_c=2-after_epoch_9"
]

# %%
agg_both_pres_evals = agg_pres_evals[
    agg_pres_evals["metric"].isin(["accuracy", "f1"])
].sort_values("adjusted_value", ascending=False)

# %%
sns.barplot(agg_both_pres_evals, y="model", x="adjusted_value", errorbar=("pi", 100))

# %%
agg_agg_pres_results = (
    dbow_evals_long[
        dbow_evals_long["larger_train_split"]
        & (dbow_evals_long["train_epochs"] == 10)
        & (dbow_evals_long["metric"].isin(["accuracy", "f1"]))
    ]
    .groupby("model")["adjusted_value"]
    .mean()
    .reset_index()
)

# %%
sns.barplot(
    agg_agg_pres_results.sort_values("adjusted_value", ascending=False),
    y="model",
    x="adjusted_value",
)

# %% [markdown]
# #### Convergence of the best models

# %%
best_model_ids = [
    "tpp=lowercase-vs=1024-mc=small",
    "tpp=None-vs=1024-mc=small",
    "tpp=lowercase-vs=768-mc=small",
    "tpp=None-vs=768-mc=small",
    "tpp=stem-vs=1024-mc=small",
    "tpp=stem-vs=768-mc=small",
]

# %%
dbow_evals_long["model_id"] = [model_id(model) for model in dbow_evals_long["model"]]

# %%
dbow_evals_long["model_id"]

# %%
bests_matches = train_iters_permutation_testing(
    dbow_evals_long[dbow_evals_long["model_id"].isin(best_model_ids)]
)

# %%
bests_matches.shape

# %%
bests_res = (
    bests_matches.groupby(["model_id", "train_iters"])[["won", "won_by"]]
    .sum()
    .reset_index()
)

# %%
sns.barplot(
    bests_res, y="model_id", hue="train_iters", x="won_by", order=best_model_ids
)

# %% [markdown]
# #### Which models loose on accuracy but win on f1?

# %%
evals_sel = dbow_evals_long[dbow_evals_long["task_type"] == "classification"]
evals_sel = evals_sel[evals_sel["larger_train_split"]]
evals_sel = evals_sel[evals_sel["train_epochs"] == 10]

matches = overall_permutation_testing(evals_sel, metric_blacklist=["auprc"]).set_index(
    ["model", "task", "other_model"]
)
acc_better = matches[matches["won"] & (matches["metric"] == "accuracy")].index
f1_better = matches[matches["won"] & (matches["metric"] == "f1")].index

matches["better_metric"] = "both_lost"
matches.loc[acc_better, "better_metric"] = "acc"
matches.loc[f1_better, "better_metric"] = "f1"
matches.loc[acc_better.intersection(f1_better), "better_metric"] = "both"

# %%
sns.histplot(matches, x="better_metric")

# %%
sns.violinplot(matches, x="won", y="metric", hue="better_metric")

# %%
sns.barplot(matches, x="won", y="metric", hue="better_metric")

# %%
sns.violinplot(matches, x="adjusted_value", hue="better_metric", y="metric")

# %%
sns.barplot(
    matches[matches["metric"].isin(["precision", "recall"])],
    x="won",
    hue="better_metric",
    y="metric",
)

# %%
sns.histplot(matches, x="adjusted_value", y="metric", hue="better_metric")

# %% [markdown]
# ## DM

# %%
overall_matches = overall_permutation_testing(dm_evals_long)

# %%
overall_res = overall_matches.groupby("model")[["won", "won_by"]].sum().reset_index()

# %%
overall_res = pd.concat(
    (overall_res, pd.DataFrame(map(pv_features, overall_res["model"]))), axis=1
)

# %%
_, ax = plt.subplots(figsize=(8, 15))
sns.barplot(
    overall_res,
    y="model",
    x="won",
    order=overall_res.sort_values("won", ascending=False)["model"],
    hue="train_epochs",
    ax=ax,
)


# %% [markdown]
# ### Epoch 5 vs 10


# %%
epoch_matches = epoch_permutation_testing(dm_evals_long)

# %%
epoch_matches

# %%
sns.barplot(
    epoch_matches.groupby(["larger_train_split", "train_epochs"])["won"]
    .sum()
    .reset_index(),
    x="train_epochs",
    y="won",
    hue="larger_train_split",
)

# %%
sns.barplot(
    epoch_matches.groupby(["larger_train_split", "train_epochs"])["won_by"]
    .sum()
    .reset_index(),
    x="train_epochs",
    y="won_by",
    hue="larger_train_split",
)

# %% [markdown]
# #### By train. iters

# %%
train_iters_matches = train_iters_permutation_testing(dm_evals_long)

# %%
train_iters_res = (
    train_iters_matches.groupby(["larger_train_split", "train_iters"])[
        ["won", "won_by"]
    ]
    .sum()
    .reset_index()
)

# %%
sns.barplot(
    train_iters_res,
    x="train_iters",
    hue="larger_train_split",
    y="won_by",
    order=train_iters_res.sort_values("won_by")["train_iters"],
)

# %%
train_iters_res = (
    train_iters_matches.groupby(
        ["larger_train_split", "train_iters", "task", "metric"]
    )[["won", "won_by"]]
    .sum()
    .reset_index()
)

# %%
sns.swarmplot(
    train_iters_res,
    x="metric",
    hue="train_iters",
    y="won_by",
    dodge=True,
    # order=train_iters_res.sort_values("won_by")["train_iters"],
)

# %%
sns.barplot(
    train_iters_res,
    x="metric",
    hue="train_iters",
    y="won_by",
    dodge=True,
    errorbar=("pi", 100),
    # order=train_iters_res.sort_values("won_by")["train_iters"],
)

# %%
successful_models = (
    train_iters_matches.groupby("model")["won_by"]
    .sum()
    .sort_values(ascending=False)
    .iloc[:36]
)
successful_models

# %%
suc_train_iter_res = (
    train_iters_matches[train_iters_matches["model"].isin(successful_models.index)]
    .groupby(["train_iters", "task", "metric"])[["won_by", "won"]]
    .sum()
    .reset_index()
)


# %%
sns.swarmplot(
    suc_train_iter_res,
    x="metric",
    hue="train_iters",
    y="won_by",
    dodge=True,
    # order=train_iters_res.sort_values("won_by")["train_iters"],
)

# %% [markdown]
# #### By metric

# %%
epoch_res = (
    epoch_matches.groupby(["larger_train_split", "train_epochs", "task", "metric"])[
        ["won", "won_by"]
    ]
    .sum()
    .reset_index()
)
epoch_res["big_epoch"] = (
    epoch_res["larger_train_split"].astype(str)
    + "_"
    + epoch_res["train_epochs"].astype(str)
)

# %%
sns.barplot(
    epoch_res,
    x="metric",
    y="won_by",
    hue="big_epoch",
    order=epoch_res.sort_values("won_by", ascending=True)["metric"],
    errorbar=("pi", 100),
)

# %%
sns.barplot(
    epoch_res[epoch_res["metric"] == "recall"], x="task", y="won_by", hue="big_epoch"
)


# %% [markdown]
# #### Is big better for epochs 10


# %%
evals_long_10 = dm_evals_long[dm_evals_long["train_epochs"] == 10]

# %%
evals_long_10.shape

# %%
train_matches = train_split_permutation_testing(
    evals_long_10, metric_blacklist=["auprc"], less_is_better=["mean_percentile_rank"]
)
train_res = train_matches.groupby("model")[["won", "won_by"]].sum().reset_index()
train_res = pd.concat(
    [train_res, pd.DataFrame(map(pv_features, train_res["model"]))], axis=1
)

# %%
sns.barplot(
    train_res,
    x="won_by",
    y="model",
    hue="larger_train_split",
    order=train_res.sort_values("won_by", ascending=False)["model"],
)

# %% [markdown]
# #### Focusing on k best models

# %%
k = overall_res.shape[0] // 2
top_models = overall_res.sort_values("won_by", ascending=False)["model"].iloc[:k]

# %%
evals_tops = dm_evals_long[dm_evals_long["model"].isin(top_models)]
tops_matches = epoch_permutation_testing(evals_tops)
tops_res = (
    tops_matches.groupby(["larger_train_split", "train_epochs"])[["won", "won_by"]]
    .sum()
    .reset_index()
)

# %%
sns.barplot(tops_res, y="won_by", x="train_epochs", hue="larger_train_split")

# %%
tops_res = (
    tops_matches.groupby(["larger_train_split", "train_epochs", "metric"])[
        ["won", "won_by"]
    ]
    .sum()
    .reset_index()
)
tops_res["big_epoch"] = (
    tops_res["larger_train_split"].astype(str)
    + "_"
    + tops_res["train_epochs"].astype(str)
)

# %%
sns.barplot(
    tops_res,
    x="metric",
    y="won_by",
    hue="big_epoch",
    order=epoch_res.sort_values("won_by", ascending=True)["metric"],
    errorbar=("pi", 100),
)

# %% [markdown]
# ### Large vs small training set
#
# **Do they behave the same?**

# %%
evals_long_10_big = dm_evals_long[
    dm_evals_long["larger_train_split"] & (dm_evals_long["train_epochs"] == 10)
]
evals_long_10_small = dm_evals_long[
    ~dm_evals_long["larger_train_split"] & (dm_evals_long["train_epochs"] == 10)
]

# %%
big_matches = overall_permutation_testing(
    evals_long_10_big[
        evals_long_10_big["model"]
        != "big-m.k.t_p_p=None-m.k.v_s=1024-m.k.m_c=2-after_epoch_10"
    ]
)
small_matches = overall_permutation_testing(evals_long_10_small)


# %%
def model_id(model):
    features = pv_features(model)

    return (
        f"tpp={features['pre_process']}-vs={features['vector_size']}"
        f"-mc={'small' if features['min_count'] == 2 else 'big'}"
    )


# %%
small_matches["model_id"] = [model_id(m) for m in small_matches["model"]]
small_matches["other_model_id"] = [model_id(m) for m in small_matches["other_model"]]

# %%
big_matches["model_id"] = [model_id(m) for m in big_matches["model"]]
big_matches["other_model_id"] = [model_id(m) for m in big_matches["other_model"]]

# %%
big_matches = big_matches.set_index(
    ["model_id", "other_model_id", "task", "metric"]
).sort_index()
small_matches = small_matches.set_index(
    ["model_id", "other_model_id", "task", "metric"]
).sort_index()

# %%
wons = big_matches["won"] == small_matches["won"]

# %%
sns.histplot(wons, discrete=True, stat="percent")

# %%
wons_by = (big_matches["won_by"] - small_matches["won_by"]).abs()

# %%
sns.histplot(wons_by, stat="percent", binwidth=0.1)

# %%
big_matches["won_by"].corr(small_matches["won_by"])

# %% [markdown]
# #### Does some model attribute cause better results in one or the other?

# %%
fig, (big_ax, small_ax) = plt.subplots(ncols=2, figsize=(20, 8))
sns.histplot(big_matches, x="won_by", hue="pre_process", element="step", ax=big_ax)
big_ax.set_title("big")
sns.histplot(small_matches, x="won_by", hue="pre_process", element="step", ax=small_ax)
small_ax.set_title("small")

# %%
fig, (big_ax, small_ax) = plt.subplots(ncols=2, figsize=(20, 8))
sns.histplot(big_matches, x="won_by", hue="min_count", element="step", ax=big_ax)
big_ax.set_title("big")
sns.histplot(small_matches, x="won_by", hue="min_count", element="step", ax=small_ax)
small_ax.set_title("small")

# %%
fig, (big_ax, small_ax) = plt.subplots(ncols=2, figsize=(20, 8))
sns.histplot(big_matches, x="won_by", hue="vector_size", element="step", ax=big_ax)
big_ax.set_title("big")
sns.histplot(small_matches, x="won_by", hue="vector_size", element="step", ax=small_ax)
small_ax.set_title("small")

# %% [markdown]
# #### Top k models from each

# %%
top_k_models_big = (
    big_matches.reset_index()
    .groupby("model_id")["won_by"]
    .sum()
    .reset_index()
    .sort_values("won_by", ascending=False)
    .iloc[:15]
)

# %%
top_k_models_small = (
    small_matches.reset_index()
    .groupby("model_id")["won_by"]
    .sum()
    .reset_index()
    .sort_values("won_by", ascending=False)
    .iloc[:15]
)

# %%
top_k_models_big = top_k_models_big.reset_index().drop(columns="index")
top_k_models_small = top_k_models_small.reset_index().drop(columns="index")

# %%
pd.concat([top_k_models_big, top_k_models_small], axis=1)

# %%
best_dm = top_k_models_big.copy()

# %% [markdown]
# #### Which differ?

# %%
small_differ = small_matches[big_matches["won"] != small_matches["won"]].reset_index()

# %%
sns.histplot(small_differ, x="task_type", stat="percent")

# %%
sns.histplot(small_differ, x="value", stat="percent", y="metric")

# %%
small_differ["model_id"] = pd.Categorical(
    small_differ["model_id"], small_differ["model_id"].unique().sort()
)

# %%
small_differ["other_model_id"] = pd.Categorical(
    small_differ["other_model_id"], small_differ["other_model_id"].unique().sort()
)

# %%
ax = sns.histplot(small_differ, x="model_id", y="other_model_id", stat="percent")
ax.tick_params("x", rotation=90)

# %%
other_features = pd.DataFrame(map(pv_features, small_differ["other_model"]))
other_features.columns = [f"other_{col}" for col in other_features.columns]

# %%
small_differ = pd.concat([small_differ, other_features], axis=1)

# %%
sns.histplot(
    small_differ, x="pre_process", y="other_pre_process", stat="percent", cbar=True
)

# %%
sns.histplot(
    small_differ, x="min_count", y="other_min_count", stat="percent", cbar=True
)

# %%
sns.histplot(
    small_differ, x="vector_size", y="other_vector_size", stat="percent", cbar=True
)

# %% [markdown]
# ### The best model

# %% [markdown]
# #### Best `model_id`

# %%
all_matches = pd.concat([big_matches, small_matches], axis=0).reset_index()
all_res = all_matches.groupby(["model_id"])[["won", "won_by"]].sum().reset_index()

# %%
sns.barplot(
    all_res,
    x="won",
    y="model_id",
    order=all_res.sort_values("won", ascending=False)["model_id"],
)

# %%
sns.barplot(
    all_res,
    x="won_by",
    y="model_id",
    order=all_res.sort_values("won_by", ascending=False)["model_id"],
)

# %% [markdown]
# #### Best overall

# %%
all_matches = overall_permutation_testing(dm_evals_long)
all_res = all_matches.groupby("model")[["won", "won_by"]].sum().reset_index()

# %%
_, ax = plt.subplots(figsize=(12, 16))
sns.barplot(
    all_res,
    y="model",
    x="won_by",
    order=all_res.sort_values("won_by", ascending=False)["model"],
)

# %% [markdown]
# #### Best in large training split

# %%
big_res = big_matches.groupby("model")[["won", "won_by"]].sum().reset_index()

# %%
sns.barplot(
    big_res,
    y="model",
    x="won_by",
    order=big_res.sort_values("won_by", ascending=False)["model"],
)

# %% [markdown]
# #### Best in small training split

# %%
small_res = small_matches.groupby("model")[["won", "won_by"]].sum().reset_index()

# %%
sns.barplot(
    small_res,
    y="model",
    x="won_by",
    order=small_res.sort_values("won_by", ascending=False)["model"],
)

# %%
all_matches = pd.concat([big_matches, small_matches], axis=0).reset_index()

# %%
all_res = all_matches.groupby(["model"])[["won", "won_by"]].sum().reset_index()
all_res = pd.concat(
    [
        all_res,
        pd.DataFrame(map(pv_features, all_res["model"])),
        pd.DataFrame({"model_id": map(model_id, all_res["model"])}),
    ],
    axis=1,
)

# %%
sns.barplot(
    all_res,
    y="model_id",
    x="won_by",
    order=all_res[all_res["larger_train_split"]].sort_values("won_by", ascending=False)[
        "model_id"
    ],
    hue="larger_train_split",
)

# %%
sns.barplot(all_res, y="won", x="vector_size", hue="larger_train_split", errorbar="sd")

# %%
sns.barplot(all_res, y="won", x="pre_process", hue="larger_train_split")

# %%
sns.barplot(all_res, y="won", x="min_count", hue="larger_train_split")

# %%
small_epoch10 = dm_evals_long[
    ~dm_evals_long["larger_train_split"] & (dm_evals_long["train_epochs"] == 10)
]

# %%
bests_matches = overall_permutation_testing(small_epoch10)

# %%
bests_res = bests_matches.groupby("model")[["won", "won_by"]].sum().reset_index()
bests_res = pd.concat(
    [bests_res, pd.DataFrame(map(pv_features, bests_res["model"]))], axis=1
)

# %%
sns.barplot(
    bests_res,
    x="won",
    y="model",
    order=bests_res.sort_values("won", ascending=False)["model"],
)

# %%
sns.barplot(
    bests_res,
    x="won_by",
    y="model",
    order=bests_res.sort_values("won_by", ascending=False)["model"],
)

# %% [markdown]
# #### Best in classification tasks

# %%
present_evals = dm_evals_long[
    dm_evals_long["larger_train_split"] & (dm_evals_long["train_epochs"] == 10)
]
present_matches = overall_permutation_testing(
    present_evals, metric_blacklist=["auprc", "f1", "precision", "recall"]
)
present_res = (
    present_matches.groupby(["task_type", "model"])[["won", "won_by"]]
    .sum()
    .reset_index()
)

# %%
sns.barplot(
    present_res,
    x="won_by",
    y="model",
    hue="task_type",
    order=present_res[present_res["task_type"] == "classification"].sort_values(
        "won_by", ascending=False
    )["model"],
)

# %% [markdown]
# ## DBOW vs DM

# %% [markdown]
# ### Best overall

# %%
k = 6

# %%
pd.concat([best_dm, best_dbow], axis=1)

# %%
best_dm_model_ids = best_dm["model_id"].iloc[:k]
best_dbow_model_ids = best_dbow["model_id"].iloc[:k]

# %%
dbow_evals_long["model_id"] = [model_id(model) for model in dbow_evals_long["model"]]
dm_evals_long["model_id"] = [model_id(model) for model in dm_evals_long["model"]]

# %%
best_dbow_evals = dbow_evals_long[
    dbow_evals_long["model_id"].isin(best_dbow_model_ids)
    & dbow_evals_long["larger_train_split"]
    & (dbow_evals_long["train_epochs"] == 10)
].copy()
best_dm_evals = dm_evals_long[
    dm_evals_long["model_id"].isin(best_dm_model_ids)
    & dm_evals_long["larger_train_split"]
    & (dm_evals_long["train_epochs"] == 10)
].copy()

# %%
best_dbow_evals["model_type"] = "dbow"
best_dm_evals["model_type"] = "dm"

# %%
best_evals = pd.concat([best_dbow_evals, best_dm_evals], axis=0)

# %%
best_matches = overall_permutation_testing(best_evals)
best_res = (
    best_matches.groupby(["model_id", "model_type"])[["won", "won_by"]]
    .sum()
    .reset_index()
)

# %%
sns.barplot(
    best_res,
    y="model_id",
    hue="model_type",
    x="won",
    order=best_res.sort_values("won", ascending=False)["model_id"],
)

# %%
sns.barplot(
    best_res,
    y="model_id",
    hue="model_type",
    x="won_by",
    order=best_res.sort_values("won_by", ascending=False)["model_id"],
)

# %%
best_res_cls = (
    best_matches.groupby(["model_type", "model_id", "task_type"])[["won", "won_by"]]
    .sum()
    .reset_index()
)

# %%
best_res_cls["model_type_id"] = (
    best_res_cls["model_type"] + "_" + best_res_cls["model_id"]
)

# %%
sns.barplot(
    best_res_cls,
    y="model_type_id",
    x="won_by",
    hue="task_type",
    order=best_res_cls.sort_values("won_by", ascending=False)["model_type_id"],
)

# %% [markdown]
# ### Presentable best

# %%
pres_best_dm_model_ids = best_dm_model_ids
pres_best_dbow_model_ids = pd.concat(
    [
        best_dbow_model_ids[best_dbow_model_ids != "tpp=stem-vs=768-mc=small"],
        pd.Series(["tpp=lowercase-vs=100-mc=small"]),
    ]
).reset_index()[0]

# %%
best_dbow_evals = dbow_evals_long[
    dbow_evals_long["model_id"].isin(pres_best_dbow_model_ids)
    & dbow_evals_long["larger_train_split"]
    & (dbow_evals_long["train_epochs"] == 10)
].copy()
best_dm_evals = dm_evals_long[
    dm_evals_long["model_id"].isin(pres_best_dm_model_ids)
    & dm_evals_long["larger_train_split"]
    & (dm_evals_long["train_epochs"] == 10)
].copy()

# %%
best_dbow_evals["model_type"] = "dbow"
best_dm_evals["model_type"] = "dm"

# %%
best_evals = pd.concat([best_dbow_evals, best_dm_evals], axis=0)

# %%
best_matches = overall_permutation_testing(best_evals)
best_res = (
    best_matches.groupby(["model_id", "model_type"])[["won", "won_by"]]
    .sum()
    .reset_index()
)

# %%
sns.barplot(
    best_res,
    y="model_id",
    hue="model_type",
    x="won",
    order=best_res.sort_values("won", ascending=False)["model_id"],
)

# %%
sns.barplot(
    best_res,
    y="model_id",
    hue="model_type",
    x="won_by",
    order=best_res.sort_values("won_by", ascending=False)["model_id"],
)

# %%
best_res_cls = (
    best_matches.groupby(["model_type", "model_id", "task_type"])[["won", "won_by"]]
    .sum()
    .reset_index()
)

# %%
best_res_cls["model_type_id"] = (
    best_res_cls["model_type"] + "_" + best_res_cls["model_id"]
)

# %%
sns.barplot(
    best_res_cls,
    y="model_type_id",
    x="won_by",
    hue="task_type",
    order=best_res_cls.sort_values("won_by", ascending=False)["model_type_id"],
)

# %%
best_res_cls[best_res_cls["task_type"] == "classification"].sort_values("won_by")
