# %%
import os
import pandas as pd
from transformer_document_embedding.scripts.utils import load_yaml

# %%
pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_colwidth", 500)

# %%
EVAL_DIR = "../evaluations/student_eval_correct/"

# %%
model_evals = []
for model_dir in os.scandir(EVAL_DIR):
    model_eval_results = load_yaml(os.path.join(model_dir.path, "results.yaml"))
    model_results = {
        (task, metric): score
        for task, metrics in model_eval_results.items()
        for metric, score in metrics.items()
    }
    model_results["model"] = model_dir.name
    model_evals.append(model_results)

# %%
evals = pd.DataFrame(model_evals)
evals

# %%
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
evals_long

# %%
from typing import Any


def student_features(model_str: str) -> dict[str, Any]:
    if "-" not in model_str:
        # Its a baseline
        return {}

    model_str_segments = model_str.split("-")
    adjusted_segments = []
    open_bracket_segment = None
    for segment in model_str_segments:
        if open_bracket_segment is None:
            if "[" not in segment or "]" in segment:
                adjusted_segments.append(segment)
            else:
                open_bracket_segment = segment
        else:
            open_bracket_segment += "-" + segment
            if "]" in segment:
                adjusted_segments.append(open_bracket_segment)
                open_bracket_segment = None

    def get_projection_str(relevant_segment: str) -> str:
        projection_str = relevant_segment[
            relevant_segment.find("[") + 1 : relevant_segment.find("]")
        ]
        proj_str_segments = []
        for layer_feats in projection_str.split(","):
            shortened_layer_feats = []
            for feat in layer_feats.split("-"):
                if "None" in feat:
                    continue
                shortened_layer_feats.append(feat[feat.find("=") + 1 :])
            proj_str_segments.append("".join(shortened_layer_feats))
        return "-".join(proj_str_segments)

    features = {
        "structural": model_str.startswith("depth_loss"),
        "structural_short": "short" in adjusted_segments[0],
    }

    if features["structural"]:
        features["loss_type"] = adjusted_segments[1][
            adjusted_segments[1].find("=") + 1 :
        ]
    else:
        features["dbow_dim"] = adjusted_segments[0][-3:]
        features["student_projection"] = get_projection_str(adjusted_segments[1])
        features["contextual_projection"] = get_projection_str(adjusted_segments[2])

    return features


def pv_dbow_features(model_str: str) -> dict[str, Any]:
    feats = {
        "larger_train_split": False,
    }
    segments = model_str.split("-")
    if segments[0] == "big":
        feats["larger_train_split"] = True
        del segments[0]

    feats["checkpoint_epoch"] = segments[-1][-1]
    del segments[-1]

    prop_trans = {
        "m.k.t_p_p": "pre_process",
        "m.k.m_c": "min_count",
        "m.k.v_s": "vector_size",
    }
    for seg in segments:
        prop, value = seg.split("=")
        feats[prop_trans[prop]] = value

    return feats


# %%
model_features = pd.DataFrame(map(student_features, evals_long["model"])).fillna(
    {
        "structural": False,
        "structural_short": False,
        "dbow_dim": "",
        "student_projection": "",
        "contextual_projection": "",
        "loss_type": "",
    }
)
evals_long = pd.concat((model_features, evals_long), axis=1)

# %%
evals_long

# %%
cls_tasks = ["imdb", "pan", "oc", "s2orc", "aan"]
retrieval_tasks = ["sims_wines", "sims_games"]


def only_cls(df):
    return df[df["task"].isin(cls_tasks)]


def only_retrieval(df):
    return df[df["task"].isin(retrieval_tasks)]


def only_structural(df):
    return df[
        df["structural"] | (df["model"] == "longformer") | (df["model"] == "sbert")
    ]


def only_contextual(df):
    return df[
        ~df["structural"]
        | (df["model"] == "longformer")
        | (df["model"] == "dbow_768d")
        | (df["model"] == "dbow_100d")
    ]


def only_contextual_100d(df):
    return df[
        (~df["structural"] & (df["dbow_dim"] == "100"))
        | (df["model"] == "longformer")
        | (df["model"] == "dbow_100d")
    ]


def only_contextual_768d(df):
    return df[
        (~df["structural"] & (df["dbow_dim"] == "768"))
        | (df["model"] == "longformer")
        | (df["model"] == "dbow_768d")
    ]


# %% [markdown]
# ---

# %% [markdown]
# # Plotting

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
plt.rc("figure", figsize=(14, 8))

# %%
ax = sns.barplot(
    evals_long,
    y="value",
    x="metric",
    hue="model",
    hue_order=sorted(evals_long["model"].unique()),
)

# %% [markdown]
# ## Student transformer

# %% [markdown]
# ### Only structural

# %%
only_struct = only_structural(evals_long)
only_struct_short = only_struct[~only_struct["structural_short"]]

# %%
sns.barplot(only_struct_short, x="value", y="metric", hue="model", errorbar=("pi", 100))

# %%
tmp = only_struct_short.set_index(["task", "metric", "model"]).copy()
tmp["adjusted_value"] = (
    tmp["value"] - only_struct_short.groupby(["task", "metric"])["value"].min()
)

# %%
sns.barplot(tmp, x="adjusted_value", y="metric", hue="model", errorbar=("pi", 100))


# %%
def permutation_testing(df, lower_is_better, metric_blacklist=None):
    if metric_blacklist is None:
        metric_blacklist = []

    models = df["model"].unique()
    metrics_per_task = {
        task: {
            metric: df[(df["task"] == task) & (df["metric"] == metric)].set_index(
                "model"
            )
            for metric in df[df["task"] == task]["metric"].unique()
            if metric not in metric_blacklist
        }
        for task in df["task"].unique()
    }

    metrics_limits = df.groupby(["task", "metric"])["value"].agg(["min", "max"])

    for task, metrics in metrics_per_task.items():
        for metric, view in metrics.items():
            view["value"] = (
                view["value"] - metrics_limits.loc[(task, metric), "min"]
            ) / (
                metrics_limits.loc[(task, metric), "max"]
                - metrics_limits.loc[(task, metric), "min"]
            )

    matches = []
    for model in models:
        for other in models:
            if model == other:
                continue
            for task, metric_views in metrics_per_task.items():
                for metric, metric_view in metric_views.items():
                    perf = metric_view.loc[model]["value"]
                    other_perf = metric_view.loc[other]["value"]
                    won_by = (
                        other_perf - perf
                        if metric in lower_is_better
                        else perf - other_perf
                    )
                    is_better = won_by > 0
                    matches.append(
                        {
                            "model": model,
                            "other_model": other,
                            "task": task,
                            "metric": metric,
                            "value": perf,
                            "other_value": other_perf,
                            "is_better": is_better,
                            "adjusted_won_by": won_by,
                        }
                    )

    return pd.DataFrame(matches)


# %%
matches = permutation_testing(only_struct, ["mean_percentile_rank"], ["auprc"])

# %%
matches

# %%
match_res = (
    matches.groupby("model")[["is_better", "adjusted_won_by"]].agg("sum").reset_index()
)

# %%
match_res

# %%
sns.barplot(
    match_res,
    y="model",
    x="is_better",
    order=match_res.sort_values("is_better", ascending=False)["model"],
)

# %%
sns.barplot(
    match_res,
    y="model",
    x="adjusted_won_by",
    order=match_res.sort_values("adjusted_won_by", ascending=False)["model"],
)

# %%
matches["task_type"] = "classification"
matches.loc[
    matches["task"].isin(["sims_games", "sims_wines"]), "task_type"
] = "retrieval"
matches_res_by_task = (
    matches.groupby(["task_type", "model"])[["is_better", "adjusted_won_by"]]
    .sum()
    .reset_index()
)

# %%
sns.barplot(
    matches_res_by_task,
    y="model",
    x="is_better",
    hue="task_type",
    order=matches_res_by_task[
        matches_res_by_task["task_type"] == "classification"
    ].sort_values("is_better", ascending=False)["model"],
)

# %%
sns.barplot(
    matches_res_by_task,
    y="model",
    x="adjusted_won_by",
    hue="task_type",
    order=matches_res_by_task[
        matches_res_by_task["task_type"] == "classification"
    ].sort_values("adjusted_won_by", ascending=False)["model"],
)

# %%
matches_res_by_task

# %%
tmp = only_struct.set_index(["loss_type", "task", "metric"])

only_structural_long = tmp[~tmp["structural_short"]].copy()
only_structural_long["short_delta"] = (
    tmp[~tmp["structural_short"]]["value"] - tmp[tmp["structural_short"]]["value"]
)
only_structural_long.reset_index(inplace=True)
only_structural_long

# %%
sns.barplot(only_structural_long, x="short_delta", y="metric", hue="loss_type")

# %% [markdown]
# ### Only contextual

# %% [markdown]
# #### 100d

# %%
only_100d = only_contextual_100d(evals_long)
ax = sns.barplot(
    only_100d,
    y="value",
    x="metric",
    hue="model",
    hue_order=sorted(only_100d["model"].unique()),
)
# sns.move_legend(ax, "lower left")

# %%
matches = permutation_testing(only_100d, [], ["auprc"])

# %%
matches_agg = (
    matches.groupby("model")[["is_better", "adjusted_won_by"]].sum().reset_index()
)

# %%
sns.barplot(
    matches_agg,
    y="model",
    x="is_better",
    order=matches_agg.sort_values("is_better", ascending=False)["model"],
)

# %%
sns.barplot(
    matches_agg,
    y="model",
    x="adjusted_won_by",
    order=matches_agg.sort_values("adjusted_won_by", ascending=False)["model"],
)

# %%
matches["task_type"] = "classification"
matches.loc[
    matches["task"].isin(["sims_games", "sims_wines"]), "task_type"
] = "retrieval"
matches_res_by_task = (
    matches.groupby(["task_type", "model"])[["is_better", "adjusted_won_by"]]
    .sum()
    .reset_index()
)

# %%
sns.barplot(
    matches_res_by_task,
    y="model",
    x="is_better",
    hue="task_type",
    order=matches_res_by_task[
        matches_res_by_task["task_type"] == "classification"
    ].sort_values("is_better", ascending=False)["model"],
)

# %%
sns.barplot(
    matches_res_by_task,
    y="model",
    x="adjusted_won_by",
    hue="task_type",
    order=matches_res_by_task[
        matches_res_by_task["task_type"] == "classification"
    ].sort_values("adjusted_won_by", ascending=False)["model"],
)

# %%
matches_res_by_task

# %% [markdown]
# #### 768d

# %%
only_768d = only_contextual_768d(evals_long)
ax = sns.barplot(
    only_768d,
    y="value",
    x="metric",
    hue="model",
    hue_order=sorted(only_768d["model"].unique()),
)
# sns.move_legend(ax, "lower left")

# %%
matches = permutation_testing(only_768d, [], ["auprc"])

# %%
matches_agg = (
    matches.groupby("model")[["is_better", "adjusted_won_by"]].sum().reset_index()
)

# %%
sns.barplot(
    matches_agg,
    y="model",
    x="is_better",
    order=matches_agg.sort_values("is_better", ascending=False)["model"],
)

# %%
sns.barplot(
    matches_agg,
    y="model",
    x="adjusted_won_by",
    order=matches_agg.sort_values("adjusted_won_by", ascending=False)["model"],
)

# %%
matches["task_type"] = "classification"
matches.loc[
    matches["task"].isin(["sims_games", "sims_wines"]), "task_type"
] = "retrieval"
matches_res_by_task = (
    matches.groupby(["task_type", "model"])[["is_better", "adjusted_won_by"]]
    .sum()
    .reset_index()
)

# %%
sns.barplot(
    matches_res_by_task,
    y="model",
    x="is_better",
    hue="task_type",
    order=matches_res_by_task[
        matches_res_by_task["task_type"] == "classification"
    ].sort_values("is_better", ascending=False)["model"],
)

# %%
sns.barplot(
    matches_res_by_task,
    y="model",
    x="adjusted_won_by",
    hue="task_type",
    order=matches_res_by_task[
        matches_res_by_task["task_type"] == "classification"
    ].sort_values("adjusted_won_by", ascending=False)["model"],
)

# %%
matches_res_by_task

# %% [markdown]
# #### 100d vs 768d

# %%
only_contextual = only_contextual(evals_long)
ax = sns.barplot(
    only_contextual,
    y="value",
    x="metric",
    hue="model",
    hue_order=sorted(only_contextual["model"].unique()),
)
# sns.move_legend(ax, "lower left")

# %%
matches = permutation_testing(only_contextual, [], ["auprc"])

# %%
matches_agg = (
    matches.groupby("model")[["is_better", "adjusted_won_by"]].sum().reset_index()
)

# %%
sns.barplot(
    matches_agg,
    y="model",
    x="is_better",
    order=matches_agg.sort_values("is_better", ascending=False)["model"],
)

# %%
sns.barplot(
    matches_agg,
    y="model",
    x="adjusted_won_by",
    order=matches_agg.sort_values("adjusted_won_by", ascending=False)["model"],
)

# %%
matches["task_type"] = "classification"
matches.loc[
    matches["task"].isin(["sims_games", "sims_wines"]), "task_type"
] = "retrieval"
matches_res_by_task = (
    matches.groupby(["task_type", "model"])[["is_better", "adjusted_won_by"]]
    .sum()
    .reset_index()
)

# %%
sns.barplot(
    matches_res_by_task,
    y="model",
    x="is_better",
    hue="task_type",
    order=matches_res_by_task[
        matches_res_by_task["task_type"] == "classification"
    ].sort_values("is_better", ascending=False)["model"],
)

# %%
sns.barplot(
    matches_res_by_task,
    y="model",
    x="adjusted_won_by",
    hue="task_type",
    order=matches_res_by_task[
        matches_res_by_task["task_type"] == "classification"
    ].sort_values("adjusted_won_by", ascending=False)["model"],
)

# %%
matches_res_by_task
