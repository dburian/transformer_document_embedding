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


def feature(model_str: str) -> dict[str, Any]:
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


# %%
model_features = pd.DataFrame(map(feature, evals_long["model"]))
evals_long = pd.concat((model_features, evals_long), axis=1)
evals_long.fillna(
    {
        "dbow_dim": "",
        "student_projection": "",
        "contextual_projection": "",
        "loss_type": "",
    },
    inplace=True,
)

# %%
evals_long

# %%
cls_tasks = ["imdb", "pan", "oc", "s2orc", "aan"]
retrieval_tasks = ["sims_wines", "sims_games"]


def only_cls(df):
    return df[df["task"].isin(cls_tasks)]


def only_retrieval(df):
    return df[df["task"].isin(retrieval_tasks)]


# %% [markdown]
# ---

# %% [markdown]
# ## Plotting

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
plt.rc("figure", figsize=(15, 10))

# %%
data = evals_long[~evals_long["structural"] & (evals_long["dbow_dim"] == "768")]
ax = sns.barplot(
    only_cls(data),
    y="value",
    x="metric",
    hue="model",
    hue_order=sorted(data["model"].unique()),
    legend="brief",
)
# sns.move_legend(ax, "lower left")

# %%
only_cls(data)
