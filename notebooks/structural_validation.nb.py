# %% [markdown]
# # Validation evaluation of structural student

# %%
import pandas as pd
from transformer_document_embedding import notebook_utils as ntb_utils
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# %%
plt.rc("figure", figsize=(15, 8))
sns.set_context("paper")
sns.set_style("whitegrid")
# deep, muted, bright, pastel, dark, colorblind
sns.color_palette("muted")
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
new_evals = ntb_utils.load_validation_results(
    "../evaluations/structural_eval", struct_features
)
new_glb_evals = ntb_utils.load_validation_results(
    "../evaluations/glb_structural_eval",
    struct_features,
)
old_evals = ntb_utils.load_validation_results(
    "../evaluations/old_structural_eval",
    struct_features,
)
old_cls_evals = ntb_utils.load_validation_results(
    "../evaluations/old_cls_structural_eval",
    struct_features,
)

# %% [markdown]
# # Exploration
#
# > **Outdated**

# %% [markdown]
# ## Retrieval tasks
# %%
ret_evals = old_evals[old_evals["task_type"] == "retrieval"]


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
pres_all = {
    "new": new_evals.copy(),
    "new_glb": new_glb_evals.copy(),
    "old": old_evals.copy(),
    "old_cls": old_cls_evals.copy(),
}
for name, ev in pres_all.items():
    ev["run"] = name
pres_all = pd.concat(pres_all.values())
pres_all = pres_all[pres_all["metric"] == "accuracy"]


pres_all = ntb_utils.add_normalized_score(pres_all, ["run"])
# %%
sns.boxplot(
    pres_all,
    y="model",
    x="normalized_score",
    hue="run",
    order=pres_all[pres_all["run"] == "new_glb"]
    .groupby("model")["normalized_score"]
    .mean()
    .sort_values()
    .index,
    # errorbar=("pi", 50),
)

# %%
sns.barplot(
    pres_all,
    y="model",
    x="score",
    hue="run",
    order=pres_all[pres_all["run"] == "new_glb"]
    .groupby("model")["score"]
    .mean()
    .sort_values()
    .index,
    errorbar=("pi", 50),
)

# %%
sns.boxplot(
    pres_all,
    y="model",
    x="normalized_score",
    hue="run",
    order=pres_all[pres_all["run"] == "new_glb"]
    .groupby("model")["normalized_score"]
    .mean()
    .sort_values()
    .index,
    # errorbar=("pi", 50),
)

# %%
pres_all.groupby(["model", "run"])["normalized_score"].agg(["mean", "std"]).sort_values(
    "mean"
)

# %%
runs = pres_all["run"].unique()
fig, axes = plt.subplots(len(runs), figsize=(8, 24), gridspec_kw={"hspace": 0.15})
fig.tight_layout()

for ax, run in zip(axes, runs, strict=True):
    data = pres_all[pres_all["run"] == run]
    sns.barplot(
        data,
        x="score",
        y="model",
        hue="task",
        ax=ax,
        order=data.groupby("model")["score"].mean().sort_values().index,
    )
    ax.set_title(run)

# %% [markdown]
# ### Just new

# %%
new_evals = new_evals[new_evals["metric"] == "accuracy"]
new_evals = ntb_utils.add_normalized_score(new_evals)

# %%
sns.barplot(
    new_evals,
    x="normalized_score",
    y="model",
    order=new_evals.groupby("model")["normalized_score"].mean().sort_values().index,
)

# %%
sns.barplot(
    new_evals,
    x="score",
    y="model",
    hue="task",
    order=new_evals.groupby("model")["score"].mean().sort_values().index,
)

# %% [markdown]
# ### Just new without glb attention

# %%
new_glb_evals = new_glb_evals[new_glb_evals["metric"] == "accuracy"]
new_glb_evals = ntb_utils.add_normalized_score(new_glb_evals)

# %%
sns.barplot(
    new_glb_evals,
    x="normalized_score",
    y="model",
    order=new_glb_evals.groupby("model")["normalized_score"].mean().sort_values().index,
)

# %%
sns.barplot(
    new_glb_evals,
    x="score",
    y="model",
    hue="task",
    order=new_glb_evals.groupby("model")["score"].mean().sort_values().index,
)

# %% [markdown]
# ## Best max-marginals

# %%
mm_evals = new_glb_evals[new_glb_evals["loss_type"] == "max_marginals"]
mm_evals = mm_evals[mm_evals["metric"] == "accuracy"]

mm_evals = ntb_utils.add_normalized_score(mm_evals)

# %%
sns.barplot(
    mm_evals,
    y="mm_loss_type",
    hue="mm_lam",
    x="normalized_score",
    errorbar=("pi", 50),
)


# %%
bests_models = [
    mm_evals[mm_evals["mm_loss_type"] == loss_type]
    .groupby("model")["normalized_score"]
    .mean()
    .sort_values()
    .index[-1]
    for loss_type in ["cos_dist", "mse"]
]
sns.barplot(
    mm_evals[mm_evals["model"].isin(bests_models)],
    x="score",
    y="model",
    hue="task",
)
