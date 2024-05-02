# %% [markdown]
# # Validation evaluation of structural student

# %%
import pandas as pd
from transformer_document_embedding import notebook_utils as ntb_utils
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# %%
ntb_utils.seaborn_defaults()
ntb_utils.uncompressed_tables()


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
# # Comparisons

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
sns.boxplot(
    new_glb_evals,
    x="normalized_score",
    y="model",
    order=new_glb_evals.groupby("model")["normalized_score"].mean().sort_values().index,
    showmeans=True,
    meanprops={"markeredgecolor": "black", "fillstyle": "none"},
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

# %% [markdown]
# ## Similarity-based tasks

# %%
sims_new_evals = ntb_utils.load_validation_results(
    "../evaluations/structural_eval_sims/", struct_features
)
sims_new_evals = ntb_utils.add_normalized_score(sims_new_evals)

sims_glb_evals = ntb_utils.load_validation_results(
    "../evaluations/glb_structural_eval_sims/", struct_features
)
sims_glb_evals = ntb_utils.add_normalized_score(sims_glb_evals)


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
    sims_new_evals,
    y="model",
    x="normalized_score",
    hue="metric",
    errorbar=("pi", 100),
    order=sims_new_evals.groupby("model")["normalized_score"]
    .mean()
    .sort_values()
    .index,
)

# %%
sns.barplot(
    sims_glb_evals,
    y="model",
    x="normalized_score",
    hue="metric",
    errorbar=("pi", 100),
    order=sims_glb_evals.groupby("model")["normalized_score"]
    .mean()
    .sort_values()
    .index,
)

# %%
metric = "mean_reciprocal_rank"

# %%
sns.barplot(
    sims_new_evals[sims_new_evals["metric"] == metric],
    y="model",
    x="score",
    hue="task",
    errorbar=("pi", 100),
    order=sims_new_evals[sims_new_evals["metric"] == metric]
    .groupby("model")["score"]
    .mean()
    .sort_values()
    .index,
)
plt.title(metric + ", glb")

# %%
sns.barplot(
    sims_glb_evals[sims_glb_evals["metric"] == metric],
    y="model",
    x="score",
    hue="task",
    errorbar=("pi", 100),
    order=sims_glb_evals[sims_glb_evals["metric"] == metric]
    .groupby("model")["score"]
    .mean()
    .sort_values()
    .index,
)
plt.title(metric + ", no glb")

# %% [markdown]
# ---
#
# > **Outdated**

# %%
matches = permutation_testing(ret_evals)
res = matches.groupby("model")[["won", "won_by"]].mean().reset_index()

# %%
sns.barplot(res, y="model", x="won_by", order=res.sort_values("won_by")["model"])

# %%
sns.barplot(res, y="model", x="won", order=res.sort_values("won")["model"])

# %% [markdown]
# # Paper graphs

# %%
new_glb_evals = new_glb_evals[new_glb_evals["metric"] == "accuracy"]
new_glb_evals = ntb_utils.add_normalized_score(new_glb_evals)

new_glb_evals["baseline"] = new_glb_evals["model"].isin(["longformer", "sbert"])
new_glb_evals["model_type_nice"] = new_glb_evals["baseline"].apply(
    lambda b: "Baseline" if b else "Student"
)

nice_loss = {
    "mse": "MSE",
    "cos_dist": "cosine",
    "contrastive": "contrastive",
    "max_marginals": "max-margin",
}

new_glb_evals["loss_type_nice"] = new_glb_evals.apply(
    lambda row: "Baseline"
    if row["baseline"]
    else nice_loss[row["loss_type"]].capitalize(),
    axis=1,
)

new_glb_evals.loc[new_glb_evals["loss_type_nice"] == "Mse", "loss_type_nice"] = "MSE"


def model_nice_str(model_str):
    if model_str == "longformer":
        return model_str.capitalize()
    if model_str == "sbert":
        return model_str.upper()

    features = struct_features(model_str)

    if features["loss_type"] == "max_marginals":
        return (
            f"max-margin;{nice_loss[features['mm_loss_type']]}"
            r";$\gamma$="
            f"{features['mm_lam']}"
        )

    return f"{nice_loss[features['loss_type']]}"


new_glb_evals["model_nice"] = new_glb_evals["model"].apply(model_nice_str)

# %%
sns.boxplot(
    new_glb_evals,
    x="normalized_score",
    y="model_nice",
    hue="baseline",
    order=new_glb_evals.groupby("model_nice")["normalized_score"]
    .mean()
    .sort_values()
    .index,
    **ntb_utils.boxplot_kwargs,
)

# %% [markdown]
# ## Only simple losses

# %%
simple_losses = new_glb_evals[
    new_glb_evals["baseline"] | new_glb_evals["loss_type"].isin(["mse", "cos_dist"])
]
simple_losses = ntb_utils.add_normalized_score(simple_losses)

# %%
fig, ax = plt.subplots(figsize=(8, 5))

sns.boxplot(
    simple_losses,
    x="normalized_score",
    y="model_nice",
    hue="model_type_nice",
    hue_order=["Student", "Baseline"],
    order=simple_losses.groupby("model_nice")["normalized_score"]
    .mean()
    .sort_values()
    .index,
    **ntb_utils.boxplot_kwargs,
)

sns.move_legend(ax, "lower left")

ax.set_xlabel("Normalized accuracy")
ax.set_ylabel("Model")
ax.get_legend().set_title("")

# %%
fig.savefig("../paper/img/structural_simple_losses.pdf", bbox_inches="tight")

# %% [markdown]
# ## Both losses

# %%
fig, ax = plt.subplots(figsize=(8, 7))

sns.boxplot(
    new_glb_evals,
    x="normalized_score",
    y="model_nice",
    hue="loss_type_nice",
    order=new_glb_evals.groupby("model_nice")["normalized_score"]
    .mean()
    .sort_values()
    .index,
    **ntb_utils.boxplot_kwargs,
)

ax.set_ylabel("Model")
ax.set_xlabel("Normalized accuracy")
ax.get_legend().set_title("Loss")

# %%
fig.savefig("../paper/img/structural_both_losses.pdf", bbox_inches="tight")
