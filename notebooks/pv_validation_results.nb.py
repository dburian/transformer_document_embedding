# %% [markdown]
# # Validation evaluation of PVs

# %%
import pandas as pd
from transformer_document_embedding import notebook_utils as ntb_utils
import seaborn as sns
import matplotlib.pyplot as plt

# %%
ntb_utils.uncompressed_tables()
ntb_utils.seaborn_defaults()

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

    if segments[-1].startswith("after_epoch_"):
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


def concat_pv_features(model_str: str) -> dict[str, Any]:
    dbow_start_idx = model_str.find("-dbow=")
    dm_str = model_str[len("dm=") : dbow_start_idx]
    dbow_str = model_str[len("-dbow=") + dbow_start_idx :]

    return (
        {f"dm_{name}": value for name, value in pv_features(dm_str).items()}
        | {f"dbow_{name}": value for name, value in pv_features(dbow_str).items()}
        | {
            "dm_model": dm_str,
            "dbow_model": dbow_str,
        }
    )


def final_pv_features(model_str: str) -> dict[str, Any]:
    feats = {
        "dim": 1024 if "100d" not in model_str else 100,
        "arch": "dbow" if "dbow" in model_str else "dm",
    }

    return feats


# %%
dbow_evals = ntb_utils.load_validation_results(
    "../evaluations/pv_dbow_gs_eval/", pv_features
)

# %%
dbow_evals = dbow_evals[
    dbow_evals["larger_train_split"] & (dbow_evals["train_epochs"] == 10)
]

# %%
dm_evals = ntb_utils.load_validation_results(
    "../evaluations/pv_dm_gs_eval/", pv_features
)

# %%
dm_evals = dm_evals[dm_evals["larger_train_split"] & (dm_evals["train_epochs"] == 10)]

# %%
pv_evals = ntb_utils.load_validation_results(
    "../evaluations/pv_best_eval/", concat_pv_features
)

# %%

final_evals = ntb_utils.load_validation_results(
    "../evaluations/final_pv", final_pv_features
)

# %% [markdown]
# # DBOW

# %% [markdown]
# ## Best models
#
# Which are the best models?

# %% [markdown]
# ### Presentable results
#
# Classification, big train split, 10 epochs, normalized value, accuracy

# %%
dbow_pres_evals = dbow_evals[dbow_evals["metric"] == "accuracy"]

# %%
dbow_pres_evals = ntb_utils.add_normalized_score(dbow_pres_evals)

# %%
sns.barplot(
    dbow_pres_evals,
    y="model",
    x="normalized_score",
    order=dbow_pres_evals.groupby("model")["normalized_score"]
    .mean()
    .sort_values()
    .index,
    errorbar=("pi", 50),
)

# %%
sns.barplot(
    dbow_pres_evals,
    y="model",
    x="score",
    order=dbow_pres_evals.groupby("model")["score"].mean().sort_values().index,
    errorbar=("pi", 50),
)

# %%
plt.figure(figsize=(12, 8))
sns.barplot(
    dbow_pres_evals,
    y="model",
    hue="task",
    x="normalized_score",
    order=dbow_pres_evals.groupby("model")["normalized_score"]
    .mean()
    .sort_values()
    .index,
    errorbar=("pi", 50),
)

# %%
plt.figure(figsize=(12, 8))
sns.barplot(
    dbow_pres_evals,
    y="model",
    hue="task",
    x="score",
    order=dbow_pres_evals.groupby("model")["score"].mean().sort_values().index,
    errorbar=("pi", 50),
)

# %%
ax = sns.barplot(
    dbow_pres_evals,
    hue="model",
    x="score",
    y="task",
    hue_order=dbow_pres_evals.groupby("model")["score"]
    .mean()
    .sort_values()
    .index[-10:],
    errorbar=("pi", 50),
)
sns.move_legend(ax, "center left")

# %% [markdown]
# # DM

# %% [markdown]
# ## Best models
#
# Which are the best models?

# %% [markdown]
# ### Presentable results
#
# Classification, big train split, 10 epochs, normalized value, accuracy

# %%
dm_pres_evals = dm_evals[dm_evals["metric"] == "accuracy"]

# %%
dm_pres_evals = ntb_utils.add_normalized_score(dm_pres_evals)

# %%
sns.barplot(
    dm_pres_evals,
    y="model",
    x="normalized_score",
    order=dm_pres_evals.groupby("model")["normalized_score"].mean().sort_values().index,
    errorbar=("pi", 50),
)

# %%
sns.barplot(
    dm_pres_evals,
    y="model",
    x="score",
    order=dm_pres_evals.groupby("model")["score"].mean().sort_values().index,
    errorbar=("pi", 50),
)

# %%
sns.barplot(
    dm_pres_evals,
    y="model",
    hue="task",
    x="normalized_score",
    order=dm_pres_evals.groupby("model")["normalized_score"].mean().sort_values().index,
    errorbar=("pi", 50),
)

# %%
sns.barplot(
    dm_pres_evals,
    y="model",
    hue="task",
    x="score",
    order=dm_pres_evals.groupby("model")["score"].mean().sort_values().index,
    errorbar=("pi", 50),
)

# %%
_, ax = plt.subplots(figsize=(12, 10))
sns.barplot(
    dm_pres_evals,
    hue="model",
    x="score",
    y="task",
    hue_order=dm_pres_evals.groupby("model")["score"].mean().sort_values().index[-10:],
    errorbar=("pi", 50),
    ax=ax,
)
sns.move_legend(ax, "center left")

# %% [markdown]
# # Comparison of best DMs and DBOWs

# %%
pres_dbow = ntb_utils.add_normalized_score(
    dbow_evals[dbow_evals["metric"] == "accuracy"]
)
pres_dbow["model"] = "dbow_" + pres_dbow["model"]
pres_dbow["arch"] = "dbow".upper()
best_dbow_models = (
    pres_dbow.groupby("model")["normalized_score"].mean().sort_values().index[-3:]
)

# %%
pres_dm = ntb_utils.add_normalized_score(dm_evals[dm_evals["metric"] == "accuracy"])
pres_dm["model"] = "dm_" + pres_dm["model"]
pres_dm["arch"] = "dm".upper()
best_dm_models = (
    pres_dm.groupby("model")["normalized_score"].mean().sort_values().index[-3:]
)

# %%
pv_pres_evals = pv_evals[pv_evals["metric"] == "accuracy"].copy()
pv_pres_evals["arch"] = "dm+dbow".upper()

# %%
all_models = pd.concat([pres_dm, pres_dbow, pv_pres_evals])
all_models = ntb_utils.add_normalized_score(all_models)


# %%
def presentable_model(model_str):
    if "dm=" in model_str:
        model_feats = concat_pv_features(model_str)
        return "+".join(
            (
                presentable_model(f"{submodel}_" + model_feats[f"{submodel}_model"])
                for submodel in ["dm", "dbow"]
            )
        )

    arch_sep_idx = model_str.find("_")
    arch = model_str[:arch_sep_idx]
    model_str = model_str[arch_sep_idx + 1 :]
    feats = pv_features(model_str)

    min_count = feats["min_count"]
    if min_count == 5000:
        min_count = "10%"
    return f"{arch.upper()};{feats['vector_size']};{feats['pre_process']};{min_count}"


all_models["presentable_model"] = all_models["model"].apply(presentable_model)

# %%
fig, ax = plt.subplots(figsize=(6, 14))
sns.boxplot(
    all_models,
    x="normalized_score",
    y="presentable_model",
    hue="arch",
    order=all_models.groupby("presentable_model")["normalized_score"]
    .mean()
    .sort_values()
    .index,
    **ntb_utils.boxplot_kwargs,
    ax=ax,
)
ax.set_ylabel("Model")
ax.set_xlabel("Normalized accuracy")
ax.get_legend().set(title="Architecture")

# %%
fig.savefig("../paper/img/pv_val_scores.pdf", bbox_inches="tight")

# %%
all_models.groupby("presentable_model")["score"].mean().sort_values()

# %%
fig, ax = plt.subplots(figsize=(10, 8))
sns.violinplot(
    all_models,
    x="normalized_score",
    y="arch",
    order=all_models.groupby("arch")["normalized_score"].mean().sort_values().index,
    ax=ax,
)
ax.set_ylabel("Model")
ax.set_xlabel("Normalized accuracy")
# ax.get_legend().set(title='Architecture')

# %%
best_models = pd.concat(
    [
        pres_dm[pres_dm["model"].isin(best_dm_models)],
        pres_dbow[pres_dbow["model"].isin(best_dbow_models)],
    ]
)
best_models = ntb_utils.add_normalized_score(best_models)

# %%
sns.barplot(
    best_models,
    x="normalized_score",
    y="model",
    order=best_models.groupby("model")["normalized_score"].mean().sort_values().index,
)

# %%
sns.barplot(
    best_models,
    x="score",
    y="model",
    order=best_models.groupby("model")["score"].mean().sort_values().index,
)

# %%
best_model = pres_dbow[
    pres_dbow["model"] == "dbow_m.k.t_p_p=None-m.k.v_s=1024-m.k.m_c=2"
].set_index("task")

# %%
improvements = []

concat_scores = pv_pres_evals[["model", "task", "score"]].set_index(["model", "task"])

for task in best_model.index.get_level_values("task").unique():
    for concat_model in concat_scores.index.get_level_values("model").unique():
        impr = (
            concat_scores.loc[(concat_model, task), "score"]
            - best_model.loc[task, "score"]
        )
        improvements.append(impr)

# %%
pv_pres_evals = ntb_utils.add_normalized_score(pv_pres_evals)

# %%
best_concat_model = pv_pres_evals[
    pv_pres_evals["model"]
    == pv_pres_evals.groupby("model")["normalized_score"].mean().sort_values().index[-1]
].set_index("task")

# %%
improvements = []

for task in best_model.index.get_level_values("task").unique():
    impr = best_concat_model.loc[task, "score"] - best_model.loc[task, "score"]
    improvements.append(impr)

# %%
improvements

# %%
sns.histplot(improvements, binwidth=0.001)

# %%
pd.Series(improvements).mean()

# %% [markdown]
# # PV

# %%
pv_pres_evals = pv_evals[pv_evals["metric"] == "accuracy"]
pv_pres_evals = ntb_utils.add_normalized_score(pv_pres_evals)

# %%
sns.boxplot(
    pv_pres_evals,
    x="normalized_score",
    y="model",
    order=pv_pres_evals.groupby("model")["normalized_score"].mean().sort_values().index,
)

# %%
pv_pres_evals.groupby("model")["normalized_score"].mean().sort_values()

# %%
sns.barplot(
    pv_pres_evals,
    x="normalized_score",
    y="model",
    hue="task",
    order=pv_pres_evals.groupby("model")["normalized_score"].mean().sort_values().index,
)

# %%
sns.barplot(
    pv_pres_evals,
    x="score",
    y="model",
    hue="task",
    order=pv_pres_evals.groupby("model")["score"].mean().sort_values().index,
)

# %%
all_best = pd.concat([pv_pres_evals, best_models])
all_best = ntb_utils.add_normalized_score(all_best)

# %%
sns.barplot(
    all_best,
    x="normalized_score",
    y="model",
    order=all_best.groupby("model")["normalized_score"].mean().sort_values().index,
)

# %%
all_best.groupby("model")["normalized_score"].mean().sort_values()

# %%
sns.barplot(
    all_best,
    x="score",
    y="model",
    order=all_best.groupby("model")["score"].mean().sort_values().index,
)

# %%
sns.barplot(
    all_best,
    x="score",
    y="model",
    hue="task",
    order=all_best.groupby("model")["score"].mean().sort_values().index,
)


# %%
def add_baselines(pv_evals):
    baselines = []
    for dm_model in pv_evals["dm_model"].unique():
        dms = dm_evals[dm_evals["model"] == f"big-{dm_model}-after_epoch_9"].copy()
        print(dms.shape)
        dms["dm_model"] = dms["model"].apply(
            lambda model_str: model_str[len("big-") : -len("-after_epoch_9")]
        )
        dms["model"] = "dm=" + dms["dm_model"]
        dms = dms.rename(
            columns={
                "larger_train_split": "dm_larger_train_split",
                "pre_process": "dm_pre_process",
                "vector_size": "dm_vector_size",
            }
        )
        baselines.append(dms)

    for dbow_model in pv_evals["dbow_model"].unique():
        dbows = dbow_evals[
            dbow_evals["model"] == f"big-{dbow_model}-after_epoch_9"
        ].copy()
        dbows["dbow_model"] = dbows["model"].apply(
            lambda model_str: model_str[len("big-") : -len("-after_epoch_9")]
        )
        dbows["model"] = "dbow=" + dbows["dbow_model"]
        dbows = dbows.rename(
            columns={
                "larger_train_split": "dbow_larger_train_split",
                "pre_process": "dbow_pre_process",
                "vector_size": "dbow_vector_size",
            }
        )
        baselines.append(dbows)

    return pd.concat([pv_evals, *baselines])


# %%
all_evals = add_baselines(pv_evals)

# %%
all_pres_evals = all_evals[all_evals["metric"] == "accuracy"]

# %%
all_pres_evals = ntb_utils.add_normalized_score(all_pres_evals)

# %%
sns.barplot(
    all_pres_evals,
    x="score",
    y="model",
    hue="task",
    order=all_pres_evals.groupby("model")["score"].mean().sort_values().index,
)

# %%
sns.barplot(
    all_pres_evals,
    x="normalized_score",
    y="model",
    hue="task",
    order=all_pres_evals.groupby("model")["normalized_score"]
    .mean()
    .sort_values()
    .index,
)

# %% [markdown]
# - compare how much we gain by combining dbow and dm (ONCE we have the real results)

# %% [markdown]
# # Final PVs
#
# TODO: compare with debug PVs

# %%
final_evals = final_evals[final_evals["metric"] == "accuracy"]
final_evals = ntb_utils.add_normalized_score(final_evals)

# %%
sns.boxplot(
    final_evals,
    x="normalized_score",
    y="model",
    hue="arch",
    order=ntb_utils.model_order(final_evals, "normalized_score"),
    showmeans=True,
    meanprops={"markeredgecolor": "black", "fillstyle": "none", "markeredgewidth": 1},
)

# %%
sns.barplot(
    final_evals,
    x="score",
    y="model",
    hue="task",
    order=ntb_utils.model_order(final_evals, "normalized_score"),
)
