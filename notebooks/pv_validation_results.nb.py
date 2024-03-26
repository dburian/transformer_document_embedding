# %% [markdown]
# # Validation evaluation of PVs

# %%
import pandas as pd
from transformer_document_embedding import notebook_utils as ntb_utils
import seaborn as sns
import matplotlib.pyplot as plt

# %%
plt.rc("figure", figsize=(15, 8))


# %%
pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_colwidth", 500)

sns.set_theme("paper", "whitegrid")
sns.set_palette("muted")

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


# %%
dbow_evals = ntb_utils.load_validation_results(
    "../evaluations/pv_dbow_gs_eval/", pv_features
)

# %%
dbow_evals

# %%
dm_evals = ntb_utils.load_validation_results(
    "../evaluations/pv_dm_gs_eval/", pv_features
)

# %%
pv_evals = ntb_utils.load_validation_results(
    "../evaluations/pv_best_eval/", concat_pv_features
)

# %%
pv_evals

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
dbow_pres_evals = dbow_evals[
    (
        # dbow_evals["larger_train_split"]
        # & (dbow_evals["train_epochs"] == 10)
        dbow_evals["metric"]
        == "accuracy"
    )
]

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
dm_pres_evals = dm_evals[
    (
        dm_evals["metric"]
        == "accuracy"
        # & dm_evals["larger_train_split"]
        # & (dm_evals["train_epochs"] == 10)
    )
]

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
best_dbow_models = (
    pres_dbow.groupby("model")["normalized_score"].mean().sort_values().index[-3:]
)

# %%
pres_dm = ntb_utils.add_normalized_score(dm_evals[dm_evals["metric"] == "accuracy"])
pres_dm["model"] = "dm_" + pres_dm["model"]
best_dm_models = (
    pres_dm.groupby("model")["normalized_score"].mean().sort_values().index[-3:]
)

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
ax = sns.barplot(
    best_models,
    x="score",
    y="model",
    hue="task",
    order=best_models.groupby("model")["score"].mean().sort_values().index,
)
sns.move_legend(ax, "center left")

# %% [markdown]
# # PV

# %%
pv_pres_evals = pv_evals[pv_evals["metric"] == "accuracy"]
pv_pres_evals = ntb_utils.add_normalized_score(pv_pres_evals)

# %%
sns.barplot(
    pv_pres_evals,
    x="normalized_score",
    y="model",
    order=pv_pres_evals.groupby("model")["normalized_score"].mean().sort_values().index,
)

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
