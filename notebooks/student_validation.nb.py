# %% [markdown]
# # Validation evaluation of students

# %%
import pandas as pd
import transformer_document_embedding.notebook_utils as ntb_utils
from typing import Any
import seaborn as sns
import matplotlib.pyplot as plt

# %%
ntb_utils.uncompressed_tables()
ntb_utils.seaborn_defaults()


# %%
def get_nice_projection(relevant_segment: str) -> str:
    projection_str = relevant_segment[
        relevant_segment.find("[") + 1 : relevant_segment.find("]")
    ]
    if len(projection_str) == 0:
        # No projection
        return "-"

    nice = []

    nice_layer = {
        "f": lambda dim: str(dim),
        "a": lambda act: f"({'ReLU' if act == 'relu' else act})",
        "d": lambda drop: f"({float(drop):.1f}d)",
    }

    for block_str in projection_str.split(","):
        block_nice = ""
        for layer_str in block_str.split("-"):
            name, value = layer_str.split("=")
            block_nice += nice_layer[name](value)
        nice.append(block_nice)

    return "x".join(nice)


def student_features(model_str: str) -> dict[str, Any]:
    feats = {
        "structural_loss": "none",
        "contextual_dim": "-",
        "student_projection": "none",
        "contextual_projection": "none",
    }

    if "-" not in model_str:
        # Its a baseline
        for structural_loss_type in ["mm_mse", "cos"]:
            if structural_loss_type in model_str:
                feats["structural_loss"] = structural_loss_type

        return feats

    model_str_segments = model_str.split("-")
    segments = []
    open_bracket_segment = None
    for segment in model_str_segments:
        if open_bracket_segment is None:
            if "[" not in segment or "]" in segment:
                segments.append(segment)
            else:
                open_bracket_segment = segment
        else:
            open_bracket_segment += "-" + segment
            if "]" in segment:
                segments.append(open_bracket_segment)
                open_bracket_segment = None

    for structural_loss_type in ["mm_mse", "cos"]:
        if structural_loss_type in segments[0]:
            feats["structural_loss"] = structural_loss_type

    for contextual_dim in ["100d", "1024d", "2048d"]:
        if contextual_dim in segments[0]:
            feats["contextual_dim"] = int(contextual_dim[:-1])

    feats["student_projection"] = get_nice_projection(segments[1])
    feats["contextual_projection"] = get_nice_projection(segments[2])

    return feats


# %%
evals_100d = ntb_utils.load_validation_results(
    "../evaluations/cca_projections_100d_debug/", student_features
)
evals_1024d = ntb_utils.load_validation_results(
    "../evaluations/cca_projections_1024d_debug", student_features
)
evals_2048d = ntb_utils.load_validation_results(
    "../evaluations/cca_projections_2048d_debug", student_features
)


def only_acc(df):
    return df[df["metric"] == "accuracy"]


evals_100d = only_acc(evals_100d)
evals_1024d = only_acc(evals_1024d)
evals_2048d = only_acc(evals_2048d)

evals = pd.concat([evals_100d, evals_1024d, evals_2048d])
evals = evals[~evals.duplicated()].copy()
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
            lambda lt: lt[len("max_marginals_") :] if "max_marginals" in lt else pd.NA,
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


structural_evals = ntb_utils.load_validation_results(
    "../evaluations/glb_structural_eval/", struct_features
)
structural_evals = structural_evals[structural_evals["metric"] == "accuracy"]

# %%
cosine_structural_evals = structural_evals[
    structural_evals["loss_type"] == "cos_dist"
].copy()
cosine_structural_evals["model_nice"] = "only-structural;cosine"
cosine_structural_evals["contextual_dim"] = "-"

mm_mse_structural_evals = structural_evals[
    (structural_evals["loss_type"] == "max_marginals")
    & (structural_evals["mm_loss_type"] == "mse")
    & (structural_evals["mm_lam"] == 1)
].copy()
mm_mse_structural_evals["model_nice"] = "only-strucutral;mm-MSE"
mm_mse_structural_evals["contextual_dim"] = "-"

# %%
dm_baselines = ntb_utils.load_validation_results("../evaluations/pv_dm_gs_eval/")
dm_baselines = dm_baselines[
    dm_baselines["model"] == "m.k.t_p_p=lowercase-m.k.v_s=100-m.k.m_c=2"
].copy()
dm_baselines["model_nice"] = "DM;100d"
dm_baselines["contextual_dim"] = 100

dbow_baselines = ntb_utils.load_validation_results("../evaluations/pv_dbow_gs_eval/")
dbow_baselines = dbow_baselines[
    dbow_baselines["model"] == "m.k.t_p_p=None-m.k.v_s=1024-m.k.m_c=2"
].copy()
dbow_baselines["model_nice"] = "DBOW;1024d"
dbow_baselines["contextual_dim"] = 1024

pv_baselines = ntb_utils.load_validation_results("../evaluations/pv_best_eval/")
pv_baselines = pv_baselines[
    pv_baselines["model"]
    == "dm=m.k.t_p_p=stem-m.k.v_s=1024-m.k.m_c=2-dbow=m.k.t_p_p=None-m.k.v_s=1024-m.k.m_c=2"
].copy()
pv_baselines["model_nice"] = "PV;2048d"
pv_baselines["contextual_dim"] = 2048

pv_baselines = pd.concat([dm_baselines, dbow_baselines, pv_baselines])
pv_baselines = pv_baselines[pv_baselines["metric"] == "accuracy"]

# %%
baselines = ntb_utils.load_validation_results("../evaluations/baselines_val/")
baselines["contextual_dim"] = "-"
baselines["model_nice"] = baselines["model"].apply(
    lambda m: m.capitalize() if m.startswith("long") else m.upper()
)
baselines = baselines[baselines["metric"] == "accuracy"]

# %% [markdown]
# ---

# %% [markdown]
# # Projection GS

# %% [markdown]
# ## 100d models

# %%
evals_100d = ntb_utils.add_normalized_score(evals_100d)

# %%
sns.boxplot(
    evals_100d,
    x="normalized_score",
    y="model",
    hue="structural_loss",
    dodge=False,
    order=ntb_utils.model_order(evals_100d, "normalized_score"),
    showmeans=True,
    meanprops={"markeredgecolor": "black", "fillstyle": "none"},
)

# %%
sns.swarmplot(
    evals_100d,
    x="score",
    y="model",
    hue="task",
    order=ntb_utils.model_order(evals_100d, "score"),
)

# %% [markdown]
# ## 1024d models

# %%
evals_1024d = ntb_utils.add_normalized_score(evals_1024d)

# %%
sns.boxplot(
    evals_1024d,
    x="normalized_score",
    y="model",
    hue="structural_loss",
    dodge=False,
    order=ntb_utils.model_order(evals_1024d, "normalized_score"),
    showmeans=True,
    meanprops={"markeredgecolor": "black", "fillstyle": "none"},
)

# %%
evals_1024d[evals_1024d["structural_loss"] == "none"].groupby("model")[
    "normalized_score"
].mean().sort_values()

# %%
sns.swarmplot(
    evals_1024d,
    x="score",
    y="model",
    hue="task",
    order=ntb_utils.model_order(evals_1024d, "score"),
)

# %% [markdown]
# ## 2048d models

# %%
evals_2048d = ntb_utils.add_normalized_score(evals_2048d)

# %%
sns.boxplot(
    evals_2048d,
    x="normalized_score",
    y="model",
    hue="structural_loss",
    dodge=False,
    order=ntb_utils.model_order(evals_2048d, "normalized_score"),
    showmeans=True,
    meanprops={"markeredgecolor": "black", "fillstyle": "none"},
)

# %%
evals_2048d.groupby("model")["normalized_score"].mean().sort_values()

# %%
sns.swarmplot(
    evals_2048d,
    x="score",
    y="model",
    hue="task",
    order=ntb_utils.model_order(evals_2048d, "score"),
)

# %%
tmp = evals_2048d[evals_2048d["structural_loss"] == "cos"]

sns.boxplot(
    tmp,
    x="normalized_score",
    y="model",
    hue="structural_loss",
    dodge=False,
    order=ntb_utils.model_order(tmp, "normalized_score"),
    showmeans=True,
    meanprops={"markeredgecolor": "black", "fillstyle": "none"},
)

# %% [markdown]
# ## All models

# %%
evals = ntb_utils.add_normalized_score(evals)

evals["loss"] = evals["contextual_dim"].astype(str) + "-" + evals["structural_loss"]
evals["contextual_dim_cat"] = evals["contextual_dim"].astype(str)

# %%
_, ax = plt.subplots(figsize=(8, 12))

sns.boxplot(
    evals,
    x="normalized_score",
    y="model",
    hue="contextual_dim_cat",
    dodge=False,
    order=ntb_utils.model_order(evals, "normalized_score"),
    showmeans=True,
    meanprops={"markeredgecolor": "black", "fillstyle": "none", "markeredgewidth": 0.5},
    ax=ax,
)

# %%
sns.boxplot(
    evals,
    x="normalized_score",
    y="model",
    hue="contextual_dim_cat",
    dodge=False,
    order=ntb_utils.model_order(evals, "normalized_score")[-30:],
    showmeans=True,
    meanprops={"markeredgecolor": "black", "fillstyle": "none", "markeredgewidth": 0.6},
)

# %% [markdown]
# ### with 'mse'

# %%
mse_evals = evals[evals["structural_loss"] != "cos"]

sns.boxplot(
    mse_evals,
    x="normalized_score",
    y="model",
    hue="contextual_dim_cat",
    dodge=False,
    order=ntb_utils.model_order(mse_evals, "normalized_score")[-30:],
    showmeans=True,
    meanprops={"markeredgecolor": "black", "fillstyle": "none", "markeredgewidth": 0.6},
)

# %%
mse_evals.groupby("model")["normalized_score"].mean().sort_values()[-30:]

# %%
sns.swarmplot(
    mse_evals,
    x="score",
    y="model",
    hue="contextual_dim_cat",
    dodge=False,
    order=ntb_utils.model_order(mse_evals, "score")[-30:],
)

# %% [markdown]
# ### with 'cos'

# %%
cos_evals = evals[evals["structural_loss"] != "mm_mse"]

# %%
sns.boxplot(
    cos_evals,
    x="normalized_score",
    y="model",
    hue="contextual_dim_cat",
    dodge=False,
    order=ntb_utils.model_order(cos_evals, "normalized_score")[-30:],
    showmeans=True,
    meanprops={"markeredgecolor": "black", "fillstyle": "none", "markeredgewidth": 0.6},
)

# %%
cos_evals.groupby("model")["normalized_score"].mean().sort_values()[-30:]

# %%
sns.swarmplot(
    cos_evals,
    x="score",
    y="model",
    hue="contextual_dim_cat",
    dodge=False,
    order=ntb_utils.model_order(cos_evals, "score")[-30:],
)

# %%
evals_2048d.groupby("model")["normalized_score"].mean().sort_values()

# %%
sns.barplot(
    evals_2048d,
    x="score",
    y="model",
    hue="task",
    order=ntb_utils.model_order(evals_2048d, "score"),
)


# %% [markdown]
# ## Figures


# %%
def get_nice_model(model_str: str) -> str:
    feats = student_features(model_str)

    if feats["contextual_dim"] == "-":
        if model_str == "longformer":
            return model_str.capitalize()
        if model_str == "sbert":
            return model_str.upper()

    projection_nice = (
        f"S:{feats['student_projection']};C:{feats['contextual_projection']}"
    )

    return f"{feats['contextual_dim']}d;{projection_nice}"


# %% [markdown]
# ### Only contextual

# %%
only_ctx = evals[evals["structural_loss"] == "none"].copy()
only_ctx["model_nice"] = only_ctx["model"].apply(get_nice_model)

# %%
only_ctx = pd.concat(
    [
        baselines,
        only_ctx,
        pv_baselines,
        cosine_structural_evals,
    ]
)
only_ctx = ntb_utils.add_normalized_score(only_ctx)

# %%
fig, ax = plt.subplots(figsize=(6, 7))

sns.boxplot(
    only_ctx,
    y="model_nice",
    x="normalized_score",
    hue="contextual_dim",
    order=ntb_utils.model_order(only_ctx, by="normalized_score", of="model_nice"),
    **ntb_utils.boxplot_kwargs,
    ax=ax,
)

ax.get_legend().set_title("Contextual embedding dim.")
ax.set_ylabel("Model")
ax.set_xlabel("Normalized accuracy")

# %%
fig.savefig("../paper/img/projections_contextual.pdf", bbox_inches="tight")

# %% [markdown]
# ### Cosine

# %%
cos = evals[evals["structural_loss"] == "cos"].copy()
cos["model_nice"] = cos["model"].apply(get_nice_model)

# %%
del_100d = (cos["contextual_dim"] == 100) & (cos["student_projection"] == "768")
del_1024d = (cos["contextual_dim"] == 1024) & (
    (cos["student_projection"] == "1024")
    | (cos["contextual_projection"].isin(["768x1024", "1024(ReLU)x1024"]))
)
del_2048d = (cos["contextual_dim"] == 2048) & (
    (cos["student_projection"] == "2048")
    | (cos["contextual_projection"] == "1024x2048")
    | (cos["student_projection"].str.contains("d"))
)

# %%
cos = cos[~(del_100d | del_1024d | del_2048d)]

# %%
cos = pd.concat([baselines, cos, pv_baselines, cosine_structural_evals])
cos = ntb_utils.add_normalized_score(cos)

# %%
fig, ax = plt.subplots(figsize=(6, 6.5))

sns.boxplot(
    cos,
    y="model_nice",
    x="normalized_score",
    hue="contextual_dim",
    order=ntb_utils.model_order(cos, by="normalized_score", of="model_nice"),
    **ntb_utils.boxplot_kwargs,
    ax=ax,
)

ax.get_legend().set_title("Contextual embedding dim.")
ax.set_ylabel("Model")
ax.set_xlabel("Normalized accuracy")

# %%
fig.savefig("../paper/img/projections_contextual_cos.pdf", bbox_inches="tight")

# %% [markdown]
# ### MSE

# %%
mse = evals[evals["structural_loss"] == "mm_mse"].copy()
mse["model_nice"] = mse["model"].apply(get_nice_model)

# %%
del_1024 = mse.student_projection == "1024"
del_2048 = (mse.student_projection == "2048") | (
    (mse.student_projection == "768(ReLU)x4096(ReLU)x2048")
    & (mse.contextual_projection == "2048")
)

# %%
mse = mse[~(del_1024 | del_2048)]

# %%
mse = pd.concat(
    [
        baselines,
        pv_baselines,
        mse,
        mm_mse_structural_evals,
    ]
)
mse = ntb_utils.add_normalized_score(mse)


# %%
fig, ax = plt.subplots(figsize=(6, 6.5))

sns.boxplot(
    mse,
    y="model_nice",
    x="normalized_score",
    hue="contextual_dim",
    order=ntb_utils.model_order(mse, by="normalized_score", of="model_nice"),
    **ntb_utils.boxplot_kwargs,
    ax=ax,
)

ax.get_legend().set_title("Contextual embedding dim.")
ax.set_ylabel("Model")
ax.set_xlabel("Normalized accuracy")

# %%
fig.savefig("../paper/img/projections_contextual_mm_mse.pdf", bbox_inches="tight")

# %% [markdown]
# # Weighting GS


# %%
def weighting_feats(model_str: str) -> dict[str, Any]:
    if "-" not in model_str:
        return {
            "baseline": True,
            "structural_loss": "none",
            "contextual_dim": 0,
            "lam": 0,
            "max_structural_len": None,
        }
    segments = model_str.split("-")

    feats = {}
    feats["structural_loss"] = "cos" if segments[0].startswith("cos") else "mm_mse"
    segments[0] = segments[0][len(feats["structural_loss"]) + 1 :]

    feats["contextual_dim"] = segments[0][: segments[0].find("_") - 1]
    del segments[0]

    props = {
        "h.k.l": "lam",
        "h.k.m_s_l": "max_structural_len",
    }
    for segment in segments:
        prop, value = segment.split("=")
        feats[props[prop]] = value

    return feats


# %%
cos_weight = ntb_utils.load_validation_results(
    "../evaluations/cos_weighting/", weighting_feats
)
cos_weight = pd.concat(
    [
        cos_weight,
        baselines,
        cosine_structural_evals,
    ]
)
cos_weight = cos_weight[cos_weight["metric"] == "accuracy"]
cos_weight = ntb_utils.add_normalized_score(cos_weight)

# %%
mm_mse_weight = ntb_utils.load_validation_results(
    "../evaluations/mm_mse_weighting/", weighting_feats
)
mm_mse_weight = pd.concat(
    [
        mm_mse_weight,
        baselines,
        mm_mse_structural_evals,
    ]
)
mm_mse_weight = mm_mse_weight[mm_mse_weight["metric"] == "accuracy"]
mm_mse_weight = ntb_utils.add_normalized_score(mm_mse_weight)

# %% [markdown]
# ## Cos

# %%
sns.boxplot(
    cos_weight,
    x="normalized_score",
    y="model",
    hue="contextual_dim",
    order=ntb_utils.model_order(cos_weight, "normalized_score"),
    **ntb_utils.boxplot_kwargs,
)

# %%
sns.barplot(
    cos_weight,
    x="score",
    y="model",
    hue="task",
    order=ntb_utils.model_order(cos_weight, "normalized_score"),
)

# %% [markdown]
# ## MM_MSE

# %%
sns.boxplot(
    mm_mse_weight,
    x="normalized_score",
    y="model",
    hue="contextual_dim",
    order=ntb_utils.model_order(mm_mse_weight, "normalized_score"),
    **ntb_utils.boxplot_kwargs,
)

# %%
sns.barplot(
    mm_mse_weight,
    x="score",
    y="model",
    hue="task",
    order=ntb_utils.model_order(mm_mse_weight, "normalized_score"),
)

# %% [markdown]
# ## All

# %%
all = pd.concat([cos_weight, mm_mse_weight])
all = all.drop_duplicates(["model", "task"])

all = ntb_utils.add_normalized_score(all)

# %%
sns.boxplot(
    all,
    x="normalized_score",
    y="model",
    hue="contextual_dim",
    order=ntb_utils.model_order(all, "normalized_score"),
    **ntb_utils.boxplot_kwargs,
)


# %% [markdown]
# ## Figures


# %%
def weight_nice_model(model_str: str) -> str:
    feats = weighting_feats(model_str)

    return f"{feats['max_structural_len']};" r"$\lambda$=" f"{feats['lam']}"


# %% [markdown]
# ### Cos

# %%
cos_weight = ntb_utils.load_validation_results(
    "../evaluations/cos_weighting/", weighting_feats
)
cos_weight = cos_weight[cos_weight["contextual_dim"] == "2048"].copy()
cos_weight["model_nice"] = cos_weight["model"].apply(weight_nice_model)

# %%
cos = evals[evals["structural_loss"] == "cos"].copy()
cos["model_nice"] = cos["model"].apply(get_nice_model)
cos = cos[cos["model_nice"] == "2048d;S:768(ReLU)x4096(ReLU)x2048;C:-"]
cos["model_nice"] = "no-weighting"
cos["lam"] = "-"

# %%
baselines["lam"] = "-"
cosine_structural_evals["lam"] = "1.0"
pv_baselines["lam"] = "-"

cos_weight = pd.concat(
    [
        cos_weight,
        baselines,
        cos,
        pv_baselines[pv_baselines["contextual_dim"] == 2048],
        cosine_structural_evals,
    ]
)
cos_weight = cos_weight[cos_weight["metric"] == "accuracy"]
cos_weight = ntb_utils.add_normalized_score(cos_weight)

# %%
fig, ax = plt.subplots(figsize=(7, 6))

all_colors = sns.color_palette()
light_palette = sns.light_palette(all_colors[1], as_cmap=True)
lams = cos_weight["lam"].sort_values().unique()

sns.boxplot(
    cos_weight,
    y="model_nice",
    x="normalized_score",
    hue="lam",
    hue_order=lams,
    order=ntb_utils.model_order(cos_weight, by="normalized_score", of="model_nice"),
    **ntb_utils.boxplot_kwargs,
    ax=ax,
    palette=[all_colors[0], *[light_palette(float(lam)) for lam in lams[1:]]],
)

ax.get_legend().set_title(r"$\lambda$")
ax.set_ylabel("Model")
ax.set_xlabel("Normalized accuracy")

# %%
fig.savefig("../paper/img/cos_weighting.pdf", bbox_inches="tight")

# %% [markdown]
# ### MM MSE

# %%
mm_mse_weight = ntb_utils.load_validation_results(
    "../evaluations/mm_mse_weighting/", weighting_feats
)
mm_mse_weight = mm_mse_weight[mm_mse_weight["contextual_dim"] == "100"].copy()
mm_mse_weight["model_nice"] = mm_mse_weight["model"].apply(weight_nice_model)

# %%
mse = evals[evals["structural_loss"] == "mm_mse"].copy()
mse["model_nice"] = mse["model"].apply(get_nice_model)
mse = mse[mse["model_nice"] == "100d;S:768(ReLU)x1024(ReLU)x768;C:768"]
mse["model_nice"] = "no-weighting"
mse["lam"] = "-"

# %%
baselines["lam"] = "-"
mm_mse_structural_evals["lam"] = "1.0"

mm_mse_weight = pd.concat(
    [
        mm_mse_weight,
        baselines,
        pv_baselines[pv_baselines["contextual_dim"] == 100],
        mm_mse_structural_evals,
        mse,
    ]
)
mm_mse_weight = mm_mse_weight[mm_mse_weight["metric"] == "accuracy"]
mm_mse_weight = ntb_utils.add_normalized_score(mm_mse_weight)

# %%
fig, ax = plt.subplots(figsize=(7, 6))

all_colors = sns.color_palette()
light_palette = sns.light_palette(all_colors[1], as_cmap=True)
lams = cos_weight["lam"].sort_values().unique()


sns.boxplot(
    mm_mse_weight,
    y="model_nice",
    x="normalized_score",
    hue="lam",
    hue_order=lams,
    order=ntb_utils.model_order(mm_mse_weight, by="normalized_score", of="model_nice"),
    **ntb_utils.boxplot_kwargs,
    ax=ax,
    palette=[all_colors[0], *[light_palette(float(lam)) for lam in lams[1:]]],
)

ax.get_legend().set_title(r"$\lambda$")
ax.set_ylabel("Model")
ax.set_xlabel("Normalized accuracy")

# %%
fig.savefig("../paper/img/mm_mse_weighting.pdf", bbox_inches="tight")

# %% [markdown]
# # Final experiments comparison

# %%
cos_masked = cos_weight[cos_weight["model_nice"] == "384;" + r"$\lambda$=0.5"].copy()
cos_masked["model_nice"] = "masked-cosine;" r"$\lambda$=0.5"

cos_sum = evals[evals["structural_loss"] == "cos"].copy()
cos_sum["model_nice"] = cos_sum["model"].apply(get_nice_model)
cos_sum = cos_sum[cos_sum["model_nice"] == "2048d;S:768(ReLU)x4096(ReLU)x2048;C:-"]
cos_sum["model_nice"] = "cosine;no-weighting"

best_cos = pd.concat(
    [
        cos_masked,
        cos_sum,
    ]
)

# %%
mm_mse_contextual = mm_mse_weight[
    mm_mse_weight["model_nice"] == "None;" + r"$\lambda$=0.5"
].copy()
mm_mse_contextual["model_nice"] = "mm-MSE;" r"$\lambda$=0.5"

best_mm_mse = pd.concat(
    [
        mm_mse_contextual,
        mm_mse_structural_evals,
    ]
)

# %%
best_models = pd.concat(
    [
        best_cos,
        best_mm_mse,
        baselines,
        pv_baselines[pv_baselines["contextual_dim"].isin([100, 2048])],
    ]
)
best_models = best_models[best_models["metric"] == "accuracy"]
best_models = ntb_utils.add_normalized_score(best_models)

# %%

fig, axes = plt.subplots(
    2, 1, figsize=(9, 9), gridspec_kw={"height_ratios": [0.7, 0.3], "hspace": 0.3}
)

sns.barplot(
    best_models,
    x="score",
    y="model_nice",
    hue="task",
    order=ntb_utils.model_order(best_models, of="model_nice"),
    ax=axes[0],
)


sns.boxplot(
    best_models,
    x="normalized_score",
    y="model_nice",
    order=ntb_utils.model_order(best_models, "normalized_score", of="model_nice"),
    **ntb_utils.boxplot_kwargs,
    ax=axes[1],
)

axes[0].get_legend().set_title("Task")
# sns.move_legend(axes[0], 'upper left')

axes[0].set_ylabel("Model")
axes[1].set_ylabel("Model")

axes[0].set_xlabel("Accuracy")
axes[1].set_xlabel("Normalized accuracy")

# %%

fig, ax = plt.subplots(figsize=(8, 5.5))


sns.boxplot(
    best_models,
    x="normalized_score",
    y="model_nice",
    order=ntb_utils.model_order(best_models, "normalized_score", of="model_nice"),
    **ntb_utils.boxplot_kwargs,
    ax=ax,
)

ax.set_ylabel("Model")
ax.set_xlabel("Normalized accuracy")

# %%
fig.savefig("../paper/img/experiments_final_models.pdf", bbox_inches="tight")
