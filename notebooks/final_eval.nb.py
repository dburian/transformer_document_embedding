# %% [markdown]
# # Final evaluation

# %%
from datasets import Dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformer_document_embedding import notebook_utils as ntb_utils
import pandas as pd
import seaborn as sns
from transformer_document_embedding.datasets.document_dataset import DocumentDataset
from transformer_document_embedding.models.embedding_model import EmbeddingModel

from transformer_document_embedding.pipelines.classification_finetune import (
    get_head_features,
)
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from typing import Any, Callable, NamedTuple

# %%
ntb_utils.uncompressed_tables()
ntb_utils.seaborn_defaults()


# %%
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
def gen_nice_model(model_str):
    presets = {
        "longformer": "Longformer",
        "sbert": "SBERT",
        "cos_final": "cosine-masked",
        "dm_100d_train_aic-after_epoch_3": "DM",
        "pv_2048d": "PV",
        "just_mm_mse_final": "MSE-contextual",
        "mm_mse_contextual_final": "only-MSE",
    }

    return presets.get(model_str, model_str)


def gen_nice_metric(metric):
    presets = {
        "accuracy": "Accuracy",
        "mean_reciprocal_rank": "MRR",
        "map": "MAP",
    }

    return presets.get(metric, metric)


considered_models = [
    "longformer",
    "sbert",
    "cos_final",
    "dm_100d_train_aic-after_epoch_3",
    "pv_2048d",
    "just_mm_mse_final",
    "mm_mse_contextual_final",
]


def parse_plot_data(df):
    df = df[df["model"].isin(considered_models)].copy()

    df["model_nice"] = df["model"].apply(gen_nice_model)
    df["metric_nice"] = df["metric"].apply(gen_nice_metric)

    df_acc = df[df["metric"] == "accuracy"]
    df_sim = df[df["metric"].isin(["mean_reciprocal_rank", "map"])]

    df_acc = ntb_utils.add_normalized_score(df_acc)

    cat_type = pd.CategoricalDtype(
        ntb_utils.model_order(df_acc, "normalized_score", "model_nice"), ordered=True
    )
    df_acc["model_nice"] = df_acc["model_nice"].astype(cat_type)

    return df_acc, df_sim


def gen_acc_plot(df, fig=None, ax=None):
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    sns.lineplot(
        df,
        x="model_nice",
        y="score",
        hue="task",
        orient="x",
        ax=ax,
    )
    sns.scatterplot(
        df,
        x="model_nice",
        y="score",
        hue="task",
        ax=ax,
    )

    ax.get_legend().remove()
    tasks = df["task"].unique()
    ax.legend(
        [
            Line2D([], [], color=c, linestyle="-", marker="o")
            for c in sns.color_palette(n_colors=len(tasks))
        ],
        tasks,
        title="Tasks",
    )
    ax.tick_params("x", labelrotation=45)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")

    return fig, ax


def gen_norm_acc_plot(df, fig=None, ax=None):
    if fig is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))

    sns.boxplot(
        df,
        x="normalized_score",
        y="model_nice",
        order=ntb_utils.model_order(df, "normalized_score", "model_nice"),
        **ntb_utils.boxplot_kwargs,
    )
    ax.set_ylabel("Model")
    ax.set_xlabel("Normalized accuracy")

    return fig, ax


# %% [markdown]
# # Minimal evals
#
# One layer classifier

# %%
evals_min = ntb_utils.load_validation_results("../evaluations/final_eval_min")

evals_min = ntb_utils.add_normalized_score(evals_min)
evals_min_acc = evals_min[evals_min["metric"] == "accuracy"]
evals_min = evals_min[
    evals_min["metric"].isin(["accuracy", "mean_reciprocal_rank", "map"])
]

# %%
sns.boxplot(
    evals_min_acc,
    x="normalized_score",
    y="model",
    order=ntb_utils.model_order(evals_min_acc, "normalized_score"),
    **ntb_utils.boxplot_kwargs,
)

# %%
sns.barplot(
    evals_min,
    y="model",
    x="score",
    hue="task",
)

# %% [markdown]
# # 1k evals
#
# 1k training docs

# %%
evals_1k = ntb_utils.load_validation_results("../evaluations/final_eval_1k")

acc_1k, sim_1k = parse_plot_data(evals_1k)

# %%
fig, ax = gen_norm_acc_plot(acc_1k)

# %%
fig.savefig("../paper/img/1k_norm_evals.pdf", bbox_inches="tight")

# %%
fig, ax = gen_acc_plot(acc_1k)

# %%
fig.savefig("../paper/img/1k_evals.pdf", bbox_inches="tight")

# %%
sns.barplot(
    acc_1k,
    x="model_nice",
    y="score",
    hue="task",
)

# %%
sns.barplot(
    evals_1k,
    y="model",
    x="score",
    hue="metric",
)

# %% [markdown]
# # Short evals
#
# 10k training docs for all tasks

# %%
short_evals = ntb_utils.load_validation_results("../evaluations/final_eval_short")

acc_short, sim_short = parse_plot_data(short_evals)

# %%
fig, ax = gen_norm_acc_plot(acc_short)

# %%
fig.savefig("../paper/img/short_norm_evals.pdf", bbox_inches="tight")

# %%
fig, ax = gen_acc_plot(acc_short)

# %%
fig.savefig("../paper/img/short_evals.pdf", bbox_inches="tight")

# %%
sns.boxplot(
    short_evals_acc,
    x="normalized_score",
    y="model",
    order=ntb_utils.model_order(short_evals_acc, "normalized_score"),
    **ntb_utils.boxplot_kwargs,
)

# %%
sns.barplot(
    short_evals,
    y="model",
    x="score",
    hue="metric",
)

# %% [markdown]
# # Old evals
#
# Old logging of loss

# %%
old_evals = ntb_utils.load_validation_results("../evaluations/final_eval_old/")
old_evals = ntb_utils.add_normalized_score(old_evals)
old_evals_acc = old_evals[old_evals["metric"] == "accuracy"]
old_evals = old_evals[
    old_evals["metric"].isin(["accuracy", "map", "mean_reciprocal_rank"])
]

# %%
sns.boxplot(
    old_evals_acc,
    x="normalized_score",
    y="model",
    order=ntb_utils.model_order(old_evals_acc, "normalized_score"),
    **ntb_utils.boxplot_kwargs,
)

# %% [markdown]
# # Final evals

# %%
final_evals = ntb_utils.load_validation_results("../evaluations/final_eval/")

final_acc, final_sim = parse_plot_data(final_evals)

# %%
fig, ax = gen_norm_acc_plot(final_acc)

# %%
fig.savefig("../paper/img/final_norm_evals.pdf", bbox_inches="tight")

# %%
fig, ax = gen_acc_plot(final_acc)

# %%
fig.savefig("../paper/img/final_evals.pdf", bbox_inches="tight")

# %%
sns.boxplot(
    final_evals_acc,
    y="model",
    x="normalized_score",
    order=ntb_utils.model_order(final_evals_acc, by="normalized_score"),
    **ntb_utils.boxplot_kwargs,
)

# %%
sns.barplot(
    final_evals_acc,
    y="model",
    x="score",
    hue="task",
    order=ntb_utils.model_order(final_evals_acc, by="score"),
)

# %%
sns.barplot(
    final_evals,
    y="task",
    x="score",
    hue="model",
)

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))

sns.barplot(
    final_sim,
    y="metric_nice",
    x="score",
    hue="model_nice",
    hue_order=final_sim[final_sim["metric"] == "map"]
    .groupby("model_nice")["score"]
    .mean()
    .sort_values()
    .index,
    errorbar=("pi", 100),
    ax=ax,
)
ax.set_ylabel("Metric")
ax.set_xlabel("Score")
ax.get_legend().set_title("Model")

# %%
fig.savefig("../paper/img/final_sims_evals.pdf", bbox_inches="tight")

# %%
final_sim_tmp = final_sim.rename(
    columns={
        "task": "Task",
        "metric_nice": "Metric",
        "model_nice": "Model",
        "score": "Score",
    }
)
map_model_order = (
    final_sim_tmp[final_sim_tmp["metric"] == "map"]
    .groupby("Model")["Score"]
    .mean()
    .sort_values()
    .index
)
final_sim_tmp["Model"] = final_sim_tmp["Model"].astype(
    pd.CategoricalDtype(map_model_order, ordered=True)
)
final_sim_tmp["Task"] = final_sim_tmp["Task"].apply(
    lambda t: "games" if t == "sims_games" else "wines"
)

fig, ax = plt.subplots(figsize=(6.5, 4))

sns.scatterplot(
    final_sim_tmp,
    x="Model",
    y="Score",
    hue="Task",
    style="Metric",
    ax=ax,
    s=55,
)

ax.tick_params("x", labelrotation=45)
# sns.move_legend(ax, 'center')
leg = ax.get_legend()
leg.remove()
ax.legend(
    leg.get_lines(), [t.get_text() for t in leg.get_texts()], ncols=2, mode="expand"
)
ax.get_legend().set_bbox_to_anchor((0, 1, 1, 0.25))

# %%
fig.savefig("../paper/img/final_sims_evals_per_task.pdf", bbox_inches="tight")

# %% [markdown]
# # 1k vs 10k vs Final

# %%
all_evals = [
    ntb_utils.load_validation_results("../evaluations/final_eval_1k/"),
    ntb_utils.load_validation_results("../evaluations/final_eval_short/"),
    ntb_utils.load_validation_results("../evaluations/final_eval/"),
]

all = []
for eval, suff in zip(all_evals, [";1k", ";10k", ";all"]):
    eval = eval[eval["model"].isin(considered_models)].copy()
    eval["model_nice"] = eval["model"].apply(gen_nice_model) + suff
    eval["model"] = eval.model + suff
    eval["train_data"] = suff[1:]
    all.append(eval)


all = pd.concat(all)

# %%
all = ntb_utils.add_normalized_score(all)

# %%
all_acc = all[all["metric"] == "accuracy"].copy()

# %%
all_acc["train_data"] = all_acc["train_data"].astype(
    pd.CategoricalDtype(["1k", "10k", "all"], ordered=True)
)

# %%
fig, ax = plt.subplots(figsize=(6.5, 8.5))

sns.boxplot(
    all_acc,
    y="model_nice",
    x="normalized_score",
    hue="train_data",
    order=all_acc.groupby(["train_data", "model_nice"], observed=True)[
        "normalized_score"
    ]
    .mean()
    .reset_index()
    .sort_values(["train_data", "normalized_score"])["model_nice"],
    **ntb_utils.boxplot_kwargs,
    ax=ax,
)

ax.set_ylabel("Embedding model")
ax.set_xlabel("Normalized accuracy")
ax.get_legend().set_title("Finetuning data")

# %%
fig.savefig("../paper/img/final_eval_norm_all.pdf", bbox_inches="tight")

# %%
by_train_data = {
    #'1k': ntb_utils.load_validation_results('../evaluations/final_eval_1k/'),
    "10k": ntb_utils.load_validation_results("../evaluations/final_eval_short/"),
    "all": ntb_utils.load_validation_results("../evaluations/final_eval/"),
}


for train_data, eval in by_train_data.items():
    eval = eval[eval["model"].isin(considered_models)].copy()
    eval["model_nice"] = eval["model"].apply(gen_nice_model)
    eval = eval[eval["metric"] == "accuracy"]
    by_train_data[train_data] = eval

model_order = ntb_utils.add_normalized_score(by_train_data["all"])
model_order = ntb_utils.model_order(model_order, "normalized_score", of="model_nice")

cat_type = pd.CategoricalDtype(model_order, ordered=True)

for eval in by_train_data.values():
    eval["model_nice"] = eval["model_nice"].astype(cat_type)


# %%
def gen_acc_plot(df, fig=None, ax=None):
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # sns.lineplot(
    #    df,
    #    x='model_nice',
    #    y='score',
    #    hue='task',
    #    orient='x',
    #    ax=ax,
    # )
    if False:
        sns.scatterplot(
            df,
            x="model_nice",
            y="score",
            hue="task",
            style="task",
            ax=ax,
            s=55,
        )

    if False:
        sns.swarmplot(
            df,
            x="model_nice",
            y="score",
            hue="task",
            ax=ax,
            size=6,
        )

    tasks = df["task"].unique()

    lines = ax.get_legend().get_lines()
    ax.get_legend().get_texts()
    ax.get_legend().remove()

    ax.legend(
        lines,
        tasks,
        ncols=len(tasks),
        mode="expand",
        title="Tasks",
    )
    ax.tick_params("x", labelrotation=45)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")

    return fig, ax


# %%
def barplot_per_task(df, fig, ax):
    sns.barplot(
        df,
        x="task",
        y="score",
        hue="model_nice",
        ax=ax,
    )

    df["model_nice"].unique()

    lines = ax.get_legend().get_patches()
    texts = ax.get_legend().get_texts()
    ax.get_legend().remove()

    ax.legend(
        lines,
        [t.get_text() for t in texts],
        ncols=4,
        mode="expand",
        title="Models",
    )
    # ax.tick_params('x', labelrotation=45)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Task")

    return fig, ax


# %%
fig, axes = plt.subplots(
    len(by_train_data), 1, figsize=(8, 6), gridspec_kw={"hspace": 0.15}
)

for ax, (train_data, evals) in zip(axes, by_train_data.items()):
    barplot_per_task(evals, fig, ax)
    ax.set_title(train_data)

    if train_data != "all":
        ax.set_xlabel("")
        ax.set_xticks(ax.get_xticks(), labels=[])
    if train_data != "10k":
        ax.get_legend().set_visible(False)
    else:
        # sns.move_legend(ax, 'center right')
        # ax.get_legend().set_bbox_to_anchor((1.25, -0.1))
        ax.get_legend().set_bbox_to_anchor((0, 1.05, 1, 0.4))

# %%
fig.savefig("../paper/img/final_cls_evals.pdf", bbox_inches="tight")

# %%
all_evals = [
    ntb_utils.load_validation_results("../evaluations/final_eval_1k/"),
    ntb_utils.load_validation_results("../evaluations/final_eval_short/"),
    ntb_utils.load_validation_results("../evaluations/final_eval/"),
]

all = []
for eval, suff in zip(all_evals, [";1k", ";10k", ";all"]):
    eval = eval[eval["model"].isin(considered_models)].copy()
    eval["model_nice"] = eval["model"].apply(gen_nice_model)
    eval["train_data"] = suff[1:]
    all.append(eval)


all = pd.concat(all)

# %%
all_acc = all[all["metric"] == "accuracy"]
final_round = all_acc[all_acc["train_data"] == "all"]
final_round = ntb_utils.add_normalized_score(final_round)
all["model_nice"] = all["model_nice"].astype(
    pd.CategoricalDtype(
        ntb_utils.model_order(final_round, "normalized_score", of="model_nice"),
        ordered=True,
    )
)

# %%
fig, ax = plt.subplots(figsize=(8, 10))

sns.lineplot(
    all_acc,
    y="model_nice",
    x="score",
    hue="task",
    style="train_data",
    # order=all_acc.groupby(['train_data', 'model'])['normalized_score'].mean().reset_index().sort_values(['train_data', 'normalized_score'])['model'],
    # **ntb_utils.boxplot_kwargs,
    ax=ax,
    orient="y",
)

# %%

# %% [markdown]
# # Final vs 1k

# %%
all_evals = [eval.copy() for eval in [evals_1k, final_evals]]
for eval, suff in zip(all_evals, ["_1k", "_final"]):
    eval["model"] = eval.model + suff

all = pd.concat(all_evals)

# %%
all = ntb_utils.add_normalized_score(all)
all_acc = all[all["metric"] == "accuracy"]

# %%
sns.boxplot(
    all_acc,
    y="model",
    x="normalized_score",
    order=ntb_utils.model_order(all_acc, by="normalized_score"),
    **ntb_utils.boxplot_kwargs,
)

# %% [markdown]
# # Final vs short

# %%
all_evals = [eval.copy() for eval in [short_evals, final_evals]]
for eval, suff in zip(all_evals, ["_short", "_final"]):
    eval["model"] = eval.model + suff

all = pd.concat(all_evals)

# %%
all = ntb_utils.add_normalized_score(all)
all_acc = all[all["metric"] == "accuracy"]

# %%
sns.boxplot(
    all,
    y="model",
    x="normalized_score",
    order=ntb_utils.model_order(all, by="normalized_score"),
    **ntb_utils.boxplot_kwargs,
)

# %% [markdown]
# # The table

# %%
evals = {
    "1k": ntb_utils.load_validation_results("../evaluations/final_eval_1k/"),
    "10k": ntb_utils.load_validation_results("../evaluations/final_eval_short/"),
    "all": ntb_utils.load_validation_results("../evaluations/final_eval/"),
}


def gen_table_task(task):
    presets = {
        "sims_games": "games",
        "sims_wines": "wines",
    }

    nice = presets.get(task, task)
    return r"\Task{" + nice + r"}"


def gen_table_model(nice_model):
    if nice_model in ["DM", "PV", "cosine-masked", "MSE-contextual", "only-MSE"]:
        return r"\TableModel{" + nice_model + r"}"
    return nice_model


for lim_name, ev in evals.items():
    ev["model_nice"] = ev["model"].apply(gen_nice_model)
    ev["model_table"] = ev["model_nice"].apply(gen_table_model)
    ev["task_table"] = ev["task"].apply(gen_table_task)
    ev["metric_nice"] = ev["metric"].apply(gen_nice_metric)
    ev = ntb_utils.add_normalized_score(ev)
    evals[lim_name] = ev[ev["metric"].isin(["accuracy", "map"])]

task_order = [
    r"\Task{" + t + r"}"
    for t in ["arxiv", "imdb", "aan", "oc", "pan", "s2orc", "games", "wines"]
]
cls_task_order = task_order[:-2]
sim_task_order = task_order[-2:]

# %%
cls_rows = []
sim_rows = []
for limiting in ["all", "10k", "1k"]:
    lim_evals = evals[limiting]
    for model in [
        "Longformer",
        "DM",
        "PV",
        "SBERT",
        "cosine-masked",
        "MSE-contextual",
        "only-MSE",
    ]:
        model_lim_evals = lim_evals[lim_evals["model_nice"] == model]
        mna = model_lim_evals[model_lim_evals["metric"] == "accuracy"][
            "normalized_score"
        ].mean()
        mmap = model_lim_evals[model_lim_evals["metric"] == "map"][
            "normalized_score"
        ].mean()

        considered_tasks = task_order if limiting == "all" else task_order[:-2]

        task_values = {}
        for task in considered_tasks:
            score = model_lim_evals[model_lim_evals["task_table"] == task][
                "score"
            ].item()
            task_values[task] = f"{score:.3f}"[1:]

        cls_rows.append(
            {
                "Model": gen_table_model(model),
                **{
                    task: score
                    for task, score in task_values.items()
                    if task in cls_task_order
                },
                "Mean accuracy": f"{model_lim_evals[model_lim_evals['metric'] == 'accuracy']['score'].mean().item():.3f}"[
                    1:
                ],
                "Mean normalized accuracy": f"{mna:.3f}"[1:],
            }
        )

        if limiting == "all":
            sim_rows.append(
                {
                    "Model": gen_table_model(model),
                    **{
                        task: score
                        for task, score in task_values.items()
                        if task in sim_task_order
                    },
                    "Mean MAP": f"{model_lim_evals[model_lim_evals['metric'] == 'map']['score'].mean().item():.3f}"[
                        1:
                    ],
                    "Mean normalized MAP": f"{mmap:.3f}"[1:],
                }
            )

# %%
cls_table = pd.DataFrame(cls_rows)
sim_table = pd.DataFrame(sim_rows)

# %%
cls_table

# %%
sim_table

# %%
cls_latex = cls_table.style.hide(axis="index")
print(cls_latex.to_latex(hrules=True, column_format="lccccccrr"))

# %%
sim_latex = sim_table.style.hide(axis="index")
print(sim_latex.to_latex(hrules=True, column_format="lcccc"))

# %% [markdown]
# # Analysis

# %%
from transformer_document_embedding import notebook_utils as ntb_utils
from transformer_document_embedding.datasets.imdb import IMDB
from transformer_document_embedding.datasets.document_pair_classification import (
    DocumentPairClassification,
)
from transformer_document_embedding.datasets import col
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# %%
models = {
    "sbert": ntb_utils.load_model_save(
        "../results/teacher_embedding:TeacherEmbedding/transformer:TransformerEmbedder/sbert_save/trained_model"
    ),
    "cos_final": ntb_utils.load_model_save(
        "../results/teacher_embedding:TeacherEmbedding/transformer:TransformerEmbedder/cos_final/trained_model"
    ),
    "mm_mse_contextual_final": ntb_utils.load_model_save(
        "../results/teacher_embedding:TeacherEmbedding/transformer:TransformerEmbedder/mm_mse_contextual_final/trained_model"
    ),
}

# %%
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

# %%
tasks = {
    "imdb": IMDB(),
    "pan": DocumentPairClassification("../data/MDA/PAN"),
    "aan": DocumentPairClassification("../data/MDA/AAN"),
}


# %% [markdown]
# ## KNN classifier


# %%
def gen_embeddings(
    task: DocumentDataset, model: EmbeddingModel, *, samples: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    test_split = task.splits["test"]
    labels = test_split.unique(col.LABEL)
    docs_per_each_label = samples // len(labels) + 1

    label_counts = {label: 0 for label in labels}

    docs = []
    for doc in test_split:
        if label_counts[doc[col.LABEL]] < docs_per_each_label:
            docs.append(doc)
            label_counts[doc[col.LABEL]] += 1

        if sum(label_counts.values()) == len(labels) * docs_per_each_label:
            break

    to_gen = Dataset.from_pandas(pd.DataFrame(docs))

    feats = get_head_features(task.evaluation_kind, to_gen, model, batch_size=4)

    def get_length(
        docs: dict[str, Any], text_col: str, length_col: str
    ) -> dict[str, Any]:
        tokenized = tokenizer(
            docs[text_col],
            padding="longest",
            truncation=False,
            return_tensors="np",
            verbose=False,
            add_special_tokens=False,
        )
        lengths = tokenized["attention_mask"].sum(axis=1)
        return {length_col: lengths}

    used_length_cols = []
    for text_col, length_col in zip(
        *[
            [f"{stem}_{i}" if i >= 0 else stem for i in range(-1, 2)]
            for stem in ["text", "length"]
        ]
    ):
        if text_col not in feats.column_names:
            continue

        feats = feats.map(
            get_length,
            fn_kwargs={"text_col": text_col, "length_col": length_col},
            batched=True,
            batch_size=256,
            num_proc=4,
        )
        used_length_cols.append(length_col)
    feats.set_format("numpy", columns=[col.EMBEDDING, col.LABEL] + used_length_cols)

    return (
        feats[col.EMBEDDING],
        feats[col.LABEL],
        *[feats[length_col] for length_col in used_length_cols],
    )


def train_knn(feats: np.ndarray, labels: np.ndarray) -> tuple[Any, dict[str, Any]]:
    feats_train, feats_test, labels_train, labels_test, _, test_idx = train_test_split(
        feats, labels, np.arange(len(labels)), test_size=0.33
    )

    classifier = KNeighborsClassifier(
        n_neighbors=10,
    )

    classifier.fit(feats_train, labels_train)
    labels_pred = classifier.predict(feats_test)

    report = classification_report(labels_test, labels_pred, output_dict=True)

    correct = labels_pred == labels_test

    return classifier, report, correct, test_idx


# %%
res = []
lengths = []
for task_name, task in tqdm(tasks.items()):
    for model_name, model in tqdm(models.items()):
        feats, labels, *lengths = gen_embeddings(task, model)
        _, report, test_corrects, test_idxs = train_knn(feats, labels)
        res.append(
            {
                "model": model_name,
                "task": task_name,
                "accuracy": report["accuracy"],
            }
        )
        all_df_lengths = []
        for length_set in lengths:
            all_df_lengths.append(
                pd.DataFrame(
                    {
                        "length": length_set[test_idxs],
                        "correct": test_corrects,
                    }
                )
            )
        model_lengths = pd.concat(all_df_lengths)
        model_lengths["model"] = model_name
        model_lengths["task"] = task_name
        lengths.append(model_lengths)

res = pd.DataFrame(res)
lengths = pd.concat(lengths)

# %%
res = pd.DataFrame(res)

# %%
res.set_index("task").sort_values("accuracy").sort_index()
