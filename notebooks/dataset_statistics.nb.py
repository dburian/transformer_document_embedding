# %% [markdown]
# # Generating dataset statistics

# %%
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Union, cast

from transformer_document_embedding.datasets import col
from transformer_document_embedding.datasets.arxiv_papers import ArxivPapers
from transformer_document_embedding.datasets.document_dataset import DocumentDataset
from transformer_document_embedding.datasets.document_pair_classification import (
    DocumentPairClassification,
)
from transformer_document_embedding.datasets.wikipedia_similarities import (
    WikipediaSimilarities,
)
from transformer_document_embedding.datasets.imdb import IMDB

import transformer_document_embedding.notebook_utils as ntb_utils
import numpy as np
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerFast

if TYPE_CHECKING:
    from datasets import Dataset

# %%
ntb_utils.seaborn_defaults()
ntb_utils.uncompressed_tables()

# %%
NUM_PROC = 4
SBERT_TOKENIZER = "sentence-transformers/all-mpnet-base-v2"
LONGFORMER_TOKENIZER = "allenai/longformer-base-4096"

# %%
data_size_limit = {
    "train": 10000,
    "validation": 10000,
}

splits = {
    "train": "train",
    "test": "validation",
}

kwargs = {"splits": splits, "data_size_limit": data_size_limit}

train_data_size_limit = {
    "train": 15000,
    "validation": 7680,
}

gen_stats_for = {
    # "wiki_short": TeacherEmbedding(path="/mnt/data/datasets/wikipedia_resampled_eval/", data_size_limit=train_data_size_limit),
    #'wiki': TeacherEmbedding(
    #    path='/mnt/data/datasets/wikipedia_resampled_eval/',
    # ),
    # "arxiv": ArxivPapers("/mnt/data/datasets/arxiv_hf", **kwargs),
    # "imdb": IMDB(data_size_limit=data_size_limit, splits={"train": "train"}),
    # "oc": DocumentPairClassification("../data/MDA/OC", **kwargs),
    # "aan": DocumentPairClassification("../data/MDA/AAN", **kwargs),
    # "s2orc": DocumentPairClassification("../data/MDA/S2ORC", **kwargs),
    # "pan": DocumentPairClassification("../data/MDA/PAN", **kwargs),
    # "val_corpus": TeacherEmbedding('/mnt/data/datasets/val_corpus_500k/'),
    # "val_corpus_short": TeacherEmbedding('/mnt/data/datasets/val_corpus_500k/', data_size_limit=train_data_size_limit),
    "aan": DocumentPairClassification("../data/MDA/AAN"),
    "arxiv": ArxivPapers("/mnt/data/datasets/arxiv_hf"),
    "imdb": IMDB(),
    "oc": DocumentPairClassification("../data/MDA/OC"),
    "pan": DocumentPairClassification("../data/MDA/PAN"),
    "s2orc": DocumentPairClassification("../data/MDA/S2ORC"),
    "wines": WikipediaSimilarities(
        path="../data/wikipedia_similarities.py",
        dataset="wine",
        splits={"test": "test"},
    ),
    "games": WikipediaSimilarities(
        path="../data/wikipedia_similarities.py",
        dataset="game",
        splits={"test": "test"},
    ),
}
# %%

text_columns = [col.TEXT, col.TEXT_0, col.TEXT_1]
sbert_length_columns = [col.LENGTH, "length_0", "length_1"]
longformer_length_columns = [f"longformer_{col}" for col in sbert_length_columns]

sbert_thres = np.array([128, 256, 384, 512])
longformer_thres = np.array([512 * i for i in range(1, 9)])


def get_length(
    docs: dict[str, Any],
    text_column: str,
    length_column: str,
    tokenizer: PreTrainedTokenizerFast,
) -> dict[str, Any]:
    tokenized = tokenizer(
        docs[text_column],
        padding="longest",
        truncation=False,
        return_tensors="np",
        verbose=False,
        add_special_tokens=False,
    )
    lengths = tokenized["attention_mask"].sum(axis=1)
    return {length_column: lengths}


def gen_stats_for_split(split: Dataset) -> dict[str, Union[float, int]]:
    for text_col, length_col in zip(text_columns, sbert_length_columns, strict=True):
        if text_col in split.column_names and length_col not in split.column_names:
            split = split.map(
                get_length,
                fn_kwargs={
                    "text_column": text_col,
                    "length_column": length_col,
                    "tokenizer": AutoTokenizer.from_pretrained(SBERT_TOKENIZER),
                },
                batched=True,
                batch_size=64,
                num_proc=NUM_PROC,
            )

    for text_col, length_col in zip(
        text_columns, longformer_length_columns, strict=True
    ):
        if text_col in split.column_names and length_col not in split.column_names:
            split = split.map(
                get_length,
                fn_kwargs={
                    "text_column": text_col,
                    "length_column": length_col,
                    "tokenizer": AutoTokenizer.from_pretrained(LONGFORMER_TOKENIZER),
                },
                batched=True,
                batch_size=64,
                num_proc=NUM_PROC,
            )

    label_counts = (
        {label: 0 for label in split.unique(col.LABEL)}
        if col.LABEL in split.column_names
        and not isinstance(split.features[col.LABEL], list)
        else {}
    )
    words_total = 0
    sbert_tokens = []
    longformer_tokens = []
    source_documents = 0
    num_target_documents = []

    for docs in tqdm(
        split.with_format("np").iter(batch_size=128),
        desc="Stats for split",
        total=np.ceil(len(split) / 128),
    ):
        docs = cast(dict[str, Union[list[Any], np.ndarray]], docs)
        if len(label_counts) > 0:
            for label, count in zip(
                *np.unique(docs[col.LABEL], return_counts=True), strict=True
            ):
                label_counts[label] += count
        elif col.LABEL in docs:
            batch_target_documents = np.array(
                [len(targets) for targets in docs[col.LABEL]]
            )
            source_documents += (batch_target_documents > 0).sum()
            num_target_documents.extend(
                [
                    num_targets
                    for num_targets in batch_target_documents
                    if num_targets > 0
                ]
            )

        for text_col in text_columns:
            if text_col in docs:
                words_total += sum((len(t.split()) for t in docs[text_col]))

        for length_col in sbert_length_columns:
            if length_col in docs:
                sbert_tokens.extend(docs[length_col])

        for length_col in longformer_length_columns:
            if length_col in docs:
                longformer_tokens.extend(docs[length_col])

    stats = {
        "words": words_total,
        "sbert_tokens": sbert_tokens,
        "longformer_tokens": longformer_tokens,
        "label_counts": label_counts,
        "docs_total": len(sbert_tokens),
        "docs": len(split),
        "source_documents": source_documents,
        "target_documents": num_target_documents,
    }

    return stats


# %%
stats = {}
for ds_name, ds_stats in tqdm(gen_stats_for.items(), desc="Dataset"):
    stats[ds_name] = {}
    splits = ds_stats.splits if isinstance(ds_stats, DocumentDataset) else ds_stats
    for split_name, split in splits.items():
        stats[ds_name][split_name] = gen_stats_for_split(split)

# %%
stats_df = {}
for ds_name, ds_stats in stats.items():
    for split_name, split_stats in ds_stats.items():
        stats_df[(ds_name, split_name)] = {
            "wpd": split_stats["words"] / split_stats["docs_total"],
            "stpd_med": np.median(split_stats["sbert_tokens"]),
            "stpd": np.mean(split_stats["sbert_tokens"]),
            "stpd_std": np.std(split_stats["sbert_tokens"]),
            "ltpd": np.mean(split_stats["longformer_tokens"]),
            "ltpd_std": np.std(split_stats["longformer_tokens"]),
            "docs": split_stats["docs"],
            **{
                f"sbert_over_{thres}": np.mean(split_stats["sbert_tokens"] > thres)
                * 100
                for thres in sbert_thres
            },
            **{
                f"longformer_over_{thres}": np.mean(
                    split_stats["longformer_tokens"] > thres
                )
                * 100
                for thres in longformer_thres
            },
            **{
                f"label_perc_{label}": count / split_stats["docs"] * 100
                for label, count in split_stats["label_counts"].items()
            },
        }

# %%
stats_df = pd.DataFrame(stats_df)

# %%
stats_df

# %% [markdown]
# ## Presentable stats for training data

# %% [markdown]
# ### Split views

# %%
dataset = "val_corpus"

# %%
detail = []
for split_name, split_stats in stats[dataset].items():
    detail.append(
        {
            "Split": split_name.capitalize(),
            r"\# documents": f"{split_stats['docs']:_}".replace("_", " "),
            r"\# tokens": f"{sum(split_stats['longformer_tokens']):.2e}",
            r"Avg. tokens per document": f"{stats_df[dataset][split_name].loc['ltpd']:.2f}"
            r"$\pm$"
            f"{stats_df[dataset][split_name].loc['ltpd_std']:.2f}",
            "SBERT tokens over 384": f"{np.mean(np.array(split_stats['sbert_tokens']) > 384)*100:.2f}"
            r"\%",
            "SBERT tokens over 512": f"{np.mean(np.array(split_stats['sbert_tokens']) > 512)*100:.2f}"
            r"\%",
        }
    )

detail = pd.DataFrame(detail).set_index(["Split"])

# %%
detail

# %%
latex_style = detail.T.style
print(latex_style.to_latex(hrules=True, column_format="lrr", clines="skip-last;data"))


# %% [markdown]
# ## Distribution graphs for training data


# %%
def wiki_flag(docs):
    flags = ["wikipedia" in url for url in docs["url"]]
    return {"wikipedia_flag": flags}


gen_stats_for[dataset]._splits = gen_stats_for[dataset].splits.map(
    wiki_flag, batched=True, batch_size=256, num_proc=NUM_PROC
)


# %%
fig, ax = plt.subplots(figsize=(8, 4))

sns.ecdfplot(stats[dataset]["train"]["longformer_tokens"], ax=ax)
sns.ecdfplot(stats[dataset]["validation"]["longformer_tokens"], ax=ax)
ax.legend(ax.lines, ["train", "validation"], title="Splits")

ax.set_xscale("log")
x_lim = 16
ax.set_xticks([2**i for i in range(3, x_lim)], [str(2**i) for i in range(3, x_lim)])
ax.set_yticks([0.1 * i for i in range(0, 11)])
ax.set_xlabel("Longformer tokens")

ax.set_xlim((8, 2**x_lim))
ax.set_ylim((-0.05, 1.05))

# %%
fig.savefig("../paper/img/val_data_ecdf.png", dpi=300, bbox_inches="tight")

# %%
tmp_df = pd.concat(
    [
        pd.DataFrame(
            {
                "Token count": stats[dataset][split]["longformer_tokens"],
                "Split": [
                    split
                    for _ in range(len(stats[dataset][split]["longformer_tokens"]))
                ],
                "Wikipedia": gen_stats_for[dataset].splits[split]["wikipedia_flag"],
            }
        )
        for split in ["train", "validation"]
    ]
)

# %%
# tmp_df['Part'] = tmp_df.apply(lambda row: f"({'Wikipedia' if row['Wikipedia'] else 'RealNews'}) {row['Split']}", axis=1)
tmp_df["Source"] = tmp_df["Wikipedia"].apply(
    lambda wiki: "Wikipedia" if wiki else "RealNews"
)

# %%
fig, axes = plt.subplots(2, 1, figsize=(8, 4))

base_colors = sns.color_palette(n_colors=4)

for split, ax in zip(["train", "validation"], axes):
    sns.histplot(
        tmp_df[tmp_df["Split"] == split],
        x="Token count",
        hue="Source",
        ax=ax,
        stat="percent",
        common_norm=False,
        element="step",
        multiple="stack",
        binwidth=30,
    )

    ax.set_title(split.capitalize())

xticks = np.arange(-500, 4500, 500)
lims = (-250, 4250)
axes[0].set_xticks(xticks, [])
axes[0].set_xlabel("")
axes[0].set_xlim(lims)
# axes[0].get_legend().remove()

axes[1].set_xticks(xticks)
axes[1].set_xlim(lims)
# sns.move_legend(axes[1], "upper left", bbox_to_anchor=(1, 1.35))


# %%
fig.savefig("../paper/img/val_data_dist.pdf", bbox_inches="tight")

# %% [markdown]
# ## Presentable stats for tasks

# %% [markdown]
# ### Overview validation dataset stats
#
# - rows: datasets
# - columns: number of classes, mean class percentage +- std, evaluated with

# %%
label_rows = [
    row
    for row in stats_df.index
    if row.startswith("label_perc") and not row.endswith("-1")
]
overview = []
for ds_name in gen_stats_for:
    ds_overview: dict[tuple[str, str], Any] = (
        {
            ("", "Dataset"): f"\\Task{{{ds_name}}}",
        }
        | {
            ("Documents", split_name.capitalize()): (
                r"\dag" if split_stats["docs"] == 10000 else ""
            )
            + f"{split_stats['docs']:_}".replace("_", " ")
            for split_name, split_stats in stats[ds_name].items()
        }
        | {("", "Classes"): stats_df[(ds_name, "train")].loc[label_rows].notna().sum()}
    )
    for split_name in stats_df[ds_name].columns:
        label_stats = stats_df[(ds_name, split_name)].loc[label_rows].to_numpy()
        ds_overview[
            (
                "Class percentage",
                f"{split_name.capitalize() if split_name == 'train' else 'Validation'}",
            )
        ] = (
            f"{np.nanmean(label_stats):.2f}"
            r"$\pm$"
            f"{np.nanstd(label_stats):.2f}"
            r"\%"
        )

    overview.append(ds_overview)
overview = pd.DataFrame(overview).fillna("-")

# %%
overview

# %%
overview.columns = pd.MultiIndex.from_tuples(overview.columns)

# %%
latex_style = pd.io.formats.style.Styler(
    overview,
).hide(axis="index")

# %%
print(
    latex_style.to_latex(
        column_format="llrrr", hrules=True, multicol_align="c", clines="all;index"
    )
)

# %% [markdown]
# ### Overview evaluation stats
# %%
label_rows = [
    row
    for row in stats_df.index
    if row.startswith("label_perc") and not row.endswith("-1")
]
overview = []
for ds_name in gen_stats_for:
    ds_overview: dict[tuple[str, str], Any] = {
        ("", "Dataset"): f"\\Task{{{ds_name}}}",
        ("", "Inputs"): "pairs of documents"
        if ds_name not in ["imdb", "arxiv"]
        else "documents",
        ("", "Classes"): stats_df[(ds_name, "train")].loc[label_rows].notna().sum(),
    }
    for split_name in stats_df[ds_name].columns:
        if split_name in ["unsupervised", "validation"]:
            continue
        label_stats = stats_df[(ds_name, split_name)].loc[label_rows].to_numpy()
        ds_overview[
            (
                "Class percentage",
                f"{split_name.capitalize()}",
            )
        ] = (
            f"{np.nanmean(label_stats):.2f}"
            r"$\pm$"
            f"{np.nanstd(label_stats):.2f}"
            r"\%"
        )

    overview.append(ds_overview)
overview = pd.DataFrame(overview).fillna("-")
overview.columns = pd.MultiIndex.from_tuples(overview.columns)

# %%
overview

# %%
latex_style = overview.style.hide(axis="index")

# %%
print(
    latex_style.to_latex(
        column_format="llrrr", hrules=True, multicol_align="c", clines="all;index"
    )
)

# %% [markdown]
# ### Split validation stats
#
# - rows: multirows (dataset, split)
# - columns: number of documents, over 384, over 512 (SBERT)

# %%
detail = []
for ds_name, ds_stats in stats.items():
    detail.append(
        {
            ("", "Dataset"): f"\\Task{{{ds_name}}}",
        }
        | {
            ("Documents", split_name.capitalize()): (
                r"\dag" if split_stats["docs"] == 10000 else ""
            )
            + f"{split_stats['docs']:_}".replace("_", " ")
            for split_name, split_stats in ds_stats.items()
        }
    )

detail = pd.DataFrame(detail).fillna("-")
detail.columns = pd.MultiIndex.from_tuples(detail.columns)

# %%
detail

# %%
latex_style = detail.style.hide(axis="index")
print(latex_style.to_latex(hrules=True, column_format="llrrr", clines="skip-last;data"))

# %% [markdown]
# ### Split evaluation stats
#
# - rows: multirows (dataset, split)
# - columns: number of documents, over 384, over 512 (SBERT)

# %% [markdown]
# #### Classification

# %%
detail = []
for ds_name, ds_stats in stats.items():
    for split_name, split_stats in ds_stats.items():
        if split_name in ["unsupervised", "validation"]:
            continue
        detail.append(
            {
                "Dataset": f"\\Task{{{ds_name}}}",
                "Split": split_name.capitalize(),
                r"\#Documents": (r"\dag" if split_stats["docs"] == 10000 else "")
                + f"{split_stats['docs']:_}".replace("_", " "),
                "Tokens over 384": f"{np.mean(np.array(split_stats['sbert_tokens']) > 384)*100:.2f}"
                r"\%",
                "Tokens over 512": f"{np.mean(np.array(split_stats['sbert_tokens']) > 512)*100:.2f}"
                r"\%",
            }
        )

detail = pd.DataFrame(detail).set_index(["Dataset", "Split"])

# %%
detail

# %%
latex_style = detail.style
print(latex_style.to_latex(hrules=True, column_format="llrrr", clines="skip-last;data"))

# %% [markdown]
# #### Similarities

# %%
detail = []
for ds_name, ds_stats in stats.items():
    for split_name, split_stats in ds_stats.items():
        if split_name in ["train", "unsupervised", "validation"]:
            continue
        detail.append(
            {
                "Dataset": f"\\Task{{{ds_name}}}",
                r"Documents": f"{split_stats['docs']:_}".replace("_", " "),
                r"Sources": f"{split_stats['source_documents']}",
                r"Targets per source": f"{np.mean(split_stats['target_documents']):.2f}"
                r"$\pm$"
                f"{np.std(split_stats['target_documents']):.2f}",
                "Tokens over 384": f"{np.mean(np.array(split_stats['sbert_tokens']) > 384)*100:.2f}"
                r"\%",
                "Tokens over 512": f"{np.mean(np.array(split_stats['sbert_tokens']) > 512)*100:.2f}"
                r"\%",
            }
        )

detail = pd.DataFrame(detail).set_index(["Dataset"])

# %%
detail

# %%
latex_style = detail.style
print(
    latex_style.to_latex(hrules=True, column_format="lrrrrr", clines="skip-last;data")
)

# %% [markdown]
# ### Longformer token's distribution

# %%
import matplotlib

# %%
fig, ax = plt.subplots(figsize=(8, 4))

colors = sns.color_palette()

labels = []
for ds_color, (ds_name, ds_stats) in zip(colors, stats.items()):
    for linestyle, split_name in zip(["-", "--"], ["test", "train"]):
        split_stats = ds_stats[split_name]
        sns.ecdfplot(
            split_stats["longformer_tokens"],
            color=ds_color,
            ax=ax,
            linestyle=linestyle,
        )
        if split_name == "train":
            labels.append(ds_name)

ax.legend(ax.lines[0::2], labels, title="Dataset")
ax.set_xscale("log")
ax.set_xticks([2**i for i in range(3, 13)], [str(2**i) for i in range(3, 13)])
ax.set_yticks([0.1 * i for i in range(0, 11)])
ax.set_xlabel("Tokens (log)")

ax.set_xlim((8, 4096 + 1024))
ax.set_ylim((-0.05, 1.05))

# %%
original_legend = ax.get_legend()

# %%
test_line = matplotlib.lines.Line2D(
    [0], [0], linestyle="-", color="black", label="test"
)
train_line = matplotlib.lines.Line2D(
    [0], [0], linestyle="--", color="black", label="train"
)
split_legend = ax.legend(
    handles=[test_line, train_line],
    title="Split",
    loc="lower left",
    bbox_to_anchor=(0, 0.17),
)
ax.add_artist(original_legend)
fig

# %%
fig.savefig("../paper/img/eval_tasks_token_ecdf.pdf", bbox_inches="tight")

# %%
fig, ax = plt.subplots(figsize=(6, 4))

colors = sns.color_palette()

labels = []
for ds_color, (ds_name, ds_stats) in zip(colors, stats.items()):
    split_colors = sns.light_palette(ds_color, n_colors=7)
    split_colors = [split_colors[3], split_colors[-1]]
    for split_color, split_name in zip(split_colors, ["test", "train"]):
        split_stats = ds_stats[split_name]
        sns.kdeplot(
            split_stats["longformer_tokens"],
            color=split_color,
            ax=ax,
        )
        labels.append(f"{ds_name.upper()} {split_name}")

ax.legend(ax.lines, labels)
ax.set_xscale("log")
ax.set_xticks([2**i for i in range(3, 13)], [str(2**i) for i in range(3, 13)])

ax.set_xlim((8, 4096 + 1024))

# %% [markdown]
# ---

# %%
stats_df.loc["number_of_labels"] = stats_df.apply(
    lambda col: sum(
        not pd.isna(col[row]) for row in col.keys() if row.startswith("label_")
    ),
    axis=0,
)

stats_df.loc["mean_label_perc"] = stats_df.apply(
    lambda col: np.nanmean(
        [col[row] * 100 for row in col.keys() if row.startswith("label_")]
    )
)
stats_df.loc["std_label_perc"] = stats_df.apply(
    lambda col: np.nanstd(
        [col[row] * 100 for row in col.keys() if row.startswith("label_")]
    )
)

stats_df = stats_df.drop(
    index=[col for col in stats_df.index if col.startswith("label_")]
)

stats_df.loc["metric"] = [
    "Binary accuracy" if num_labels == 2 else "Micro-averaged accuracy"
    for num_labels in stats_df.loc["number_of_labels"]
]

stats_df.loc["evaluation_mode"] = [
    "-"
    if col[1] != "train"
    else ("Validation" if col[0] != "imdb" else "Cross-validation")
    for col in stats_df.columns
]
stats_df.columns = pd.MultiIndex.from_tuples(
    [(col[0], col[1] if col[1] != "test" else "validation") for col in stats_df.columns]
)

# %%
stats_df

# %%
latex_df = pd.DataFrame(columns=stats_df.columns)
latex_df.loc["Number of documents"] = stats_df.apply(
    lambda col: f"{int(col['docs']):,d}".replace(",", " ")
)
latex_df.loc["Over threshold"] = stats_df.apply(lambda col: f"{col['perc_tat']:.3g}%")
latex_df.loc["Number of classes"] = stats_df.apply(
    lambda col: f"{int(col['number_of_labels']):d}"
)
latex_df.loc["Metric"] = stats_df.loc["metric"]
latex_df.loc["Evaluated with"] = stats_df.loc["evaluation_mode"]
# latex_df.loc['Words per document'] = latex_df.apply(lambda col: f"{col['mean_wpd']:.2f}+-{col['std_wpd']:.2f}")
# latex_df.loc['Tokens per document'] = latex_df.apply(lambda col: f"{col['mean_tpd']:.2f}+-{col['std_tpd']:.2f}")

# %%
col_tuples = [
    (tup[0].upper() if tup[0] != "arxiv" else "Arxiv papers", tup[1].capitalize())
    for tup in latex_df.columns
]
latex_df.columns = pd.MultiIndex.from_tuples(col_tuples, names=("Dataset", "Split"))

# %%
latex_df

# %%
latex_df[["Arxiv papers", "IMDB"]]

# %%
print(
    latex_df.loc[["Number of documents", "Over threshold"]].T.to_latex(
        multicolumn_format="c", escape=True, multirow=True, column_format="llccc"
    )
)

# %%
only_ds = latex_df.loc[["Number of classes", "Metric", "Evaluated with"]]
only_ds = only_ds.drop(
    columns=[tup for tup in only_ds.columns if tup[1] == "Validation"]
)
only_ds.columns = only_ds.columns.droplevel(1)

# %%
only_ds

# %%
print(only_ds.T.to_latex(column_format="lcll"))

# %%
latex_df.columns.droplevel(1)

# %%
print(
    latex_df[["Arxiv papers", "IMDB", "OC"]].to_latex(
        multicolumn_format="c", escape=True, multirow=True, column_format="lcccccc"
    )
)

# %%
print(
    latex_df[["AAN", "S2ORC", "PAN"]].to_latex(
        multicolumn_format="c", escape=True, multirow=True, column_format="lcccccc"
    )
)

# %% [markdown]
# ## Distribution graphs

# %%
sns.histplot(
    stats["wiki"]["train"]["token_counts"],
    bins=[0, 384, 512, 768, 1024, 2048, 3072, 4096],
    stat="percent",
)

# %%
1.8e9 / 1.7e3

# %%
