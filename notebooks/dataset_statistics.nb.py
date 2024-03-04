# %% [markdown]
# # Generating dataset statistics

# %%
from transformer_document_embedding.datasets.teacher_embedding import TeacherEmbedding
from transformer_document_embedding.datasets import col
import numpy as np
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# %%
plt.rc("figure", figsize=(16, 10))

# %%
NUM_PROC = 6

# %%
datasets = {
    "wiki": TeacherEmbedding(path="/mnt/data/datasets/wikipedia_resampled_eval/"),
    "wiki_shortened": TeacherEmbedding(
        path="/mnt/data/datasets/wikipedia_resampled_eval/",
        data_size_limit={"train": 150000, "validation": 7680},
    ),
}


# %%
def gen_stats_for_split(split):
    def get_stats_per_doc(doc):
        return {
            "words": len(doc[col.TEXT].split()),
        }

    stats = split.map(get_stats_per_doc, num_proc=NUM_PROC)

    return {
        "median_wpd": np.median(stats["words"]).item(),
        "mean_wpd": np.mean(stats["words"]).item(),
        "std_wpd": np.std(stats["words"]).item(),
        "docs": len(stats),
        "median_tpd": np.median(stats["length"]).item(),
        "mean_tpd": np.mean(stats["length"]).item(),
        "std_tpd": np.std(stats["length"]).item(),
        "word_counts": stats["words"],
        "token_counts": stats["length"],
    }


# %%
stats = {}
for ds_name, ds in tqdm(datasets.items(), desc="Dataset"):
    stats[ds_name] = {}
    for split_name, split in ds.splits.items():
        stats[ds_name][split_name] = gen_stats_for_split(split)

# %%
stats_df = pd.DataFrame(
    {
        (ds, split): {
            stat: value
            for stat, value in stats[ds][split].items()
            if stat not in ["word_counts", "token_counts"]
        }
        for ds in stats.keys()
        for split in stats[ds].keys()
    }
)

# %%
stats_df

# %%
counts_df = pd.DataFrame(
    [
        {
            "ds": ds,
            "split": split,
            "count": count[: count.find("_")],
            "value": value,
            "doc": i,
        }
        for ds in stats
        for split in stats[ds]
        for count in ["word_counts", "token_counts"]
        for i, value in enumerate(stats[ds][split][count])
    ]
)

# %%
counts_df.head()

# %%
ax = sns.histplot(counts_df[counts_df["count"] == "word"], x="value", hue="split")
ax.set_xlim((-100, 4000))

# %%
ax = sns.histplot(counts_df[counts_df["count"] == "token"], x="value", hue="split")
ax.set_xlim((-100, 4000))

# %%
tmp = counts_df[(counts_df["ds"] == "wiki") & (counts_df["count"] == "token")].copy()
batch_size = 1024
tmp["batch"] = tmp["doc"] // batch_size

# %%
sns.lineplot(tmp, y="value", x="batch", hue="split")
