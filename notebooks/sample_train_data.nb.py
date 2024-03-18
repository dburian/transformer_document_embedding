# %% [markdown]
# # Create smaller training corpus by sampling the big one
#
# 1. Load and randomly sample Wikipedia and Realnews datasets
# 2. Concatenate samples
# 3. Shuffle & add ids
# 4. Add SBERT length
# %%
from __future__ import annotations
from transformers import AutoTokenizer
from datasets import load_from_disk, DatasetDict
import numpy as np
from typing import TYPE_CHECKING
from datasets import concatenate_datasets


if TYPE_CHECKING:
    from typing import Any

# %%
WIKI_PATH = ""
RN_PATH = ""
NEW_CORPUS_PATH = ""
NUM_PROC = 24
TOKENIZER = "sentence-transformers/all-mpnet-base-v2"
VAL_SIZE_PERC = 5
SHORT_THRES = 384
RANDOM_SEED = 42
CORPUS_TRAIN_LENGTH = 500000
CORPUS_VAL_LENGTH = 10000

# %% [markdown]
# ## 1. Randomly sample both dataset

# %%
wikipedia = load_from_disk(WIKI_PATH)

# %%
realnews = load_from_disk(RN_PATH)

# %%
rng_gen = np.random.default_rng(RANDOM_SEED)

# %%
total_idxs = CORPUS_TRAIN_LENGTH // 2 + CORPUS_VAL_LENGTH // 2

wiki_indxs = rng_gen.choice(len(wikipedia["train"]), total_idxs)
wiki_train_indxs = wiki_indxs[: CORPUS_TRAIN_LENGTH // 2]
wiki_val_indxs = wiki_indxs[CORPUS_TRAIN_LENGTH // 2 :]

print(len(wiki_val_indxs))

# %%
wiki_val = DatasetDict(
    {
        "train": wikipedia["train"].select(wiki_train_indxs),
        "validation": wikipedia["train"].select(wiki_val_indxs),
    }
)

# %%
realnews_train_indxs = rng_gen.choice(len(realnews["train"]), CORPUS_TRAIN_LENGTH // 2)
realnews_val_indxs = rng_gen.choice(len(realnews["validation"]), CORPUS_VAL_LENGTH // 2)

# %%
realnews_val = DatasetDict(
    {
        "train": realnews["train"].select(realnews_train_indxs),
        "validation": realnews["validation"].select(realnews_val_indxs),
    }
)

# %%
wiki_val

# %%
realnews_val

# %% [markdown]
# ## 2. Concatenate samples

# %%
realnews_extra_cols = [
    "summary",
    "authors",
    "publish_date",
    "status",
    "url_used",
    "domain",
    "warc_date",
]

val_ds = DatasetDict(
    {
        "train": concatenate_datasets(
            [
                wiki_val["train"].remove_columns("id"),
                realnews_val["train"].remove_columns(realnews_extra_cols),
            ]
        ),
        "validation": concatenate_datasets(
            [
                wiki_val["validation"].remove_columns("id"),
                realnews_val["validation"].remove_columns(realnews_extra_cols),
            ]
        ),
    }
)

# %%
val_ds

# %% [markdown]
# ## 3. Shuffle & add ids

# %%
val_ds = val_ds.shuffle()

# %%
begin_id = 0


def map_fn(_, idx: int) -> dict[str, Any]:
    return {"id": idx + begin_id}


for name, split in val_ds.items():
    val_ds[name] = split.map(map_fn, with_indices=True)
    begin_id += len(val_ds[name])

# %% [markdown]
# ## 4. Add SBERT length

# %%
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)


# %%
def get_length(inputs: dict[str, Any]) -> dict[str, Any]:
    tokenized = tokenizer(
        inputs["text"],
        padding="longest",
        truncation=False,
        return_tensors="np",
        add_special_tokens=False,
        verbose=False,
    )
    lengths = tokenized["attention_mask"].sum(axis=1)
    return {"length": lengths}


# %%
val_ds = val_ds.map(get_length, batched=True, num_proc=NUM_PROC, batch_size=128)

# %%
val_ds.save_to_disk(NEW_CORPUS_PATH, num_proc=NUM_PROC)
