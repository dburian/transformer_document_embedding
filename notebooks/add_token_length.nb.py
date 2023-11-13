# %% [markdown]
# # Add token length to a Dataset
#
# 1. Loads a `Dataset`
# 2. Applies a given tokenizer
# 3. Saves the number of tokens (excluding sep, pad, cls, unk tokens) to a column
# 4. Saves the `Dataset`

# %%
from __future__ import annotations

from datasets import load_from_disk
from transformers import AutoTokenizer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

# %%
DS_PATH = "../data/wikipedia_with_eval"
NEW_DS_PATH = "../data/wikipedia_with_eval_with_lengths"
TOKENIZER_PATH = "allenai/longformer-base-4096"

# %%
ds = load_from_disk(DS_PATH)

# %%
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)


# %%
def get_length(inputs: dict[str, Any]) -> dict[str, Any]:
    tokenized = tokenizer(
        inputs["text"],
        padding="longest",
        truncation="longest_first",
        return_tensors="np",
    )
    lengths = tokenized["attention_mask"].sum(axis=1)
    return {"length": lengths}


# %%
ds = ds.map(get_length, batched=True, num_proc=12)

# %%
ds.save_to_disk(NEW_DS_PATH, max_shard_size="1GB", num_proc=12)
