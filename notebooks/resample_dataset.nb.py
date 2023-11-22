# %% [markdown]
# # Create a resampled copy of dataset
#
# 1. Loads a `Dataset`
# 2. Initializes a sampler
# 3. Iterates over the sampler to the desired target length
# 4. Saves the resulting dataset to disk

# %%
from __future__ import annotations
from datasets import load_from_disk, Dataset, DatasetDict
import transformer_document_embedding.utils.training as train_utils

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterator, Any
    from torch.util.data import Sampler

# %%
DS_PATH = "/mnt/data/datasets/wikipedia_with_embeddings"
NEW_DS_PATH = "/mnt/data/datasets/wikipedia_sample"
NEW_DS_SIZE = 30000

# %%
ds = load_from_disk(DS_PATH)


# %%
def new_ds_gen(split: Dataset, sampler: Sampler) -> Iterator[Any]:
    for count, idx in enumerate(sampler, start=1):
        yield split[idx]

        if count == NEW_DS_SIZE:
            break


# %%
new_splits = {}
for split_name, split in ds.items():
    sampler = train_utils.ConsistentLenghtDistSampler(
        dataset=split,
        effective_batch_size=512,
        bucket_limits=[512 + i * 512 for i in range(8)],
    )

    new_split = Dataset.from_generator(
        new_ds_gen, gen_kwargs={"split": split, "sampler": sampler}
    )
    new_splits[split_name] = new_split
new_ds = DatasetDict(new_splits)

# %%
new_ds.save_to_disk(NEW_DS_PATH, max_shard_size="1GB")
