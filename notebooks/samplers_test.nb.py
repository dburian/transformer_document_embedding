# %% [markdown]
# # Testing of samplers

# %%
from transformer_document_embedding.utils import tokenizers
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

# %%
ds = load_from_disk("/mnt/data/datasets/wikipedia_sample")["train"]
# %%
ds_sample = ds.select(range(min(100000, len(ds))))

# %%
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

# %%
effective_batch_size = 32
batch_size = 4
bucket_limits = [384, 512, 1024, 1536, 2048, 2560, 3072]

# %% [markdown]
# ### Correct lengths

# %%
dataloader = tokenizers.create_tokenized_data_loader(
    data=ds_sample,
    tokenizer=tokenizer,
    batch_size=batch_size,
    sampling="consistent",
    sampler_kwargs={
        "effective_batch_size": effective_batch_size,
        "bucket_limits": bucket_limits,
        "mega_batch_size": 33,
    },
)

# %%
len(ds_sample)

# %%
len(dataloader)

# %%
lengths = []
for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    assert torch.all(batch["length"] == torch.sum(batch["attention_mask"], axis=1))
    lengths.extend(
        [
            {
                "input": i * batch_size + j,
                "batch": i,
                "effective_batch": i // effective_batch_size,
                "length": length,
            }
            for j, length in enumerate(batch["length"].numpy(force=True))
        ]
    )

# %%
print(f"In DataLoader: {(i+1) * batch_size} elements, in dataset: {len(ds_sample)}")

# %%
lengths = pd.DataFrame(lengths)

# %%
sns.kdeplot(lengths, x="length", hue="effective_batch")

# %%
max_length = lengths["length"].max()
max_length

# %%
extended_bucket_limits = np.array(bucket_limits + [max_length + 1])
length_smaller_than_threshold = (
    lengths["length"].to_numpy()[:, None] < extended_bucket_limits[None, :]
)
bucket_idxs = len(extended_bucket_limits) - np.sum(
    length_smaller_than_threshold, axis=1
)

# %%
lengths["bucket"] = extended_bucket_limits[bucket_idxs].astype(np.int32)

# %%
lengths.describe()

# %%
lengths.dtypes

# %%
np.all(lengths["bucket"] > lengths["length"])

# %%
sns.boxplot(lengths, x="length", y="bucket", orient="h")

# %%
sns.histplot(lengths, x="bucket", bins=extended_bucket_limits)

# %%
sns.histplot(lengths, x="bucket", y="effective_batch")

# %%
sns.boxplot(lengths, x="bucket", y="effective_batch", orient="h")

# %%
sns.displot(lengths, x="bucket", row="effective_batch", hue="bucket")

# %%
# plt.plot(np.arange(len(average_lengths)), average_lengths)
# plt.plot(
#     [0, len(average_lengths) - 1],
#     [np.mean(ds_sample["length"]), np.mean(ds_sample["length"])],
# )

# %%
np.mean(ds_sample["length"])

# %%
np.std(ds_sample["length"])

# %% [markdown]
# ### Speed test

# %%
dataloader_default = tokenizers.create_tokenized_data_loader(
    data=ds_sample,
    tokenizer=tokenizer,
    batch_size=batch_size,
    sampling="default",
)

# %%
dataloader_consistent = tokenizers.create_tokenized_data_loader(
    data=ds_sample,
    tokenizer=tokenizer,
    batch_size=batch_size,
    sampling="consistent",
    sampler_kwargs={
        "effective_batch_size": effective_batch_size,
        "bucket_limits": bucket_limits,
        "mega_batch_size": 1000,
    },
)

# %%
for name, dataloader in {
    "default": dataloader_default,
    "consistent": dataloader_consistent,
}.items():
    for _ in tqdm(dataloader, desc=name):
        pass
