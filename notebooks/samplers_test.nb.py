# %% [markdown]
# # Testing of samplers

# %%
from __future__ import annotations
from transformer_document_embedding.utils import tokenizers
from datasets import load_from_disk
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# %%
plt.rc("figure", figsize=(16, 10))

# %%
ds = load_from_disk("/mnt/data/datasets/wikipedia_resampled_eval")["train"]
# %%
ds_sample = ds.select(range(min(15000, len(ds))))

# %%
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

# %%
batch_size = 6
effective_batch_size = batch_size * 4
bucket_limits = [512, 1024, 3072]

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
        "mega_batch_size": 4096,
    },
)

# %%
len(ds_sample)

# %%
lengths = []
extended_buckets = np.array(bucket_limits + [float("inf")])
for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    # bucket_sizes = {
    #     f"length_dist_{i}": bucket_size
    #     for i, bucket_size in enumerate(dataloader.sampler._buffer.bucket_sizes)
    # }
    lengths.extend(
        [
            {
                "input": i * batch_size + j,
                "batch": i,
                "effective_batch": i // effective_batch_size,
                "length": length,
                "bucket": len(extended_buckets) - np.sum(length < extended_buckets),
                # **bucket_sizes,
            }
            for j, length in enumerate(batch["length"].numpy(force=True))
        ]
    )

# %%
print(f"In DataLoader: {len(lengths)} elements, in dataset: {len(ds_sample)}")

# %%
lengths = pd.DataFrame(lengths)

# %%
ax = sns.kdeplot(lengths, x="length", hue="effective_batch")


# %%
in_batches = lengths.groupby("batch")["length"].agg("mean")
in_batches = in_batches.reset_index()

# %%
in_eff_batches = lengths.groupby("effective_batch")["length"].agg("mean").reset_index()
in_eff_batches

# %%
sns.lineplot(in_batches, x="batch", y="length")

# %%
sns.lineplot(in_eff_batches, x="effective_batch", y="length")

# %%
ax = sns.lineplot(lengths, x="input", y="length_dist_0")
for i in range(1, 9):
    ax = sns.lineplot(lengths, x="input", y=f"length_dist_{i}", ax=ax)
ax.set_yscale("log")

# %%
extended_bucket_limits = np.array(bucket_limits + [float("inf")])
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
sns.lineplot(lengths[lengths["input"] < 10000], y="length", x="input")

# %%
ax = sns.histplot(
    lengths,
    x="length",
    hue="effective_batch",
    bins=extended_bucket_limits,
    element="step",
)

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
