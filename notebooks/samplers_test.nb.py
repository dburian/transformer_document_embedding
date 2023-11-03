# %% [markdown]
# # Testing of samplers

# %%
import transformer_document_embedding.utils.torch.training as train_utils
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

# %%
ds = load_from_disk("/mnt/data/datasets/wikipedia_sample")["train"]
# %%
ds_sample = ds.select(range(min(30000, len(ds))))

# %%
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

# %%
effective_batch_size = 32
batch_size = 4
bucket_limits = [384, 512, 1024, 1536, 2048, 2560, 3072]

# %%
dataloader = train_utils.create_tokenized_data_loader(
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
sns.kdeplot(lengths, x="length", hue="effective_batch_size")

# %%
a = lengths["length"].to_numpy()[:, None] < np.array(bucket_limits)[None, :]
a[:30]

# %%

# %%
lengths["length"][:30]

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
