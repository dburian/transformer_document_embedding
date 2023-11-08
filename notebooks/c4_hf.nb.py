# %%
from datasets import load_dataset_builder, load_dataset, config
import seaborn as sns

# %%
print(config.HF_DATASETS_CACHE)

# %%
builder = load_dataset_builder("c4", "en")

# %%
vars(builder.info)

# %%
train_files = {
    "train": "en/c4-train.0000*-of-01024.json.gz"
}  # , "validation": "en/c4-validation.*.json.gz"}
c4 = load_dataset(
    "allenai/c4",
    "en",
    data_files=train_files,
    cache_dir="~/.cache/huggingface/datasets",
)

# %%
c4 = c4.map(lambda sample: {"char_len": len(sample["text"])})

# %%
c4 = c4.map(lambda sample: {"word_len": len(sample["text"].split())})

# %%
sns.histplot(c4["train"]["word_len"], binwidth=100).set_xlim(0, 20000)
