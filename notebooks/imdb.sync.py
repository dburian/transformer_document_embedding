# %% [markdown]
# # IMDB dataset
#
# %%
import random as rand

import datasets as ds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import transformers

rand.seed(42)

# %%
imdb = ds.load_dataset("imdb")

print(imdb)

# %%
train = imdb["train"]

train_df = pd.DataFrame(train)
train_df["text_wc"] = train_df["text"].apply(lambda text: len(text.split(" ")))

tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")
train_df["text_tc"] = train_df["text"].apply(lambda text: len(tokenizer.tokenize(text)))


train_df.describe()

# %%
train_df.hist()

# %%
for ex_ind in rand.choices(train_df.index, k=10):
    ex = train_df.iloc[ex_ind]
    print(f'label: {ex["label"]}, word count: {ex["text_wc"]}')
    print(ex["text"])
    print()

# %%
nintyfive_perc_wc = np.percentile(train_df["text_wc"].to_numpy(), 95)
nintyfive_perc_tc = np.percentile(train_df["text_tc"].to_numpy(), 95)
print(nintyfive_perc_wc)

fig, ax = plt.subplots()
sns.histplot(
    data=train_df[train_df["text_tc"] <= nintyfive_perc_tc],
    x="text_tc",
    ax=ax,
    label="95-percentile RoBerta token count",
)
sns.histplot(
    data=train_df[train_df["text_wc"] <= nintyfive_perc_wc],
    x="text_wc",
    ax=ax,
    label="95-percentile word count",
)

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles[1::2], fontsize="small")

# %%
perc_too_long = sum(train_df["text_tc"] >= 512) / len(train_df)

print(f"{perc_too_long*100:.2f}% of reviews are over 512 tokens.")
# %%
