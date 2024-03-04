# # Process wikipedia dataset
#
# 1. Add dbow and sbert embeddings (not part of this notebook, see `generate_embeddings` script)
# 2. Add length
# 3. Shuffle to get roughly even length -- for the end of the dataset to have roughly the same length
# 4. Resample to get more even length -- for the 
# 5. Generate validation split
# 6. Generate short version

ORIGINAL_DS_PATH = '../wikipedia_original'
RESAMPLED_DS_PATH = '../wikipedia_resampled'
RESAMPLED_EVAL_DS_PATH = '../wikipedia_resampled_eval'
RESAMPLED_EVAL_SHORT_DS_PATH = '../wikipedia_resampled_eval_short'


NUM_PROC = 24
TOKENIZER = 'sentence-transformers/all-mpnet-base-v2'
VAL_SIZE_PERC = 5
SHORT_THRES = 384

# +
from __future__ import annotations
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset, DatasetDict
from tqdm.auto import tqdm
import numpy as np
from typing import TYPE_CHECKING
from datasets import config
import os
import matplotlib.pyplot as plt
from math import floor

if TYPE_CHECKING:
    from typing import Any
# -

config.HF_CACHE_HOME

os.getenv("TMPDIR")

orig_ds = load_from_disk(ORIGINAL_DS_PATH)

orig_ds

# ## 2. Add length

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)


def get_length(inputs: dict[str, Any]) -> dict[str, Any]:
    tokenized = tokenizer(
        inputs["text"],
        padding="longest",
        truncation=False,
        return_tensors="np",
    )
    lengths = tokenized["attention_mask"].sum(axis=1)
    return {"length": lengths}


orig_ds = orig_ds.map(get_length, batched=True, num_proc=NUM_PROC)

# ## 3. Shuffle

orig_ds = orig_ds.shuffle() #.flatten_indices(num_proc=NUM_PROC)


def plot_length_dist(ds):
    batch_size = 4096 * 4
    splits = ['train']
    if 'validation' in ds:
        splits.append('validation')

    for split in tqdm(splits, 'Splits'):
        batch_iter = tqdm(
            ds[split].select_columns('length').with_format('np').iter(batch_size),
            desc='Batches',
            total=len(ds[split]) // batch_size + 1
        )
        ys = [batch['length'].mean() for batch in batch_iter]
        xs = np.arange(len(ys)) * batch_size
        plt.plot(xs, ys, label=f'{split}')
    
    plt.legend()


plot_length_dist(orig_ds)

# ## 4. Resample

import transformer_document_embedding.utils.torch.training as train_utils


def new_ds_gen(split: Dataset, sampler: Sampler) -> Iterator[Any]:
    for count, idx in enumerate(sampler, start=1):
        yield split[idx]


def resample(ds):
    new_splits = {}
    for split_name, split in ds.items():
        sampler = train_utils.ConsistentLenghtDistSampler(
            dataset=split,
            effective_batch_size=1024,
            mega_batch_size=10000,
            bucket_limits=[512 + i * 512 for i in range(8)],
        )
    
        new_split = Dataset.from_generator( 
            new_ds_gen,
            gen_kwargs={"split": split, "sampler": sampler},
            num_proc=NUM_PROC,
        )
        new_splits[split_name] = new_split
    return DatasetDict(new_splits)


resampled_ds = resample(orig_ds)

plot_length_dist(resampled_ds)

resampled_ds.save_to_disk(RESAMPLED_DS_PATH, max_shard_size='1GB', num_proc=NUM_PROC)


# ## 5. Create validation split

def add_validation_split(ds):
    train_size = len(ds['train'])
    validation_size = floor(train_size * VAL_SIZE_PERC / 100)
    val = ds['train'].select(range(train_size - validation_size, train_size))
    train = ds['train'].select(range(train_size - validation_size))
    return DatasetDict({'train': train, 'validation': val})


resampled_eval = add_validation_split(resampled_ds)

resampled_eval

plot_length_dist(resampled_eval)

resampled_eval.save_to_disk(RESAMPLED_EVAL_DS_PATH, max_shard_size='1GB', num_proc=NUM_PROC)


# ## 6. Create short version

def shorten(ds):
    return ds.filter(
        lambda lengths: np.array(lengths) <= SHORT_THRES,
        batched=True,
        num_proc=NUM_PROC,
        input_columns='length',
    )


resampled_eval_short = shorten(resampled_eval)

plot_length_dist(resampled_eval_short)

resampled_eval_short.save_to_disk(RESAMPLED_EVAL_SHORT_DS_PATH, max_shard_size='1GB', num_proc=NUM_PROC)
