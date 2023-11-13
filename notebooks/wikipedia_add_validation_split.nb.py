# # Script to create validation set for Wikipedia
#
# Validation split as the **last 0.1%** of train split

from datasets import load_from_disk, DatasetDict, Dataset
from math import floor

DS_PATH = '/storage/brno2/home/dburian/repos/tde/results/teacher_embedding:TeacherEmbedding/sbert.wiki:SBERTWikipediaSimilarities/2023-10-30_125908/embeddings'


ds = load_from_disk(DS_PATH)

len(ds['train'])

ds['train']

train_size = len(ds['train'])
validation_size = floor(train_size * 0.001)
validation_size

val = ds['train'].select(range(train_size - validation_size, train_size))

val

train = ds['train'].select(range(train_size - validation_size))

train

new_ds = DatasetDict({'train': train, 'validation': val})

new_ds.save_to_disk('../data/wikipedia_with_eval', max_shard_size='1GB')
