from datasets import load_from_disk
import numpy as np

DS_PATH = '../wikipedia_resampled_eval'
NEW_DS_PATH = '../wikipedia_resampled_eval_shorten'

ds = load_from_disk(DS_PATH)

ds = ds.filter(
    lambda lengths: np.array(lengths) < 384,
    batched=True,
    num_proc=12,
    input_columns='length',
)

ds.save_to_disk(NEW_DS_PATH, num_proc=12)
