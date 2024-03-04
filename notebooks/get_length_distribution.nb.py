from datasets import load_from_disk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

plt.rcParams["figure.figsize"] = (20,10)

original = load_from_disk('./wikipedia_original')
resampled = load_from_disk('./wikipedia_resampled')
resampled_eval = load_from_disk('./wikipedia_resampled_eval')

dss = {
    'original': original,
    'resampled': resampled,
    'resampled_eval': resampled_eval
}

for name, ds in dss.items():
    mean_train_length = sum(ds['train']['length'])/len(ds['train'])
    mean_val_length = np.nan
    if 'validation' in ds:
        mean_val_length = sum(ds['validation']['length'])/len(ds['train'])

    print(f'ds: {name}, train length: {mean_train_length}, val length: {mean_val_length}')

# +
batch_size = 4096
for name, ds in tqdm(dss.items(), desc='Datasets'):
    splits = ['train']
    if 'validation' in ds:
        splits.append('validation')

    for split in tqdm(splits, 'Splits'):
        ys = [batch['length'].mean() for batch in tqdm(ds[split].with_format('np').iter(batch_size), desc='Batches')]
        xs = np.arange(len(ys))
        plt.plot(xs, ys, label=f'{name}_{split}')
    
plt.legend()
# -

fig = plt.gcf()

fig.savefig('./lengths.png')

# !pip install ipywidgets -U
