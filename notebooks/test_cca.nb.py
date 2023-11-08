# %% [markdown]
# # Testing my CCA implementation

# %%
import numpy as np
from tqdm.auto import tqdm
import torch
import transformer_document_embedding.utils.torch.losses as losses
from sklearn.cross_decomposition import CCA

# %%
features = 100
samples = 1000
cca_dimensions = 50


# %%
def get_cca_diff():
    a, b = np.random.rand(samples, features), np.random.rand(samples, features)
    a_torch, b_torch = torch.from_numpy(a), torch.from_numpy(b)

    my_cca = losses.CCALoss(
        output_dimension=cca_dimensions, regularization_constant=0, epsilon=0
    )
    my_corr = -my_cca(a_torch, b_torch)["loss"]

    sklearn_cca = CCA(n_components=cca_dimensions, max_iter=1000)
    a_, b_ = sklearn_cca.fit_transform(a, b)
    sklearn_corr = (
        np.corrcoef(a_, b_, rowvar=False).diagonal(offset=cca_dimensions).sum()
    )

    return my_corr.numpy(force=True) - sklearn_corr


# %%
diffs = []

# %%
for _ in tqdm(range(20)):
    diffs.append(get_cca_diff())

# %%
np.mean(diffs), np.max(diffs), np.min(diffs)

# %%
diffs
