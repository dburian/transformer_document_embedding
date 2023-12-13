# %% [markdown]
# # Testing my CCA implementation

# %%
import numpy as np
from tqdm.auto import tqdm
import torch
from transformer_document_embedding.utils import cca_losses
from sklearn.cross_decomposition import CCA
import pandas as pd
from itertools import product
import seaborn as sns
from cca_zoo.linear import CCA as ZooCCA, SPLS


# %%
def compute_sklearn_cca(a, b, cca_dim, tol=1e-6, max_iter=500, **_):
    try:
        sklearn_cca = CCA(n_components=cca_dim, max_iter=max_iter, tol=tol)
        a_, b_ = sklearn_cca.fit_transform(a, b)
        return np.corrcoef(a_, b_, rowvar=False).diagonal(offset=cca_dim).sum()
    except np.linalg.LinAlgError:
        return np.nan


def compute_my_cca(a, b, cca_dim, reg_constant=0, epsilon=1e-9, **_):
    a_torch, b_torch = torch.from_numpy(a), torch.from_numpy(b)

    my_cca = cca_losses.CCALoss(
        output_dimension=cca_dim,
        regularization_constant=reg_constant,
        epsilon=epsilon,
    )
    return -my_cca(a_torch, b_torch)["loss"]


def compute_zoo_cca(a, b, cca_dim, **_):
    model = ZooCCA(latent_dimensions=cca_dim)
    return model.fit((a, b)).score((a, b))


def compute_zoo_iter_cca(a, b, cca_dim, tol=1e-5, tau=0.1, **_):
    model = SPLS(tau=tau, tol=tol, latent_dimensions=cca_dim)
    return model.fit((model_view, dbow_view)).score((model_view, dbow_view))


cca_funcs = {
    "sklearn": compute_sklearn_cca,
    "torch": compute_my_cca,
    "zoo": compute_zoo_cca,
    "zoo_iter": compute_zoo_iter_cca,
}

# %% [markdown]
# ## Testing on random inputs

# %%
cca_dimensions = [512, 768]
samples_multiplier = [15]
features = 768
reps = 2

# %%
results = []

# %%
for cca_dim, samples_m in tqdm(
    product(cca_dimensions, samples_multiplier),
    desc="GS",
    total=len(cca_dimensions) * len(samples_multiplier),
):
    for idx in tqdm(range(reps), desc="Reps"):
        a, b = np.random.rand(samples_m * cca_dim, features), np.random.rand(
            samples_m * cca_dim, features
        )
        results.append(
            {
                "sklearn": compute_sklearn_cca(a, b, cca_dim),
                "mine": compute_my_cca(a, b, cca_dim).item(),
                "cca_dim": cca_dim,
                "samples": samples_m * cca_dim,
                "features": features,
                "rep_idx": idx,
            }
        )
res_df = pd.DataFrame(results)

# %%
res_df

# %%
res_df["diff"] = res_df["sklearn"] - res_df["mine"]

# %%
res_df["diff"]

# %%
sns.lineplot(res_df, y="diff", x="samples", hue="cca_dim")

# %%
sns.lineplot(res_df, y="mine", x="samples", hue="cca_dim")

# %% [markdown]
# ## Testing on real data

# %%
import os
from transformer_document_embedding.scripts import utils
from transformer_document_embedding.scripts.config_specs import (
    ExperimentalModelSpec,
    ExperimentalTaskSpec,
)
from tqdm.auto import tqdm

# %%
BASE_EXP_PATH = (
    "../results/teacher_embedding:TeacherEmbedding/transformer.student"
    ":TransformerStudent/sbert_torch_cca"
)
MODEL_SAVE_PATH = os.path.join(BASE_EXP_PATH, "model")
MODEL_CONFIG_PATH = os.path.join(BASE_EXP_PATH, "config.yaml")

# %%
config = utils.load_yaml(MODEL_CONFIG_PATH)

# %%
model = utils.init_type(ExperimentalModelSpec.from_dict(config["model"]))
model.load(MODEL_SAVE_PATH)
task = utils.init_type(ExperimentalTaskSpec.from_dict(config["task"]))

# %%
from transformer_document_embedding.utils import training, tokenizers
import pandas as pd
from time import time

# %%
train_kwargs = config["model"]["train_kwargs"]
val_dataloader = tokenizers.create_tokenized_data_loader(
    task.validation,
    model._batch_size,
    training=False,
    sampling=train_kwargs["dataloader_sampling"],
    tokenizer=model._tokenizer,
    sampler_kwargs={
        "effective_batch_size": model._batch_size
        * train_kwargs["grad_accumulation_steps"],
        "bucket_limits": train_kwargs["bucket_limits"],
    },
)


# %%
val_outputs = []

model._model.to("cuda")
for batch in tqdm(val_dataloader, desc="Val batches", total=len(val_dataloader)):
    training.batch_to_device(batch, "cuda")

    with torch.no_grad():
        outputs = model._model(**batch)

    model_views = outputs["static_projected_view1"].numpy(force=True)
    dbow_views = outputs["static_projected_view2"].numpy(force=True)
    for idx in range(batch["input_ids"].size(0)):
        val_outputs.append(
            {
                "model": model_views[idx],
                "dbow": dbow_views[idx],
            }
        )

val_outputs = pd.DataFrame(val_outputs)

# %%
val_outputs


# %%
def time_calc(callable):
    time_before = time()
    result = callable()
    time_after = time()
    return result, time_after - time_before


# %%
results = []

# %%
cca_dim = 256
window = 256 * 4
tol = 1e-5
max_iter = 1000
reg_constant = 0
epsilon = 1e-14
computation_type = "true"

# %%
model_view = np.vstack(val_outputs["model"][-window:])
dbow_view = np.vstack(val_outputs["dbow"][-window:])

commons = {
    "cca_dim": cca_dim,
    "computation_type": computation_type,
    "window": window,
}

if computation_type == "torch":
    cca, seconds = time_calc(
        lambda: compute_my_cca(
            model_view, dbow_view, cca_dim, reg_constant=reg_constant, epsilon=epsilon
        )
    )
    results.append(
        {
            **commons,
            "reg_constant": reg_constant,
            "epsilon": epsilon,
            "cca": cca.item(),
            "time": seconds,
        }
    )
elif computation_type == "zoo":
    cca, seconds = time_calc(lambda: compute_zoo_cca(model_view, dbow_view, cca_dim))
    results.append({**commons, "cca": cca, "time": seconds})
elif computation_type == "zoo_iter":
    cca, seconds = time_calc(
        lambda: SPLS(tau=0.1, tol=tol, latent_dimensions=cca_dim)
        .fit((model_view, dbow_view))
        .score((model_view, dbow_view))
    )
    results.append(
        {
            **commons,
            "cca": cca,
            "time": seconds,
            "tol": tol,
        }
    )
elif computation_type == "sklearn":
    cca, seconds = time_calc(
        lambda: compute_sklearn_cca(
            model_view,
            dbow_view,
            cca_dim,
            tol=tol,
            max_iter=max_iter,
        )
    )
    results.append(
        {**commons, "tol": tol, "max_iter": max_iter, "cca": cca, "time": seconds}
    )
elif computation_type == "true":
    true_tol = 1e-5
    true_max_iter = 10000000000
    cca, seconds = time_calc(
        lambda: compute_sklearn_cca(
            model_view,
            dbow_view,
            cca_dim,
            tol=true_tol,
            max_iter=true_max_iter,
        )
    )
    results.append(
        {
            **commons,
            "tol": true_tol,
            "max_iter": true_max_iter,
            "cca": cca,
            "time": seconds,
        }
    )

# %%
results_pd = pd.DataFrame(results)
results_pd

# %%
fine_tuning_epsilon_impossible = results_pd
fine_tuning_epsilon_impossible.to_csv("finetuning_eps_impossible.csv")

# %%
fine_tuning_epsilon_impossible

# %% [markdown]
# ### Testing correspondence of sklearn's CCA across dimensionality and window sizes

# %%
import math
import itertools


# %%
def get_cross_corr(a, b):
    return (
        np.corrcoef(a, b, rowvar=False).diagonal(offset=a.shape[1]).sum() / a.shape[1]
    )


# %%
data_sizes = [1280 * 4]
features = [256]
fracs = [0.05, 0.2, 0.5, 0.8, 1]
reps = 1

# %%
corr_cca_results = []
data_pairs = []

# %%
for data_size, feats, frac in tqdm(
    itertools.product(data_sizes, features, fracs),
    total=len(data_sizes) * len(features) * len(fracs),
    desc="Iters",
):
    for _ in range(reps):
        a = np.random.rand(data_size, feats)
        b = a.copy() * 8 + 6.5

        inds = np.random.choice(data_size, size=math.floor(frac * data_size))
        a[inds] = np.random.rand(inds.shape[0], feats)
        b[inds] = np.random.rand(inds.shape[0], feats)

        data_pairs.append((a, b))
        corr_cca_results.append(
            {
                "data_size": data_size,
                "feats": feats,
                "frac": frac,
                "corr": get_cross_corr(a, b),
                "cca": compute_sklearn_cca(a, b, feats, max_iter=5000) / feats,
            }
        )


# %%
corr_cca_results_df = pd.DataFrame(corr_cca_results)
corr_cca_results_df

# %%

sns.lineplot(corr_cca_results_df, y="cca", x="frac", hue="data_size")
sns.lineplot(corr_cca_results_df, y="corr", x="frac", hue="data_size")

# %%
model_view = np.vstack(val_outputs["model"])
dbow_view = np.vstack(val_outputs["dbow"])
data_size = dbow_view.shape[0]

# %%
data_size

# %%
window_results = []

# %%
window_sizes = [512]
dimensions = [128]
reps = 2
tol = 1e-5
max_iter = 1000

# %%
for data_cca, (model_view, dbow_view) in zip(
    corr_cca_results_df["cca"], data_pairs, strict=True
):
    for window in tqdm(window_sizes, desc="Windows"):
        for dim in tqdm(dimensions, desc="Dims"):
            for _ in tqdm(range(reps), desc="Reps"):
                inds = np.random.choice(model_view.shape[0], size=window, replace=False)
                cca, seconds = time_calc(
                    lambda: compute_sklearn_cca(
                        model_view[inds],  # noqa: B023
                        dbow_view[inds],  # noqa: B023
                        dim,  # noqa: B023
                        tol=tol,
                        max_iter=max_iter,
                    )
                )
                window_results.append(
                    {
                        "data_cca": data_cca * model_view.shape[1],
                        "cca": cca,
                        "seconds": seconds,
                        "tol": tol,
                        "max_iter": max_iter,
                        "dim": dim,
                        "window": window,
                    }
                )

# %%
window_results_df = pd.DataFrame(window_results)
window_results_df

# %%
window_results_df["id"] = window_results_df.apply(
    lambda row: f"{row['dim']:.0f}x{row['window']:.0f}", axis=1
)
window_results_df["data_cca_scaled"] = window_results_df["data_cca"] / 256
window_results_df["cca_scaled"] = window_results_df["cca"] / window_results_df["dim"]

# %%
window_results_df.head()

# %%
sns.lineplot(window_results_df, y="cca_scaled", x="data_cca_scaled", hue="id")

# %%
window_results_df["diff"] = window_results_df["data_cca"] - window_results_df["cca"]
window_results_df["diff_scaled"] = (
    window_results_df["data_cca_scaled"] - window_results_df["cca_scaled"]
)

# %%
sns.lineplot(window_results_df, y="diff", x="data_cca", hue="id")

# %%
sns.lineplot(window_results_df, y="diff_scaled", x="data_cca", hue="id")

# %%
sns.lineplot(window_results_df, y="seconds", x="id")

# %% [markdown]
# ### Finding the best CCA model

# %%
import math

# %%
data = (np.vstack(val_outputs["model"]), np.vstack(val_outputs["dbow"]))

# %%
validation_size = math.floor(0.5 * data[0].shape[0])
validation_size


# %%
def sklearn_cca(train_data, val_data, cca_dim, tol=1e-5, max_iter=1000, **_):
    model = CCA(n_components=cca_dim, tol=tol, max_iter=max_iter)
    try:
        model.fit(*train_data)

        val_data_ = model.transform(*val_data)

        return np.corrcoef(*val_data_, rowvar=False).diagonal(offset=cca_dim).sum()
    except np.linalg.LinAlgError:
        return np.nan


def zoo_cca(train_data, val_data, cca_dim, eps=1e-9, **_):
    model = ZooCCA(latent_dimensions=cca_dim, eps=eps)
    model.fit(train_data)

    return model.score(val_data)


def zoo_iter_cca(train_data, val_data, cca_dim, tol=1e-3, tau=None, **_):
    model = SPLS(latent_dimensions=cca_dim, tau=tau, tol=tol)
    model.fit(train_data)

    return model.score(val_data)


def torch_cca(train_data, val_data, cca_dim, reg_constant=0, eps=1e-9, **_):
    a, b = torch.from_numpy(val_data[0]), torch.from_numpy(val_data[0])

    model = cca_losses.CCALoss(
        output_dimension=cca_dim,
        regularization_constant=reg_constant,
        epsilon=eps,
    )
    return -model(a, b)["loss"]


cca_models = {
    "sklearn": sklearn_cca,
    "zoo": zoo_cca,
    "zoo_iter": zoo_iter_cca,
    "torch": torch_cca,
}

# %%
kwargs = {
    "tol": 1e-5,
    "tau": None,
    "max_iters": 1000,
    "eps": 1e-9,
    "reg_constant": 0,
}

# %%
gs_res = []

# %%
all_inds = np.random.permutation(np.arange(data[0].shape[0]))
train_ids = all_inds[:-validation_size]
val_ids = all_inds[-validation_size:]

train = (data[0][train_ids], data[1][train_ids])
val = (data[0][val_ids], data[1][val_ids])

for cca_dim in [256, 512, 768]:
    for func_name, func in tqdm(cca_models.items(), desc="Models"):
        val_cca, seconds = time_calc(
            lambda: func(train, val, cca_dim, **kwargs),  # noqa: B023
        )
        gs_res.append(
            {
                **kwargs,
                "cca_dim": cca_dim,
                "model": func_name,
                "val_cca": val_cca,
                "seconds": seconds,
            }
        )

# %%
gs_res_df = pd.DataFrame(gs_res)
gs_res_df

# %%
272 / 256
