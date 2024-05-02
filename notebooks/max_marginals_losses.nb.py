# %% [markdown]
# # Validation analysis of composite losses

# %%
from transformer_document_embedding.datasets import col
from transformer_document_embedding.models.embedding_model import EmbeddingModel
import transformer_document_embedding.notebook_utils as ntb_utils
from transformer_document_embedding.utils.similarity_losses import (
    MaskedMSE,
    MaskedCosineDistance,
)
from datasets import load_from_disk
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import transformer_document_embedding.utils.training as train_utils
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from transformer_document_embedding.utils.tokenizers import create_tokenized_data_loader

# %%
ntb_utils.seaborn_defaults()

# %%
models = {
    "Longformer": lambda: ntb_utils.load_model_save(
        "../results/teacher_embedding:TeacherEmbedding/transformer:TransformerEmbedder/longformer_save/trained_model"
    ),
    "MSE": lambda: ntb_utils.load_model_save(
        "../hp_searches/glb_structural_basic_loss/h.k.s_h_k.l_t=mse/trained_model"
    ),
    "max-marginals;MSE;"
    r"$\gamma$=1.0": lambda: ntb_utils.load_model_save(
        "../hp_searches/glb_structural_max_marginals_loss/h.k.s_h_k.l_t=max_marginals_mse-h.k.s_h_k.m_m_l=1/trained_model"
    ),
    "max-marginals;MSE;"
    r"$\gamma$=1.5": lambda: ntb_utils.load_model_save(
        "../hp_searches/glb_structural_max_marginals_loss/h.k.s_h_k.l_t=max_marginals_mse-h.k.s_h_k.m_m_l=1.5/trained_model"
    ),
    "max-marginals;MSE;"
    r"$\gamma$=0.5": lambda: ntb_utils.load_model_save(
        "../hp_searches/glb_structural_max_marginals_loss/h.k.s_h_k.l_t=max_marginals_mse-h.k.s_h_k.m_m_l=0.5/trained_model"
    ),
    "cosine": lambda: ntb_utils.load_model_save(
        "../hp_searches/glb_structural_basic_loss/h.k.s_h_k.l_t=cos_dist/trained_model"
    ),
    "contrastive": lambda: ntb_utils.load_model_save(
        "../hp_searches/glb_structural_basic_loss/h.k.s_h_k.l_t=contrastive/trained_model"
    ),
    "max-marginals;cosine;"
    r"$\gamma$=0.5": lambda: ntb_utils.load_model_save(
        "../hp_searches/glb_structural_max_marginals_loss/h.k.s_h_k.l_t=max_marginals_cos_dist-h.k.s_h_k.m_m_l=0.5/trained_model"
    ),
}

# %%
models = {
    "final_cosine": ntb_utils.load_model_save(
        "../results/teacher_embedding:TeacherEmbedding/transformer:TransformerEmbedder/cos_final/trained_model"
    ),
    "final_just_mm_mse": ntb_utils.load_model_save(
        "../results/teacher_embedding:TeacherEmbedding/transformer:TransformerEmbedder/just_mm_mse_final/trained_model"
    ),
    "final_mm_mse_contextual": ntb_utils.load_model_save(
        "../results/teacher_embedding:TeacherEmbedding/transformer:TransformerEmbedder/mm_mse_contextual_final/trained_model"
    ),
}

# %%
val_corpus = load_from_disk("/mnt/data/datasets/val_corpus_500k/")
val_corpus


# %%
def get_dists(
    model: EmbeddingModel,
    device: str = "cuda",
    samples: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    assert isinstance(model, torch.nn.Module)

    device = torch.device(device)
    model.to(device)

    dists = {
        "mse": MaskedMSE().to(device).eval(),
        "cos": MaskedCosineDistance().to(device).eval(),
    }

    def get_dist(
        embeds: torch.Tensor, sbert_embeds: torch.Tensor, acc: dict[str, list[int]]
    ) -> None:
        for name, dist_fn in dists.items():
            dist_outputs = dist_fn(embeds, sbert_embeds)
            acc[name].extend(dist_outputs["loss"].numpy(force=True))

    model.eval()

    dataloader = create_tokenized_data_loader(
        val_corpus["validation"].select(range(samples)),
        batch_size=6,
        training=False,
        tokenizer=AutoTokenizer.from_pretrained("allenai/longformer-base-4096"),
    )

    positive_dists = {name: [] for name in dists}
    negative_dists = {name: [] for name in dists}

    with torch.inference_mode():
        for _, batch in tqdm(
            enumerate(dataloader), desc="Batches", total=len(dataloader)
        ):
            train_utils.batch_to_device(batch, device)
            model_outputs = model(**batch)
            get_dist(model_outputs[col.EMBEDDING], batch["sbert"], positive_dists)

            batch_size = batch["input_ids"].shape[0]
            for i in range(1, batch_size):
                get_dist(
                    model_outputs[col.EMBEDDING],
                    torch.roll(batch["sbert"], i),
                    negative_dists,
                )

    return positive_dists, negative_dists


def df_dists(models: dict[str, EmbeddingModel], samples: int) -> pd.DataFrame:
    dists = []
    for model_name, model in models.items():
        positives, negatives = get_dists(model(), samples=samples)
        positives = pd.DataFrame(positives)
        positives["type"] = "Positives"
        negatives = pd.DataFrame(negatives)
        negatives["type"] = "Negatives"

        model_dists = pd.concat([positives, negatives])
        model_dists["model"] = model_name
        dists.append(model_dists)

    return pd.concat(dists).melt(id_vars=["type", "model"])


# %%
dists = df_dists(models, 1000)

# %%
del models

# %%
fig, axes = plt.subplots(2, 1, figsize=(8.26, 9))

for ax, dist_name in zip(axes, ["cos", "mse"]):
    sns.violinplot(
        dists[dists["variable"] == dist_name],
        y="model",
        x="value",
        hue="type",
        density_norm="width",
        split=True,
        inner=None,
        ax=ax,
    )

    ax.get_legend().set_title("Distance to")
    ax.set_xlabel(f"{dist_name} distance")
    ax.set_ylabel("Model")


# %%
def rename(model):
    if "max-marginals" in model:
        return "max-margin" + model[len("max-marginals") :]
    return model


dists["model"] = dists["model"].apply(rename)


# %%
def rename_dist(dist):
    if dist == "cos":
        return "cosine"
    else:
        return "MSE"


dists["Distance type"] = dists["variable"].apply(rename_dist)

# %%
fig, axes = plt.subplots(2, 1, figsize=(8.26, 9))

for ax, dist_name in zip(axes, ["cos", "mse"]):
    sns.violinplot(
        dists[dists["variable"] == dist_name],
        y="model",
        x="value",
        hue="type",
        density_norm="width",
        split=True,
        inner=None,
        ax=ax,
    )

    ax.get_legend().set_title("Distance to")
    ax.set_xlabel("Cosine distance" if dist_name == "cos" else "Squared L2 distance")
    ax.set_ylabel("Model")

# %%
fig.savefig("../paper/img/composite_distances.pdf", bbox_inches="tight")

# %%
fig, axes = plt.subplots(3, 1)

for ax, model in zip(axes, mse_dists["model"].unique()):
    sns.violinplot(
        mse_dists[mse_dists["model"] == model],
        x="dist",
        y="model",
        hue="type",
        ax=ax,
        # dodge=False,
        split=True,
        inner="quart",
    )


# %%
sns.barplot(mse_dists, x="model", y="dist", hue="type", errorbar="se")

# %%
width = 8.26
height_per_model = 6.5 / (len(cos_models) + len(mse_models))

# %%
fig, ax = plt.subplots(figsize=(width, height_per_model * 3))

sns.violinplot(
    mse_dists,
    y="model",
    x="dist",
    hue="type",
    density_norm="width",
    split=True,
    inner=None,
    ax=ax,
)

ax.get_legend().set_title("Distance to")
ax.set_xlabel("Squared L2 distance")
ax.set_ylabel("Model")

# %%
fig, ax = plt.subplots(figsize=(width, height_per_model * 3))

sns.violinplot(
    mse_dists_cos,
    y="model",
    x="dist",
    hue="type",
    density_norm="width",
    split=True,
    inner=None,
    ax=ax,
)

ax.get_legend().set_title("Distance to")
ax.set_xlabel("Squared L2 distance")
ax.set_ylabel("Model")

# %%
fig.savefig("../paper/img/composite_mse_distances.pdf", bbox_inches="tight")

# %%
fig, ax = plt.subplots(figsize=(width, height_per_model * 4))

sns.violinplot(
    cos_dists,
    y="model",
    x="dist",
    hue="type",
    density_norm="width",
    split=True,
    inner=None,
    ax=ax,
)

sns.move_legend(ax, "upper left")
ax.get_legend().set_title("Distance to")
ax.set_xlabel("Cosine distance")
ax.set_ylabel("Model")

# %%
fig.savefig("../paper/img/composite_cos_distances.pdf", bbox_inches="tight")

# %%
cos_dists.groupby(["model", "type"])["dist"].median()
