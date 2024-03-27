# %%
from transformer_document_embedding.scripts import config_specs
import transformer_document_embedding.utils.training as train_utils
from transformer_document_embedding.utils.tokenizers import create_tokenized_data_loader
import yaml

# %%
config = config_specs.ExperimentSpec.from_dict(
    yaml.safe_load(
        """
model:
  module: transformer:TransformerEmbedder
  kwargs:
    transformer_name: allenai/longformer-base-4096
    #transformer_name: sentence-transformers/all-mpnet-base-v2
    pooler_type: mean

head:
  module: structural_contextual_head:StructuralContextualHead
  kwargs:
    max_structural_length: null
    lam: 1
    structural_head_kwargs:
      loss_type: max_marginals_cos_dist
      max_marginals_lam: 1
    contextual_head_kwargs: null



train_pipeline:
  kind: student
  kwargs:
    batch_size: 4
    epochs: 1
    warmup_steps: 100
    grad_accumulation_steps: 2
    weight_decay: 0.01
    fp16: True
    lr_scheduler_type: cos
    lr: 1.0e-4
    max_grad_norm: 1.0
    log_every_step: 1
    validate_every_step: 100
    save_best: True
    patience: 2
    global_attention_type: cls
    dataloader_sampling: consistent
    sampler_kwargs:
      bucket_limits: [512, 1024, 2048, 4096]

dataset:
  module: teacher_embedding:TeacherEmbedding
  kwargs:
    path: /mnt/data/datasets/wikipedia_resampled_eval
    contextual_embedding_col: dbow
    structural_embedding_col: hf_sbert
    data_size_limit:
      train: 1000
      validation: 1000
      test: 50
"""
    )
)

# %%
model = config.model.initialize()

# %%
learned_model = config.model.initialize()

# %%
learned_model.load_weights(
    "../hp_searches/depth_loss/m.k.d_l_k.l_t=contrastive_cos_dist/model/model"
)

# %%
head = config.head.initialize(model)

# %%
dataset = config.dataset.initialize()

# %%
import torch
from transformers import AutoTokenizer

# %%
dataloader = create_tokenized_data_loader(
    dataset.splits["validation"],
    batch_size=config.train_pipeline.kwargs["batch_size"],
    tokenizer=AutoTokenizer.from_pretrained(config.model.kwargs["transformer_name"]),
)

# %%
dataset.splits["train"]

# %%
from transformer_document_embedding.utils.similarity_losses import (
    MaskedMSE,
    MaskedCosineDistance,
)

# %%
dissimilarity = MaskedMSE()

# %%
dissimilarity = MaskedCosineDistance(dim=0)

# %%
from tqdm.auto import tqdm

# %%
prev_batch = next(iter(dataloader))
train_utils.batch_to_device(prev_batch)

# %%
embed_model = learned_model

device = "cuda"

embed_model.to(device)
head.to(device)

dists_list = []
embed_model.eval()
head.eval()
with torch.inference_mode():
    for batch_num, batch in tqdm(
        enumerate(dataloader), desc="Batches", total=len(dataloader)
    ):
        train_utils.batch_to_device(batch, device)
        model_outputs = embed_model(**batch)
        head_outputs = head(**model_outputs, **batch)
        for i, embed in enumerate(model_outputs["embedding"]):
            for j, sbert_embed in enumerate(batch["structural_embed"]):
                dis_sim = (
                    dissimilarity(embed, sbert_embed)["loss"].numpy(force=True).item()
                )
                dists_list.append(
                    {
                        "dist": dis_sim,
                        "batch": batch_num,
                        "type": "positive" if i == j else "negative",
                    }
                )
        prev_batch = batch

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# %%
plt.rc("figure", figsize=(16, 10))

# %%
dists = pd.DataFrame(dists_list)

# %%
ax = sns.histplot(dists, x="dist", hue="type", binwidth=0.001)
# ax.set_xlim((-0.03, 0.03))

# %%
ax = sns.histplot(dists, x="dist", hue="type", binwidth=0.001)

# %%
dists.groupby("type")["dist"].agg(["mean", "std", "max", "min"])

# %%
dists.groupby("type")["dist"].agg(["mean", "std", "max", "min"])

# %%
dists_in_batch = dists.groupby("batch")["dist"].agg(["mean", "std"]).reset_index()

# %%
sns.lineplot(dists_in_batch["std"])

# %%
sns.lineplot(dists, x="batch", y="dist", hue="type")

# %%
sns.lineplot(dists, x="batch", y="dist", hue="type")

# %%
random_dists = []

# %%
import random

# %%
sbert_embeds = dataset.splits["train"].with_format("torch")

# %%
with torch.inference_mode():
    for _ in tqdm(range(10000)):
        i, j = (
            random.randint(0, len(sbert_embeds) - 1),
            random.randint(0, (len(sbert_embeds) - 1)),
        )
        a, b = sbert_embeds.select([i, j])
        a, b = a["structural_embed"], b["structural_embed"]
        random_dists.append(dissimilarity(a, b)["loss"].item())

# %%
sns.histplot(random_dists)
