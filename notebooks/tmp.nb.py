# %%
from transformer_document_embedding.experiments.config import ExperimentConfig

# %%
exp_path = (
    "../results/teacher_embedding:TeacherEmbedding/"
    "longformer.student:LongformerStudent/private_galago"
)

# %%
config = ExperimentConfig.from_yaml(exp_path + "/config.yaml", "../results")

# %%
model = config.get_model_type()(**config.values["model"]["kwargs"])


# %%
model.save("./longformer_save")

# %%
model.load(exp_path + "/model/checkpoint")

# %% [markdown]
# ----
# ### Testing just DCCA

# %%
import transformer_document_embedding.utils.losses as losses
import torch
from torchviz import make_dot

# %%
model = losses.ProjectionLoss(
    loss_fn=losses.RunningCCALoss(
        output_dimension=50, view1_dimension=128, view2_dimension=128
    ),
    net1=losses.DeepNet(input_features=100, layer_features=[64, 128], norm="layer"),
    net2=losses.DeepNet(input_features=120, layer_features=[64, 128], norm="layer"),
)

# %%
a = torch.rand((4, 100))
b = torch.rand((4, 120))

# %%
loss = model(a, b)["loss"]

# %%
loss

# %%
make_dot(
    loss,
    params=dict(list(model.named_parameters())),
    show_attrs=True,
    show_saved=True,
).render("dcca", format="pdf")

# %% [markdown]
# ---
# ## Creating true wikipedia dbow correlation matrix

# %%
from datasets import load_from_disk

# %%
ds = load_from_disk("/mnt/data/datasets/wikipedia_sample")

# %%
import numpy as np

# %%
dbow = ds["train"].with_format("np")["dbow"]
n, m = dbow.shape
print((n, m))

# %%
dbow_mean = dbow.mean(axis=0)

# %%
dbow_bar = dbow - dbow_mean

# %%
dbow_sigma = (1 / (n - 1)) * np.matmul(dbow_bar.T, dbow_bar)
dbow_sigma.shape

# %%
dbow_sigma

# %%
tmp = np.arange(40).reshape(5, 8)
tmp

# %%
tmp.diagonal(1)

# %%
import pandas as pd

pd.set_option("display.max_columns", 5000)
pd.set_option("display.max_rows", 5000)
pd.set_option("display.max_colwidth", 10)

# %%
pd.DataFrame(dbow_sigma)

# %%
np.save("wiki_sample_dbow_sigma.npy", dbow_sigma)
