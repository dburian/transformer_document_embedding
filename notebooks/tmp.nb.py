# %%
from transformer_document_embedding.baselines.longformer.student import (
    LongformerStudent,
)
from transformer_document_embedding.tasks.teacher_embedding import TeacherEmbedding
import transformer_document_embedding.utils.torch.training as train_utils
from transformers import AutoTokenizer

# %%
model = LongformerStudent(
    large=False,
    batch_size=2,
    pooler_type="mean",
    static_embed_dim=100,
    contextual_max_length=384,
    static_contextual_lambda=0.5,
    static_loss_type=True,
    cca_output_dim=50,
    projection_norm="layer",
    longformer_projection_layers=[256],
    static_projection_layers=[64],
)

# %%
print(
    "\n".join(
        [f"{name}: {param.shape}" for name, param in model._model.named_parameters()]
    )
)

# %%
print(model._model.get_parameter("loss._static_loss._net2.layers.2.weight").grad)

# %%
task = TeacherEmbedding("/mnt/data/datasets/wikipedia_sample")

# %%
train_data = train_utils.create_tokenized_data_loader(
    task.train,
    tokenizer=AutoTokenizer.from_pretrained("allenai/longformer-base-4096"),
    batch_size=2,
    sampling="consistent",
    training=True,
    return_length=False,
    sampler_kwargs={
        "effective_batch_size": 2 * 10,
        "bucket_limits": [384],
        # "short_count": short_inputs_in_effective_batch,
        # "short_threshold": self._contextual_max_length,
    },
)


# %%
for batch in train_data:  # noqa: B007
    break

# %%
model._model.train()

# %%
loss = model._model(**batch)
loss

# %%
loss["loss"].backward()

# %%
loss.keys()

# %%
loss["static_loss"]

# %%
loss["contextual_loss"]

# %%
loss["loss"]

# %%
loss["static_loss"] + loss["contextual_loss"] * 0.5

# %%
param = "loss._static_loss._net2.layers.2.weight"
model._model.get_parameter(param).grad

# %% [markdown]
# ----
# ### Testing just DCCA

# %%
import transformer_document_embedding.utils.torch.losses as losses
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
# ## Testing detach

# %%
a = torch.tensor([1, 2, 3, 4], requires_grad=True, dtype=torch.float)
b = torch.tensor([1, 2, 3, 4], requires_grad=True, dtype=torch.float)

# %%
a = a.detach()
a = a + b

# %%
a

# %%
loss = torch.sum(a)

# %%
loss.backward()

# %%
make_dot(loss, params={"a": a, "b": b}).render("detach", format="pdf")

# %%
param = "transformer.longformer.encoder.layer.11.output.LayerNorm.weight"
print(model._model.get_parameter(param).grad)

# %%
from torchviz import make_dot

make_dot(loss["loss"], params=dict(list(model._model.named_parameters()))).render(
    "rnn_torchviz", format="pdf"
)

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
