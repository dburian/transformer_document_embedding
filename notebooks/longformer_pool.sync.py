import torch
from transformers import LongformerConfig, LongformerModel, LongformerTokenizer

from transformer_document_embedding.models.longformer import \
    LongformerConfig as TDELongformerConfig

# %%
config = LongformerConfig.from_pretrained("allenai/longformer-base-4096")

tde_config = TDELongformerConfig(
    classifier_activation="relu",
    pooler_type="mean",
    classifier_dropout_prob=None,
    **TDELongformerConfig.get_config_dict("allenai/longformer-base-4096")[0],
)
# %%
print(tde_config.classifier_activation)
print(tde_config.pooler_type)
print(tde_config.classifier_dropout_prob)
print(tde_config.classifier_hidden_size)
# %%
model = LongformerModel.from_pretrained("allenai/longformer-base-4096")

tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
# %%

sentences = [
    "Hi my name is Alaster Blister. Please welcome to my villa.",
    "Very short sentence",
]
inputs = tokenizer(sentences, padding=True, truncation=True)
# %%
for key, value in inputs.items():
    inputs[key] = torch.tensor(value)
# %%
inputs
# %%
inputs["input_ids"].shape
# %%
with torch.no_grad():
    outputs = model(**inputs)
# %%
outputs
# %%
print(outputs.last_hidden_state.shape)
print(outputs[0].shape)
print(outputs.pooler_output.shape)
# %%
print(outputs.last_hidden_state[0, 0])
print(outputs.pooler_output[0])
# %%


def pool_sum(
    *, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    return torch.sum(last_hidden_states * attention_mask[:, :, None], 1)


def pool_average(
    *, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    summed = torch.sum(last_hidden_states * attention_mask[:, :, None], 1)
    row_lengths = torch.sum(attention_mask, 1)
    return summed / row_lengths[:, None]


# %%

a_states = torch.tensor(
    [
        [[1, 2, 3, 4, 5], [1, 1, -1, -1, 1], [0, -9999, 99999, 99999, 0]],
        [[1, 2, 3, 4, 5], [0, -9999, 99999, 99999, 0], [0, -9999, 99999, 99999, 0]],
        [[1, 2, 3, 4, 5], [0, -9999, 99999, 99999, 0], [0, -9999, 99999, 99999, 0]],
    ]
)
a_mask = torch.tensor(
    [
        [1, 1, 0],
        [1, 0, 0],
        [1, 0, 0],
    ]
)
# %%
print(pool_sum(last_hidden_states=a_states, attention_mask=a_mask))
# %%
print(pool_average(last_hidden_states=a_states, attention_mask=a_mask))
# %% [markdown]
# ## Testing my code
# %%

from typing import cast

import transformers.models.longformer as hf_longformer
from transformers import AutoTokenizer

from transformer_document_embedding.models.longformer import (
    LongformerConfig, LongformerForSequenceClassification)

# %%

model_path = "allenai/longformer-base-4096"
config = cast(LongformerConfig, LongformerConfig.from_pretrained(model_path))
config.classifier_activation = "relu"
config.pooler_type = "sum"
config.num_labels = 2

model = cast(
    LongformerForSequenceClassification,
    LongformerForSequenceClassification.from_pretrained(
        model_path,
        config=config,
    ),
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
# %%
hf_model = cast(
    hf_longformer.LongformerForSequenceClassification,
    hf_longformer.LongformerForSequenceClassification.from_pretrained(
        model_path,
        config=config,
    ),
)
# %%
model.state_dict()
# %%
hf_model.state_dict()
# %%

sentences = [
    "Hi my name is Alaster Blister. Please welcome to my villa.",
    "Very short sentence",
]
inputs = tokenizer(sentences, padding=True, truncation=True)
# %%
for key, value in inputs.items():
    inputs[key] = torch.tensor(value)
# %%
print(inputs["input_ids"].shape)
print(inputs["attention_mask"].shape)
# %%
with torch.no_grad():
    outputs = model(**inputs)
# %%
outputs
# %%
print(outputs.last_hidden_state.shape)
print(outputs.pooler_output.shape)
# %% [markdown]
# ## Testing whole model
# %%

from transformer_document_embedding.baselines.longformer import LongformerIMDB
from transformer_document_embedding.tasks import IMDBClassification

# %%
task = IMDBClassification(
    data_size_limit=500,
    validation_source="test",
    validation_source_fraction=0.2,
)

model = LongformerIMDB(epochs=1, pooler_type="mean", classifier_activation="relu")
# %%
log_dir = "./longformer_logs"
model.train(task, log_dir=log_dir)
# %%
