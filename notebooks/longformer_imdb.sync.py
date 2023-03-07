# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %%
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from typing import Any

import tensorflow as tf
import transformers

import transformer_document_embedding as tde

# %%

tokenizer = transformers.AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
model = transformers.TFAutoModelForSequenceClassification.from_pretrained(
    "allenai/longformer-base-4096"
)
# %%
imdb_task = tde.tasks.IMDBClassification()
# %%
def tokenize(doc: dict[str, Any]) -> dict[str, Any]:
    return tokenizer(doc["text"])


train = imdb_task.train.map(tokenize)
# %%
train = model.prepare_tf_dataset(
    train, batch_size=64, shuffle=True, tokenizer=tokenizer
)
# %%
dir(model)
# %%
model.compile(optimizer=tf.keras.optimizers.Adam())
# %%
model.fit(train)
