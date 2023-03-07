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
# %% [markdown]
# # Sentence transformers
# %%
import logging
from datetime import datetime

import transformer_document_embedding as tde

# %%
logging.basicConfig(level=logging.INFO)

# %%
model = tde.models.SBertIMDB(
    log_dir=f"logs/{datetime.now().strftime('%Y-%m-%d_%H%M%S')}",
    batch_size=8,
    epochs=10,
)
task = tde.tasks.IMDBClassification(data_size_limit=500)
# %%
model.train(train=task.train)
# %%
# model.save("./model")
# %%
# model.load("./model")
# %%
# model.train(train=task.train)
