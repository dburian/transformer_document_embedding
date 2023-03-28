# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
from datetime import datetime

import numpy as np

import transformer_document_embedding as tde

# %%

task = tde.tasks.IMDBClassification(data_size_limit=100)
model = tde.baselines.longformer.LongformerIMDB(epochs=1)
# %%
log_dir = os.path.join("logs", "longformer", datetime.now().strftime("%Y-%m-%d_%H%M%S"))
# %%
model.train(
    task,
    log_dir=log_dir,
)
# %%
before_save_predictions = list(model.predict(task.test))
# %%
model_path = os.path.join(log_dir, "model")
os.makedirs(model_path, exist_ok=True)
model.save(model_path)
# %%
model.load(model_path)
# %%
after_load_predictions = list(model.predict(task.test))
# %%
# %%
for before, after in zip(before_save_predictions, after_load_predictions):
    print(before)
    np.testing.assert_almost_equal(before, after)
