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

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from datetime import datetime

import transformer_document_embedding as tde

# %%

task = tde.tasks.IMDBClassification()
model = tde.baselines.longformer.LongformerIMBD(epochs=1)
# %%
log_dir = os.path.join("logs", "longformer", datetime.now().strftime("%Y-%m-%d_%H%M%S"))
# %%
model.train(
    task,
    log_dir=log_dir,
)
# %%
before_save_predictions = model.predict(task.test)
# %%
model_path = os.path.join(log_dir, "model")
os.makedirs(model_path, exist_ok=True)
model.save(model_path)
# %%
model.load(model_path)
# %%
after_load_predictions = model.predict(task.test)
# %%
for before, after in zip(before_save_predictions, after_load_predictions):
    np.testing.assert_almost_equal(before, after)
