# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Sentence transformers
# %%
import logging
import os
from datetime import datetime

import torch
from datasets.arrow_dataset import Dataset
from sentence_transformers import InputExample
from sentence_transformers.util import batch_to_device
from torch.utils.data import DataLoader
from torcheval import metrics

import transformer_document_embedding as tde

# %%
logging.basicConfig(level=logging.INFO)

# %%
model = tde.baselines.sbert.SBertIMDB(
    batch_size=8,
    epochs=10,
)
task = tde.tasks.IMDBClassification(data_size_limit=500, validation_train_fraction=0.01)
# %%

print({key: len(split) for key, split in task.splits.items()})
# %%
train_ids = set((doc["id"] for doc in task.train))
val_ids = set((doc["id"] for doc in task.validation))

print(val_ids.isdisjoint(train_ids))


# %%
class STDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset: Dataset) -> None:
        self._hf_dataset = hf_dataset

    def __len__(self) -> int:
        return len(self._hf_dataset)

    def __getitem__(self, idx: int) -> InputExample:
        doc = self._hf_dataset[idx]
        return InputExample(texts=[doc["text"]], label=doc["label"])


# %%
train_dataset = STDataset(task.train.with_format("torch"))
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=model._model.smart_batching_collate,
)
# %%
for batch in train_loader:
    print(batch)
    print()
    siam_features, labels = batch
    for idx in range(len(siam_features)):
        siam_features[idx] = batch_to_device(siam_features[idx], model._model.device)
    labels = labels.to(model._model.device)
    with torch.no_grad():
        outputs = [model._model(features) for features in siam_features]
    break

# %%
print(batch)
print()
print(outputs)
print()
print(labels)
# %%
metric = metrics.MulticlassAccuracy(num_classes=2, device=model._model.device)
metric.update(outputs[0]["sentence_embedding"], labels)
print(metric.compute())
metric.reset()

# %%
log_dir = os.path.join("logs", datetime.now().strftime("%Y-%m-%d_%H%M%S"))
model.train(task, log_dir=log_dir, save_best_path=os.path.join(log_dir, "model"))
