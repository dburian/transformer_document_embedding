# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import torch
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
)
import transformer_document_embedding as tde
from typing import Any

# %%
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
model = AutoModelForSequenceClassification.from_pretrained(
    "allenai/longformer-base-4096"
)

# %%
imdb_task = tde.tasks.IMDBClassification(data_size_limit=1000)


# %%
def tokenize(doc: dict[str, Any]) -> dict[str, Any]:
    return tokenizer(doc["text"], padding="max_length", truncation=True)


train = imdb_task.train.map(tokenize, batched=True).shuffle().with_format("torch")

# %%
for name, param in model.named_parameters():
    print(name)
# %%

default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
}


training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    fp16=True,
    **default_args
)
trainer = Trainer(model=model, args=training_args, train_dataset=train)
result = trainer.train()

# %% [markdown]
# ----
# %%


def preprocess_batch(batch):
    pass


def pad_batch(batch):
    pass


def move_to_device(batch, device):
    pass


def get_lr_scheduler(optimizer):
    pass


def train(model, data_loader, epochs, optimizer):
    lr_scheduler = get_lr_scheduler(optimizer)

    for epoch in epochs:
        for i, batch in enumerate(data_loader):
            batch = preprocess_batch(batch)
            batch = pad_batch(batch)
            batch = move_to_device(batch, model.device)

            optimizer.zero_grad()
            outputs = model(**batch)

            outputs["loss"].backward()
            optimizer.step()
            lr_scheduler.step()
