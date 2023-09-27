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
import torch
import math
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoModelForSequenceClassification,
    LongformerForSequenceClassification,
    AutoTokenizer,
)
from transformer_document_embedding.tasks.imdb import IMDBClassification
from typing import Any, Optional
from torcheval.metrics import Metric, MulticlassAccuracy, Mean
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
import logging
from tqdm import tqdm_notebook
from transformers.data.data_collator import DataCollatorWithPadding
from torch.utils.data import DataLoader

# %%
logging.basicConfig(format="[%(levelname)s]%(name)s: %(message)s (%(module)s)")

# %%

del model  # noqa: F821
torch.cuda.empty_cache()

# %%
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
model = AutoModelForSequenceClassification.from_pretrained(
    "allenai/longformer-base-4096"
)

# %%
imdb_task = IMDBClassification(data_size_limit=100)


# %%
def tokenize(doc: dict[str, Any]) -> dict[str, Any]:
    return tokenizer(doc["text"], padding=False, truncation=True)


train = (
    imdb_task.train.shuffle()
    .map(tokenize, batched=True)
    .with_format("torch")
    .remove_columns(["text", "id"])
)

# %%
print(train[0])
print(train[0]["input_ids"].shape)

# %%

collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
dataloader = DataLoader(train, batch_size=1, shuffle=True, collate_fn=collator)

# %%
for batch in dataloader:
    print(batch)
    break
# %%
model.train()
with torch.no_grad():
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

# %%
print(outputs)

# %%
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(outputs["logits"], batch["labels"])

# %%
print(loss)
print(model.device)


# %%
def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> None:
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)


def fit(
    *,
    model: torch.nn.Module,
    epochs: int,
    train_dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, Metric],
    summary_writer: SummaryWriter,
    fp16: bool = False,
    max_grad_norm: Optional[float] = None,
    grad_accumulation_steps: int = 1,
    steps_in_epoch: Optional[int] = None,
    lr_scheduler=None,
    progress_bar: bool = True,
):
    device = model.device

    steps_in_epoch = (
        steps_in_epoch if steps_in_epoch is not None else len(train_dataloader)
    )

    scaler = torch.cuda.amp.GradScaler()

    for name, metric in metrics.items():
        metrics[name] = metric.to(device)
    loss_metric = Mean(device=device)

    for epoch in tqdm_notebook(range(epochs), desc="Epoch", disable=not progress_bar):
        for i, batch in tqdm_notebook(
            enumerate(train_dataloader),
            desc="Iteration",
            disable=not progress_bar,
            total=steps_in_epoch,
        ):
            batch_to_device(batch, device)

            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=fp16
            ):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss = (
                    loss_fn(outputs["logits"], batch["labels"])
                    / grad_accumulation_steps
                )

                loss_metric.update(loss * grad_accumulation_steps)

            for metric in metrics.values():
                metric.update(outputs["logits"], batch["labels"])

            loss = scaler.scale(loss) if fp16 else loss
            loss.backward()

            if (i + 1) % grad_accumulation_steps == 0 or (i + 1) == steps_in_epoch:
                if max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer_was_run = True
                if fp16:
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scale_after = scaler.get_scale()
                    optimizer_was_run = scale_before <= scale_after

                    scaler.update()
                else:
                    optimizer.step()

                if optimizer_was_run and lr_scheduler is not None:
                    lr_scheduler.step()

                # Or model.zero_grad()?
                optimizer.zero_grad()

        summary_writer.add_scalar("loss", loss_metric.compute(), epoch)
        loss_metric.reset()
        for name, metric in metrics.items():
            summary_writer.add_scalar(name, metric.compute(), epoch)
            metric.reset()


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
model.to(device)


# %%
def get_linear_lr_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> LambdaLR:
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step)
            / float(max(1, total_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


# %%

optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
epochs = 1
total_steps = epochs * len(dataloader)
warmup_steps = math.floor(0.1 * total_steps)
fit(
    model=model,
    epochs=epochs,
    train_dataloader=dataloader,
    loss_fn=torch.nn.CrossEntropyLoss(),
    optimizer=optimizer,
    metrics={
        "accuracy": MulticlassAccuracy(),
    },
    summary_writer=SummaryWriter(f"logs/{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"),
    fp16=True,
    max_grad_norm=1.0,
    grad_accumulation_steps=16,
    lr_scheduler=get_linear_lr_scheduler_with_warmup(
        optimizer, warmup_steps, total_steps
    ),
)

# %%
path = f"model_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
model.save_pretrained(path)

# %%
model.train()
with torch.no_grad():
    outputs = model(
        input_ids=batch["input_ids"].to(model.device),
        attention_mask=batch["attention_mask"].to(model.device),
    )

# %%
print(outputs)

# %%
model = LongformerForSequenceClassification.from_pretrained(path)

# %%
model.train()
with torch.no_grad():
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

# %%
print(outputs)
