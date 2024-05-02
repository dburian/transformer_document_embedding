# %%

# %%
import torch
from pprint import pprint
from transformers import AutoModel, AutoTokenizer
import pickle

# %%
transformer_model = "google/bigbird-roberta-base"
transformer_model = "sentence-transformers/all-mpnet-base-v2"

transformer_model = "allenai/longformer-base-4096"
# transformer_model = "roberta-base"
seq_512 = ("a word " * 255)[:-1]
seq_4096 = ("a word " * 2047)[:-1]
batch_size = 1

# %%
torch.cuda.memory._record_memory_history(True)


# %%
def do_snapshot(name):
    with open(name, mode="wb") as outfile:
        snapshot = torch.cuda.memory._snapshot()
        pickle.dump(snapshot, outfile)


def mem_stats():
    pprint(torch.cuda.memory_stats())


def mem_sum():
    print(torch.cuda.memory_summary())


# %%
model = AutoModel.from_pretrained(transformer_model, add_pooling_layer=False)

# %%
param_count = sum(
    p.shape[0] if len(p.shape) == 1 else p.shape[0] * p.shape[1]
    for p in model.parameters()
    if p.requires_grad
)
print(f"model has: {param_count/1e6}M params")

# %%
param_count = sum(
    p.shape[0] if len(p.shape) == 1 else p.shape[0] * p.shape[1]
    for p_name, p in model.named_parameters()
    if p.requires_grad and "global" not in p_name
)
print(f"model has: {param_count/1e6}M non-global params")

# %%
340 * 4

# %%
model.config

# %%
for name, p in model.named_parameters():
    print(name, p.shape)

# %%
tokenizer = AutoTokenizer.from_pretrained(transformer_model)

# %%
mem_sum()

# %%
model.to(torch.device("cuda"))

# %%
mem_sum()

# %%
do_snapshot("before_training.pickle")

# %%
input = tokenizer(seq_512)

# %%
input = tokenizer(seq_4096)

# %%
len(input["input_ids"])

# %%
batch = {
    key: torch.tensor([value for _ in range(batch_size)], device="cuda")
    for key, value in input.items()
}

# %%
do_snapshot("after_batch.pickle")

# %%
mem_sum()

# %%
optimizer = torch.optim.AdamW(model.parameters())

# %%
do_snapshot("after_optimizer.pickle")

# %%
mem_sum()

# %%
with torch.autocast("cuda", dtype=torch.float16):
    outputs = model(**batch)
    loss = outputs["last_hidden_state"].sum().backward()

# %%
outputs = model(**batch)

# %%
do_snapshot("after_forward_pass.pickle")

# %%
mem_sum()

# %%
loss = outputs["last_hidden_state"].sum().backward()

# %%
mem_sum()

# %%
do_snapshot("after_backward_call.pickle")

# %%
optimizer.step()

# %%
mem_sum()

# %%
optimizer.zero_grad()

# %%
mem_sum()

# %%
mem_sum()
