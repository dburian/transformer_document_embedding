# %%
from transformer_document_embedding.tasks.teacher_embedding import TeacherEmbedding
from transformer_document_embedding.utils.tokenizers import create_tokenized_data_loader
from transformers import AutoTokenizer, AutoModel
import torch

# %%
task = TeacherEmbedding(
    path="/mnt/data/datasets/wikipedia_sample_with_eval",
    data_size_limit={
        "validation": 100,
        "train": 10,
    },
)

# %%
data = task.train

# %%
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")


def to_dataloader(data):
    return create_tokenized_data_loader(
        data,
        batch_size=3,
        tokenizer=tokenizer,
        training=False,
    )


# %%
batch = next(iter(to_dataloader(data)))

# %%
batch

# %%
tokenizer.all_special_ids

# %%
tokenizer.all_special_tokens

# %%
global_attention_type = "cls_and_end"

# %%
if global_attention_type == "first":
    attn = torch.zeros_like(batch["attention_mask"])
    attn[:, 0] = 1
elif global_attention_type == "cls":
    attn = (batch["input_ids"] == 0).to(torch.float32)
elif global_attention_type == "cls_and_end":
    attn = ((batch["input_ids"] == 0) + (batch["input_ids"] == 2)).to(torch.float32)

batch["global_attention_mask"] = attn
print(attn)

# %%
model = AutoModel.from_pretrained("allenai/longformer-base-4096")

# %%
with torch.no_grad():
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        global_attention_mask=batch["global_attention_mask"],
    )

# %%
outputs

# %%
outputs["last_hidden_state"].shape


# %%
def no_cls_pooler(outputs, batch):
    global_mask_neg = 1 - batch["global_attention_mask"]
    print(f"mask_neg: {global_mask_neg}")
    masked_last_hidden = outputs["last_hidden_state"] * global_mask_neg.unsqueeze(-1)
    print(f"masked last hidden: {masked_last_hidden}")
    return masked_last_hidden.sum(axis=1) / global_mask_neg.sum(axis=1).unsqueeze(-1)


# %%
res = no_cls_pooler(outputs, batch)
print(res)
res.shape
