# %%
from transformer_document_embedding.tasks.teacher_embedding import TeacherEmbedding
import yaml
from transformer_document_embedding.scripts.config_specs import ExperimentalModelSpec
from transformer_document_embedding.scripts.utils import init_type
from transformer_document_embedding.utils.tokenizers import create_tokenized_data_loader
from transformers import AutoTokenizer
import torch

# %%
task = TeacherEmbedding(path="/mnt/data/datasets/wikipedia_sample_with_eval/")

# %%
model_config = ExperimentalModelSpec.from_dict(
    yaml.safe_load(
        """
  module: transformer.student:TransformerStudent
  kwargs:
    transformer_model: sentence-transformers/all-mpnet-base-v2
    #transformer_model: distilroberta-base
    pooler_type: mean
    batch_size: 4
    contextual_max_length: null
    contextual_loss_kwargs: null
    static_loss_kwargs:
      static_key: dbow
      static_embed_dim: 100
      static_loss_type: soft_cca
      projection_norm: null
      cca_output_dim: null
      transformer_projection_layers: [768]
      static_projection_layers: [512, 768]
      soft_cca_lam: 0.15
      soft_cca_sdl_alpha: 0.999
      contrastive_lam: 0.8

  train_kwargs:
    save_best: False
    epochs: 1
    warmup_steps: 840
    grad_accumulation_steps: 4
    weight_decay: 0.01
    lr: 0 #3.0e-5
    lr_scheduler_type: cos
    fp16: True
    max_grad_norm: 1.0
    log_every_step: 4
    validate_every_step: 1050
    global_attention_type: cls
    patience: null
    device: cuda
    dataloader_sampling: consistent
    bucket_limits: [1024, 2048, 3072, 4096]
"""
    )
)

# %%
model = init_type(model_config)

# %%
for doc in task.train:
    print(doc)
    break

# %%
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

# %%
tokenized_train = create_tokenized_data_loader(
    task.train,
    batch_size=3,
    tokenizer=tokenizer,
)

# %%
with torch.inference_mode():
    for i, batch in enumerate(tokenized_train):
        outputs = model._model(**batch)
        # print(batch['sbert'])
        # print(outputs['embeddings'])
        print(batch["sbert"] - outputs["embeddings"])
        print()

        if i > 5:
            break

# %% [markdown]
# ----

# %%
from datasets import Dataset, DatasetDict, concatenate_datasets
import datasets.utils.logging as hf_logging
import os
import logging

from transformer_document_embedding.utils.evaluation import smart_unbatch

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


# %%
def gen_embeds(model, suff, st=False):
    def embed_generator(split: Dataset):
        pred_iter = (
            model.encode(split["text"])
            if st
            else smart_unbatch(model.predict(split), 1)
        )
        for batch, embed in zip(split, pred_iter):
            yield {
                f"sbert{suff}": embed,
                f"text{suff}": batch["text"],
                f"id{suff}": batch["id"],
            }

    embeddings = DatasetDict()

    # TODO: We're relying on HF Task interface. Do we really need the basic
    # task? Should the basic task also have `splits` property?
    splits = task.split_names.keys()

    hf_logging.disable_progress_bar()
    for split_name in splits:
        logging.info("Generating embeddings for split '%s'", split_name)
        split = task.split_names.get(split_name, None).select(range(100))
        if split is None:
            logging.warn(
                "Split '%s' doesn't exist for this dataset. Skipping...", split_name
            )
            continue

        split_embeddings = Dataset.from_generator(
            embed_generator, gen_kwargs={"split": split}
        )
        embeddings[split_name] = concatenate_datasets([split, split_embeddings], axis=1)
    hf_logging.enable_progress_bar()

    return embeddings


# %%
embeddings = gen_embeds(model, "_new")

# %%
from sentence_transformers import SentenceTransformer

# %%
st_model = SentenceTransformer("all-mpnet-base-v2")

# %%
embeddings_st = gen_embeds(st_model, "_st", st=True)

# %%
embeddings_st_new = gen_embeds(st_model, "_st_new", st=True)


# %%
class MeanPooler(torch.nn.Module):
    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor, **_
    ) -> torch.Tensor:
        # summed = torch.sum(last_hidden_state * attention_mask[:, :, None], 1)
        # row_lengths = torch.sum(attention_mask, 1)
        # return summed / row_lengths[:, None]

        token_embeddings = last_hidden_state  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


# %%
model._model.pooler = MeanPooler()

# %%
embeddings_pooler = gen_embeds(model, "_pooler")

# %%
from pprint import pprint
import numpy as np

# %%

base_columns = ["id", "url", "title", "text", "dbow", "sbert", "length"]
suffixes = ["", "_new", "_pooler", "_st"]
embeds = [embeddings["train"]] + [
    ds["train"].remove_columns(base_columns)
    for ds in [embeddings_st, embeddings_pooler]
]

for i, doc in enumerate(concatenate_datasets(embeds, axis=1)):
    pprint({f"{key}{suff}": doc[f"{key}{suff}"] for key in ["id"] for suff in suffixes})
    for suff in suffixes[1:]:
        cos = (np.array(doc["sbert"]) @ np.array(doc[f"sbert{suff}"]).T) / (
            np.linalg.norm(doc["sbert"]) * np.linalg.norm(doc[f"sbert{suff}"])
        )
        print(f"{suff}:")
        print(
            f"  mse: {np.power(np.array(doc['sbert']) - np.array(doc[f'sbert{suff}']), 2).mean()}"
        )
        print(f"  cos_dist: {1 - cos}")
    print()
    if i > 5:
        break

# %%
# del model
# del st_model

# %%

# %%

embed_dataset_path = os.path.join(exp_path, "embeddings")
logging.info("Saving the embeddings dataset to '%s'", embed_dataset_path)
embeddings.save_to_disk(embed_dataset_path, max_shard_size=args.max_shard_size)
