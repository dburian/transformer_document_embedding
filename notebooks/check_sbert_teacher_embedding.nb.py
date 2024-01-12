# %%
from transformer_document_embedding.tasks.teacher_embedding import TeacherEmbedding
import yaml
from transformer_document_embedding.scripts.config_specs import ExperimentalModelSpec
from transformer_document_embedding.scripts.utils import init_type
from transformer_document_embedding.utils.tokenizers import create_tokenized_data_loader
from transformers import AutoTokenizer
import torch

# %%
task = TeacherEmbedding(path="/mnt/data/datasets/wikipedia_sample_with_eval")

# %%
model_config = ExperimentalModelSpec.from_dict(
    yaml.safe_load(
        """
  module: transformer.student:TransformerStudent
  kwargs:
    transformer_model: sentence-transformers/all-mpnet-base-v2
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
