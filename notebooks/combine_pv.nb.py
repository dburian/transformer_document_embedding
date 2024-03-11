# %% [markdown]
# # Combine DMs and DBOWs into a single model

# %%
from __future__ import annotations
from transformer_document_embedding.scripts.config_specs import (
    EmbeddingModelSpec,
    ExperimentSpec,
)
import os
import re
import shutil

from transformer_document_embedding.scripts.evaluate import find_config
from transformer_document_embedding.scripts.utils import load_yaml, save_config


# %%
DMs = [""]
DBOWs = [""]


# %%
def load_config(model_save_path: str) -> ExperimentSpec:
    config_path = find_config(model_save_path)
    if config_path is None:
        raise ValueError(
            f"Cannot find config file for model saved at '{model_save_path}'."
        )

    return ExperimentSpec.from_dict(**load_yaml(config_path))


# %% [markdown]
# ## Product

# %%
from itertools import product

# %%
for dm_path, dbow_path in product(DMs, DBOWs):
    dm_config, dbow_config = load_config(dm_path), load_config(dbow_path)

    combined_config = ExperimentSpec(
        model=EmbeddingModelSpec(
            module="pv:ParagraphVectorConcat",
            kwargs={
                "dm": dm_config.model.kwargs,
                "dbow": dbow_config.model.kwargs,
            },
        ),
        head=None,
        dataset=dm_config.dataset,
        train_pipeline=None,
    )

    dm_name = os.path.basename(dm_path)
    dbow_name = os.path.basename(dbow_path)
    exp_path = os.path.join(
        "results",
        combined_config.dataset.module,
        combined_config.model.module,
        f"dm={dm_name}-dbow={dbow_name}",
    )
    save_config(combined_config, exp_path)

    for original_path, target_name in zip(
        [dm_path, dbow_path], ["dm", "dbow"], strict=True
    ):
        original_name = os.path.basename(original_path)
        original_dir = os.path.dirname(original_path)
        filenames_to_copy = [
            filename
            for filename in os.listdir(original_dir)
            if filename.startswith(original_name)
        ]

        for filename in filenames_to_copy:
            target_filename = re.sub(original_name, target_name, filename)
            shutil.copy(
                os.path.join(original_dir, filename),
                os.path.join(exp_path, target_filename),
            )
    print(f"Created model from dm={dm_name} and dbow={dbow_name} at '{exp_path}'")
# %%
