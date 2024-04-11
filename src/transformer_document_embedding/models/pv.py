from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterator

import os
import numpy as np
from gensim.models import Doc2Vec
import torch
from transformer_document_embedding.models.embedding_model import EmbeddingModel
from transformer_document_embedding.utils.gensim import (
    create_text_pre_processor,
)
from tqdm.auto import tqdm

import logging

if TYPE_CHECKING:
    from datasets import Dataset
    from typing import Optional

logger = logging.getLogger(__name__)


class ParagraphVector(Doc2Vec, EmbeddingModel):
    def __init__(
        self,
        text_pre_process: Optional[str],
        load_dv: bool,
        documents=None,
        corpus_file=None,
        vector_size=100,
        dm_mean=None,
        dm=1,
        dbow_words=0,
        dm_concat=0,
        dm_tag_count=1,
        dv=None,
        dv_mapfile=None,
        comment=None,
        trim_rule=None,
        callbacks=...,
        window=5,
        epochs=10,
        shrink_windows=True,
        **kwargs,
    ):
        super().__init__(
            documents,
            corpus_file,
            vector_size,
            dm_mean,
            dm,
            dbow_words,
            dm_concat,
            dm_tag_count,
            dv,
            dv_mapfile,
            comment,
            trim_rule,
            callbacks,
            window,
            epochs,
            shrink_windows,
            **kwargs,
        )

        # We save the preferred text pre-processing with model, but pipelines
        # have the need to apply it
        self.text_pre_process = text_pre_process

        # Whether to load learned paragraph vectors when loading weights
        self.load_dv = load_dv

    @property
    def embedding_dim(self) -> int:
        return self.vector_size

    @torch.inference_mode()
    def predict_embeddings(
        self, dataset: Dataset, batch_size: int
    ) -> Iterator[torch.Tensor]:
        pre_processor = create_text_pre_processor(self.text_pre_process)

        def add_words(doc: dict[str, Any]) -> dict[str, Any]:
            return {"words": pre_processor(doc["text"])}

        corpus = dataset.map(add_words, num_proc=self.workers, remove_columns=["text"])

        def to_torch(batch: list[np.ndarray]) -> torch.Tensor:
            return torch.from_numpy(np.array(batch))

        batch = []
        for doc in tqdm(corpus, desc="Predicting docs"):
            batch.append(self.infer_vector(doc["words"]))

            if len(batch) == batch_size:
                yield to_torch(batch)
                batch = []

        if len(batch) > 0:
            yield to_torch(batch)

    def save_weights(self, path: str) -> None:
        return self.save(path)

    def load_weights(self, path: str, *, strict: bool = False) -> None:
        new_self = Doc2Vec.load(path)

        for name, attr in vars(new_self).items():
            if name == "dv" and not self.load_dv:
                continue
            setattr(self, name, attr)


class ParagraphVectorConcat(EmbeddingModel):
    """A pseudo-model that allows to combine different learned Paragraph Vectors.

    This model cannot be used in training pipelines, only in finetuning
    pipelines. Though it could be extended if necessary. But really you should
    train the PVs separately since it is much cleaner way to do things.
    """

    def __init__(self, **modules_kwargs: dict[str, Any]) -> None:
        super().__init__()

        self.modules = {
            name: ParagraphVector(**kwargs) for name, kwargs in modules_kwargs.items()
        }

    @property
    def embedding_dim(self) -> int:
        return sum(module.embedding_dim for module in self.modules.values())

    @torch.inference_mode()
    def predict_embeddings(
        self, dataset: Dataset, batch_size: int
    ) -> Iterator[torch.Tensor]:
        for embeds in zip(
            *(
                module.predict_embeddings(dataset, batch_size=batch_size)
                for module in self.modules.values()
            ),
            strict=True,
        ):
            yield torch.concat(embeds, dim=1)

    def save_weights(self, path: str) -> None:
        for name, module in self.modules.items():
            module.save_weights(os.path.join(path, name))

    def load_weights(self, path: str, *, strict: bool = False) -> None:
        for mod_name, mod in self.modules.items():
            mod_path = os.path.join(path, mod_name)

            mod.load_weights(mod_path, strict=strict)
