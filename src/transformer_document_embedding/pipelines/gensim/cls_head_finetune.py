from __future__ import annotations
from typing import TYPE_CHECKING

from torch.utils.data import DataLoader
from transformer_document_embedding.datasets import col
from transformer_document_embedding.pipelines.classification_finetune import (
    BinaryClassificationFinetune,
)
from transformer_document_embedding.utils.gensim import (
    create_text_pre_processor,
)
from typing import Any
import numpy as np

if TYPE_CHECKING:
    import torch
    from transformer_document_embedding.models.pv import ParagraphVector
    from datasets import Dataset


class PVClassificationHeadTrain(BinaryClassificationFinetune):
    """Finetuning pipeline adjusted for PV.

    The major difference is when training PV's head on the same dataset as we
    train PV, we can afford to 'predict' embeddings just by indexing into the
    document embedding matrix. This is not possible during finetuning as we are
    not sure PV was trained on the same corpus.
    """

    def to_dataloader(
        self,
        split: Dataset,
        model: ParagraphVector,
        training: bool = True,
    ) -> DataLoader[torch.Tensor]:
        def get_train_vector(doc: dict[str, Any]) -> np.ndarray:
            return model.dv[doc[col.ID]]

        pre_processor = create_text_pre_processor(model.text_pre_process)

        def get_eval_vector(doc: dict[str, Any]) -> np.ndarray:
            return model.infer_vector(pre_processor(doc[col.TEXT]))

        get_embed = get_train_vector if training else get_eval_vector
        with_embeds = split.map(lambda doc: {col.EMBEDDING: get_embed(doc)})
        with_embeds.set_format("torch")

        return DataLoader(
            with_embeds,
            batch_size=self.batch_size,
            shuffle=True,
        )


class PVPairClassificationHeadTrain(BinaryClassificationFinetune):
    """Finetuning pipeline adjusted for PV"""

    def to_dataloader(
        self,
        split: Dataset,
        model: ParagraphVector,
        training: bool = True,
    ) -> DataLoader[torch.Tensor]:
        def get_train_vector(id: int, _: str) -> np.ndarray:
            return model.dv[id]

        pre_processor = create_text_pre_processor(model.text_pre_process)

        def get_eval_vector(_: int, text: str) -> np.ndarray:
            return model.infer_vector(pre_processor(text))

        get_single_embed = get_train_vector if training else get_eval_vector

        def get_embed(doc: dict[str, Any]) -> np.ndarray:
            return np.concatenate(
                (
                    get_single_embed(doc[col.ID_0], doc[col.TEXT_0]),
                    get_single_embed(doc[col.ID_1], doc[col.TEXT_1]),
                ),
                0,
            )

        with_embeds = split.map(lambda doc: {col.EMBEDDING: get_embed(doc)})
        with_embeds.set_format("torch")

        return DataLoader(
            with_embeds,
            batch_size=self.batch_size,
            shuffle=True,
        )
