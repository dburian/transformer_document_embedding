from __future__ import annotations
from typing import TYPE_CHECKING
from typing import cast

import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import logging

import transformer_document_embedding.utils.torch.training as train_utils
from transformer_document_embedding.baselines.experimental_model import (
    Baseline,
)
from transformer_document_embedding.models.longformer import (
    LongformerConfig,
    LongformerForTextEmbedding,
)

if TYPE_CHECKING:
    from transformer_document_embedding.tasks.experimental_task import ExperimentalTask
    from datasets import Dataset
    from typing import Iterable, Optional
    import numpy as np

logger = logging.getLogger(__name__)


class LongformerWikipediaSimilarities(Baseline):
    def __init__(
        self, *, large: bool = False, batch_size: int = 1, pooler_type: str = "mean"
    ) -> None:
        self._batch_size = batch_size

        model_path = f"allenai/longformer-{'large' if large else 'base'}-4096"
        config = LongformerConfig(
            pooler_type=pooler_type,
            **LongformerConfig.get_config_dict(model_path)[0],
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = cast(
            LongformerForTextEmbedding,
            LongformerForTextEmbedding.from_pretrained(model_path, config=config),
        )

    def train(
        self,
        task: ExperimentalTask,
        *,
        log_dir: Optional[str] = None,
        early_stopping: bool = False,
        save_best_path: Optional[str] = None,
    ) -> None:
        pass

    @torch.no_grad()
    def predict(self, inputs: Dataset) -> Iterable[np.ndarray]:
        self._model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Predicting using {device}")
        self._model.to(device)

        data = train_utils.create_tokenized_data_loader(
            inputs,
            tokenizer=self._tokenizer,
            batch_size=self._batch_size,
            training=False,
        )
        for batch in tqdm(data, desc="Predicting batches"):
            train_utils.batch_to_device(batch, self._model.device)
            embeddings = self._model(**batch).pooler_output
            yield embeddings.numpy(force=True)

    def save(self, dir_path: str) -> None:
        self._model.save_pretrained(dir_path)

    def load(self, dir_path: str) -> None:
        self._model = cast(
            LongformerForTextEmbedding,
            LongformerForTextEmbedding.from_pretrained(dir_path),
        )
