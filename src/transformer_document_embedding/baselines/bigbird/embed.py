import logging
from typing import Iterable, Optional, cast

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from transformer_document_embedding.baselines.experimental_model import (
    ExperimentalModel,
)
from transformer_document_embedding.models.bigbird import (
    BigBirdConfig,
    BigBirdForTextEmbedding,
)
from transformer_document_embedding.tasks.experimental_task import ExperimentalTask
from transformer_document_embedding.utils.torch import training as train_utils

logger = logging.getLogger(__name__)


class BigBirdEmbedder(ExperimentalModel):
    def __init__(
        self,
        *,
        batch_size: int = 1,
        block_size: int = 64,
        num_random_blocks: int = 3,
        attention_type: str = "block_sparse",
        pooler_type: str = "mean",
    ) -> None:
        self._batch_size = batch_size
        model_path = "google/bigbird-roberta-base"
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

        config = BigBirdConfig.from_pretrained(
            model_path,
            pooler_type=pooler_type,
            attention_type=attention_type,
            num_random_blocks=num_random_blocks,
            block_size=block_size,
        )
        self._model = cast(
            BigBirdForTextEmbedding,
            BigBirdForTextEmbedding.from_pretrained(
                model_path,
                config=config,
            ),
        )

        # in order to use block_sparse attention, sequence_length has to be at least
        # bigger than:
        # 2 * block_size (global attentions) +
        # 3 * block_size (sliding tokens) +
        # 2 * num_random_blocks * block_size (random tokens)
        self._min_sequence_length_for_block_sparse_attn = (
            1 + (5 + 2 * num_random_blocks) * block_size
        )

    def train(
        self,
        task: ExperimentalTask,
        *,
        log_dir: Optional[str] = None,
        save_best_path: Optional[str] = None,
        early_stopping: bool = False,
    ) -> None:
        pass

    @torch.no_grad()
    def predict(self, inputs: Dataset) -> Iterable[np.ndarray]:
        self._model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)

        mem_used = torch.cuda.memory_reserved(device)
        mem_used = mem_used // 1024**2
        logger.info("Memory exhausted by model: %d MB", mem_used)

        data = train_utils.create_tokenized_data_loader(
            inputs,
            tokenizer=self._tokenizer,
            batch_size=self._batch_size,
            training=False,
            min_sequence_length=self._min_sequence_length_for_block_sparse_attn,
        )
        for batch in data:
            train_utils.batch_to_device(batch, self._model.device)
            embeddings = self._model(**batch).pooler_output
            yield embeddings.numpy(force=True)

    def save(self, dir_path: str) -> None:
        self._model.save_pretrained(dir_path)

    def load(self, dir_path: str) -> None:
        self._model = cast(
            BigBirdForTextEmbedding,
            BigBirdForTextEmbedding.from_pretrained(dir_path),
        )
