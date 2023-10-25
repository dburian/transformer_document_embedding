from __future__ import annotations
import os
import torch
import logging

from typing import cast, TYPE_CHECKING
from torch.utils.tensorboard.writer import SummaryWriter
from transformers import AutoTokenizer
from transformer_document_embedding.baselines.baseline import Baseline
import transformer_document_embedding.utils.torch.losses as losses
import transformer_document_embedding.utils.torch.training as train_utils
import transformer_document_embedding.baselines.longformer.train as longformer_training
from transformer_document_embedding.models.longformer import (
    LongformerConfig,
    LongformerForTextEmbedding,
)
from transformer_document_embedding.utils.metrics import VMemMetric
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from transformer_document_embedding.tasks.experimental_task import ExperimentalTask
    import numpy as np
    from datasets import Dataset
    from typing import Optional, Iterable

logger = logging.getLogger(__name__)


class LongformerStudent(Baseline):
    def __init__(
        self,
        large: bool = False,
        batch_size: int = 2,
        pooler_type: str = "mean",
        weight_decay: float = 0.0,
        warmup_steps: int = 10000,
        epochs: int = 100,
    ) -> None:
        self._batch_size = batch_size
        self._warmup_steps = warmup_steps
        self._weight_decay = weight_decay
        self._epochs = epochs

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
        log_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        early_stopping: bool = False,
        save_best: bool = False,
        **_,
    ) -> None:
        train_data = train_utils.create_tokenized_data_loader(
            task.train,
            tokenizer=self._tokenizer,
            batch_size=self._batch_size,
            return_length=True,
        )

        val_data = None
        val_summary_writer = None
        summary_writer = None

        if task.validation is not None:
            val_data = train_utils.create_tokenized_data_loader(
                task.validation,
                tokenizer=self._tokenizer,
                batch_size=self._batch_size,
                return_length=True,
            )

            if log_dir is not None:
                val_summary_writer = SummaryWriter(os.path.join(log_dir, "val"))

        if log_dir is not None:
            summary_writer = SummaryWriter(os.path.join(log_dir, "train"))

        optimizer = torch.optim.AdamW(
            train_utils.get_optimizer_params(self._model, self._weight_decay), lr=3e-5
        )

        self._model.gradient_checkpointing_enable()
        longformer_training.train(
            model=self._model,
            train_data=train_data,
            val_data=val_data,
            epochs=self._epochs,
            loss_fn=self._construct_loss(),
            optimizer=optimizer,
            metrics={
                "used_vmem": VMemMetric(),
            },
            summary_writer=summary_writer,
            val_summary_writer=val_summary_writer,
            fp16=True,
            max_grad_norm=1.0,
            grad_accumulation_steps=max(1, int(32 / self._batch_size)),
            lr_scheduler=train_utils.get_linear_lr_scheduler_with_warmup(
                optimizer,
                self._warmup_steps,
                self._epochs * len(train_data),
            ),
            patience=3 if early_stopping else None,
            checkpoint_path=model_dir,
            targets_getter=lambda batch: batch,
            outputs_getter=lambda outputs: outputs["pooler_output"],
        )

        if model_dir is not None and save_best:
            if val_data is None:
                # We have not saved anything
                self.save(model_dir)
            else:
                # We have saved at least something, lets restore it
                self.load(model_dir)

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

    def _construct_loss(self) -> torch.nn.Module:
        model_output_features = self._model.config.hidden_size
        static_loss = losses.DCCALoss(
            net1=losses.DeepNet(
                layer_features=[128, 256],
                input_features=model_output_features,
            ),
            net2=losses.DeepNet(
                [128, 256],
                input_features=100,
            ),
            cca_loss=losses.CCALoss(output_dimension=128),
        )

        return losses.AlwaysStaticShortContextual(
            contextual_key="sbert",
            static_key="dbow",
            static_loss=static_loss,
        ).to(torch.device("cuda"))

    def save(self, dir_path: str) -> None:
        self._model.save_pretrained(dir_path)

    def load(self, dir_path: str) -> None:
        self._model = cast(
            LongformerForTextEmbedding,
            LongformerForTextEmbedding.from_pretrained(dir_path),
        )
