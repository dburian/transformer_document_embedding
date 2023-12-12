from __future__ import annotations
from copy import copy
from typing import TYPE_CHECKING, Iterable
import logging

import torch
from torch.utils.data import DataLoader
from torcheval.metrics import (
    MulticlassAUPRC,
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
)
from tqdm.auto import tqdm


from transformer_document_embedding.models.transformer.base import TransformerBase
from transformer_document_embedding.models.trainer import (
    TorchTrainer,
)
from transformer_document_embedding.models.cls_head import ClsHead
from transformer_document_embedding.utils.metrics import TrainingMetric, VMemMetric
import transformer_document_embedding.utils.training as train_utils

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from datasets import Dataset
    from transformer_document_embedding.tasks.experimental_task import ExperimentalTask
    from typing import Any, Optional
    import numpy as np

logger = logging.getLogger(__name__)


class TransformerPairClassifier(TransformerBase):
    def __init__(
        self,
        transformer_model: str,
        cls_head_kwargs: dict[str, Any],
        label_smoothing: float,
        batch_size: int,
        pooler_type: str,
        transformer_model_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            transformer_model=transformer_model,
            transformer_model_kwargs=transformer_model_kwargs,
            batch_size=batch_size,
            pooler_type=pooler_type,
        )

        cls_head = ClsHead(
            **cls_head_kwargs,
            in_features=self._transformer.config.hidden_size * 2,
            out_features=2,
        )
        loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self._model = _SequenceClassificationModel(
            self._transformer, self._pooler, cls_head, loss
        )

    def train(
        self,
        task: ExperimentalTask,
        epochs: int,
        warmup_steps: int,
        grad_accumulation_steps: int,
        patience: Optional[int],
        weight_decay: float,
        fp16: bool,
        max_grad_norm: float,
        log_every_step: int,
        save_best: bool,
        validate_every_step: Optional[int],
        device: Optional[str] = None,
        log_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        **_,
    ) -> None:
        train_data = self._to_dataloader(task.train, training=True)
        val_data = None
        if task.validation is not None:
            val_data = self._to_dataloader(task.validation, training=False)

        optimizer = torch.optim.AdamW(
            train_utils.get_optimizer_params(self._model, weight_decay), lr=3e-5
        )

        lr_scheduler = train_utils.get_linear_lr_scheduler_with_warmup(
            optimizer,
            warmup_steps // grad_accumulation_steps,
            epochs * len(train_data) // grad_accumulation_steps,
        )

        save_model_callback = self._get_save_model_callback(save_best, model_dir)

        if self._transformer.supports_gradient_checkpointing:
            self._transformer.gradient_checkpointing_enable()

        train_logger, val_logger = self._get_train_val_loggers(
            log_dir, self._get_train_metrics(log_every_step)
        )

        trainer = TorchTrainer(
            model=self._model,
            optimizer=optimizer,
            train_logger=train_logger,
            val_logger=val_logger,
            fp16=fp16,
            max_grad_norm=max_grad_norm,
            grad_accumulation_steps=grad_accumulation_steps,
            lr_scheduler=lr_scheduler,
            validate_every_step=validate_every_step,
            save_model_callback=save_model_callback,
            patience=patience,
            device=device,
        )
        trainer.train(
            epochs=epochs,
            train_data=train_data,
            val_data=val_data,
        )

    def _get_train_metrics(self, default_log_freq: int) -> list[TrainingMetric]:
        def logits_accessor(metric, outputs, batch):
            metric.update(outputs["logits"], batch["labels"])

        return [
            TrainingMetric(
                "accuracy", MulticlassAccuracy(), default_log_freq, logits_accessor
            ),
            TrainingMetric(
                "recall", MulticlassRecall(), default_log_freq, logits_accessor
            ),
            TrainingMetric(
                "precision",
                MulticlassPrecision(),
                default_log_freq,
                logits_accessor,
            ),
            TrainingMetric(
                "auprc",
                MulticlassAUPRC(num_classes=2),
                default_log_freq,
                logits_accessor,
            ),
            VMemMetric(default_log_freq),
        ]

    @torch.inference_mode()
    def predict(self, inputs: Dataset) -> Iterable[np.ndarray]:
        self._model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Predicting using {device}")
        self._model.to(device)

        data = self._to_dataloader(inputs, training=False)
        for batch in tqdm(data, desc="Predicting batches"):
            train_utils.batch_to_device(batch, device)
            outputs = self._model(**batch)
            yield torch.argmax(outputs["logits"], dim=1).numpy(force=True)

    def save(self, dir_path: str) -> None:
        torch.save(self._model.state_dict(), dir_path)

    def load(self, dir_path: str) -> None:
        self._model.load_state_dict(torch.load(dir_path))

    def _to_dataloader(self, dataset: Dataset, training: bool) -> DataLoader:
        dataset = dataset.with_format("torch")
        dataset = dataset.remove_columns(["id"])

        collator = PairFastDataCollator(
            padding="longest",
            tokenizer=self._tokenizer,
            min_length=self._min_sequence_length,
        )

        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=training,
            collate_fn=collator,
        )


class PairFastDataCollator(train_utils.FastDataCollator):
    """Tokenizes pair of texts and puts their tokenization to two keys."""

    TOKENIZER_OUTPUT_KEYS = {
        "length",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "special_tokens_mask",
    }

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batches = [
            self._tokenize_single(features, text_key) for text_key in ["text1", "text2"]
        ]

        global_inputs = {}
        for batch in batches:
            # If value was not inserted by tokenizer, assume it is the same
            # for both inputs
            for key in set(batch.keys()) - self.TOKENIZER_OUTPUT_KEYS:
                global_inputs[key] = batch.pop(key)

        global_inputs["inputs1"] = batches[0]
        global_inputs["inputs2"] = batches[1]

        return global_inputs

    def _tokenize_single(
        self, features: list[dict[str, Any]], text_key: str
    ) -> dict[str, Any]:
        vanilla_feats = []
        for example in features:
            example = copy(example)
            example["text"] = example[text_key]
            del example[text_key]
            vanilla_feats.append(example)

        return super().__call__(vanilla_feats)


class _SequenceClassificationModel(torch.nn.Module):
    def __init__(
        self,
        transformer: PreTrainedModel,
        pooler: torch.nn.Module,
        classifier: torch.nn.Module,
        loss: torch.nn.Module,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.transformer = transformer
        self.pooler = pooler
        self.classifier = classifier
        self.loss = loss

    def forward(
        self,
        inputs1: dict[str, Any],
        inputs2: dict[str, Any],
        labels: torch.Tensor,
        **_,
    ) -> dict[str, torch.Tensor]:
        pair_pooled_outputs = [
            self._single_forward(**inputs) for inputs in [inputs1, inputs2]
        ]

        pooled_output = torch.cat(pair_pooled_outputs, dim=1)
        logits = self.classifier(pooled_output)
        loss = self.loss(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "pooled_output1": pair_pooled_outputs[0],
            "pooled_output2": pair_pooled_outputs[1],
        }

    def _single_forward(
        self,
        attention_mask: torch.Tensor,
        **input_kws,
    ) -> torch.Tensor:
        outputs = self.transformer(
            **input_kws,
            attention_mask=attention_mask,
            return_dict=True,
            inputs_embeds=None,
        )

        return self.pooler(
            **outputs,
            attention_mask=attention_mask,
        )
