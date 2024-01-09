from __future__ import annotations
from typing import TYPE_CHECKING, Iterable
import logging

import torch
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


class TransformerClassifier(TransformerBase):
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
            batch_size=batch_size,
            transformer_model_kwargs=transformer_model_kwargs,
            pooler_type=pooler_type,
        )
        cls_head = ClsHead(
            **cls_head_kwargs,
            in_features=self._transformer.config.hidden_size,
            out_features=2,
        )
        loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self._model: _SequenceClassificationModel = _SequenceClassificationModel(
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
        lr_scheduler_type: str,
        validate_every_step: Optional[int],
        freeze_transformer: bool,
        save_best: bool,
        device: Optional[str] = None,
        log_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        **_,
    ) -> None:
        # Freezing transformer if required or unfreezing if not
        self._model.transformer.requires_grad_(not freeze_transformer)

        save_model_callback = self._get_save_model_callback(save_best, model_dir)

        train_batches = self._to_dataloader(task.train)
        val_batches = None
        if task.validation is not None:
            val_batches = self._to_dataloader(task.validation, training=False)

        optimizer = torch.optim.AdamW(
            train_utils.get_optimizer_params(self._model, weight_decay), lr=3e-5
        )

        lr_scheduler = train_utils.get_lr_scheduler(
            scheduler_type=lr_scheduler_type,
            optimizer=optimizer,
            total_steps=epochs * len(train_batches) // grad_accumulation_steps,
            warmup_steps=warmup_steps // grad_accumulation_steps,
        )

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
            train_data=train_batches,
            val_data=val_batches,
        )

    def _get_train_metrics(self, default_log_frequency: int) -> list[TrainingMetric]:
        def logits_accessor(metric, outputs, batch):
            metric.update(outputs["logits"], batch["labels"])

        return [
            TrainingMetric(
                "accuracy", MulticlassAccuracy(), default_log_frequency, logits_accessor
            ),
            TrainingMetric(
                "recall", MulticlassRecall(), default_log_frequency, logits_accessor
            ),
            TrainingMetric(
                "precision",
                MulticlassPrecision(),
                default_log_frequency,
                logits_accessor,
            ),
            TrainingMetric(
                "auprc",
                MulticlassAUPRC(num_classes=2),
                default_log_frequency,
                logits_accessor,
            ),
            VMemMetric(default_log_frequency),
        ]

    @torch.inference_mode()
    def predict(self, inputs: Dataset) -> Iterable[np.ndarray]:
        self._model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Predicting using {device}")
        self._model.to(device)

        batches = self._to_dataloader(inputs, training=False)
        for batch in tqdm(batches, desc="Predicting batches"):
            train_utils.batch_to_device(batch, device)
            outputs = self._model(**batch)
            yield torch.argmax(outputs["logits"], dim=1).numpy(force=True)


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
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            inputs_embeds=None,
            **kwargs,
        )
        pooled_output = self.pooler(
            **outputs,
            attention_mask=attention_mask,
        )
        logits = self.classifier(pooled_output)
        loss = self.loss(logits, labels)

        return {
            **outputs,
            "loss": loss,
            "logits": logits,
        }
