from __future__ import annotations

import os
from typing import Any, Iterable, Optional, cast

import numpy as np
import torch
from datasets.arrow_dataset import Dataset
from sentence_transformers import (InputExample, SentenceTransformer,
                                   evaluation, models)
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torcheval.metrics.classification import MulticlassAccuracy

import transformer_document_embedding.utils.sentence_transformers as tde_st_utils
import transformer_document_embedding.utils.torch as torch_utils
from transformer_document_embedding.baselines.experimental_model import \
    ExperimentalModel
from transformer_document_embedding.tasks.imdb import IMDBClassification
from transformer_document_embedding.utils.metrics import (MeanLossMetric,
                                                          VMemMetric)


class SBertIMDB(ExperimentalModel):
    # When I need this again, lets put it in some utils module
    class STDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset: Dataset) -> None:
            self._hf_dataset = hf_dataset

        def __len__(self) -> int:
            return len(self._hf_dataset)

        def __getitem__(self, idx: int) -> InputExample:
            doc = self._hf_dataset[idx]
            return InputExample(texts=[doc["text"]], label=doc["label"])

    def __init__(
        self,
        *,
        transformer_model: str = "all-distilroberta-v1",
        batch_size: int = 6,
        epochs: int = 10,
        warmup_steps: int = 10000,
        label_smoothing: float = 0.15,
        cls_head_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._batch_size = batch_size
        self._epochs = epochs
        self._warmup_steps = warmup_steps

        modules = []

        transformer = SentenceTransformer(transformer_model)
        embed_dim = cast(int, transformer.get_sentence_embedding_dimension())
        modules.append(transformer)

        if cls_head_kwargs is None:
            cls_head_kwargs = {}

        cls_head_config = {
            "hidden_features": 25,
            "hidden_dropout": 0.1,
            "hidden_activation": "relu",
        }
        cls_head_config.update(cls_head_kwargs)

        # TODO: Move sentence transformer cls head elsewhere

        # We cannot use torch.nn.Module because of complications with
        # saving/loading. It definitely could be done -- create a wrapper
        # capable of wrapping every torch.nn.Module, but it would have to be
        # capable of reinitializing the wrapped module inside its static class
        # -- but is is easier this way.
        if cls_head_config["hidden_features"] > 0:
            modules.append(
                models.Dense(
                    in_features=embed_dim,
                    out_features=cls_head_config["hidden_features"],
                    activation_function=torch_utils.get_activation(
                        cls_head_config["hidden_activation"]
                    )(),
                )
            )
            if cls_head_config["hidden_dropout"] > 0:
                modules.append(models.Dropout(cls_head_config["hidden_dropout"]))

        last_in_features = (
            cls_head_config["hidden_features"]
            if cls_head_config["hidden_features"] > 0
            else embed_dim
        )
        assert isinstance(last_in_features, int)

        modules.append(
            models.Dense(
                in_features=last_in_features,
                out_features=2,
                activation_function=torch.nn.Identity(),
            )
        )

        self._model = SentenceTransformer(modules=modules, device=self._device)

        self._loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self._loss = tde_st_utils.losses.STLoss(self._model, self._loss_fn)

    def train(
        self,
        task: IMDBClassification,
        *,
        log_dir: Optional[str] = None,
        save_best_path: Optional[str] = None,
        early_stopping: bool = False,  # Not supported by sentence_transformers
    ) -> None:
        train_data = self._to_st_dataset(task.train)

        steps_in_epoch = len(train_data)

        evaluator = None
        if log_dir is not None:
            evaluators: list[evaluation.SentenceEvaluator] = []

            def take_first_tower(outputs: list):
                return outputs[0]

            train_writer = SummaryWriter(os.path.join(log_dir, "train"))
            evaluators.append(
                tde_st_utils.evaluation.TBMetricsEvaluator(
                    summary_writer=train_writer,
                    data_loader=train_data,
                    metrics={
                        "accuracy": MulticlassAccuracy(),
                        "loss": MeanLossMetric(self._loss_fn),
                        "used_vmem": VMemMetric(device=self._device),
                    },
                    metric_transforms={
                        "accuracy": take_first_tower,
                        "loss": take_first_tower,
                    },
                    steps_in_epoch=steps_in_epoch,
                )
            )

            if task.validation is not None:
                val_data = self._to_st_dataset(task.validation)
                val_writer = SummaryWriter(os.path.join(log_dir, "val"))

                evaluators.append(
                    tde_st_utils.evaluation.TBMetricsEvaluator(
                        summary_writer=val_writer,
                        data_loader=val_data,
                        metrics={
                            "accuracy": MulticlassAccuracy(),
                            "loss": MeanLossMetric(self._loss_fn),
                        },
                        metric_transforms={
                            "accuracy": take_first_tower,
                            "loss": take_first_tower,
                        },
                        steps_in_epoch=steps_in_epoch,
                        decisive_metric="accuracy",
                        decisive_metric_higher_is_better=True,
                    )
                )

            evaluator = evaluation.SequentialEvaluator(evaluators)

        self._model.fit(
            train_objectives=[(train_data, self._loss)],
            epochs=self._epochs,
            warmup_steps=self._warmup_steps,
            evaluator=evaluator,
            save_best_model=save_best_path is not None,
            output_path=save_best_path,
            use_amp=True,
        )

        if save_best_path is not None:
            self._model.load(save_best_path)

    def predict(self, inputs: Dataset) -> Iterable[np.ndarray]:
        for i in range(0, len(inputs), self._batch_size):
            batch = inputs[i : i + self._batch_size]
            yield self._model.encode(
                batch["text"],
                batch_size=self._batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

    def save(self, dir_path: str) -> None:
        self._model.save(dir_path)

    def load(self, dir_path: str) -> None:
        self._model = SentenceTransformer(dir_path, device=self._device)

    def _to_st_dataset(self, data: Dataset) -> DataLoader:
        ex_inputs = SBertIMDB.STDataset(data.with_format("torch"))
        return DataLoader(ex_inputs, batch_size=self._batch_size, shuffle=True)
