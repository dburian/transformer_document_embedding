from __future__ import annotations
from typing import Iterable, Optional, Any

import numpy as np
import torch
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    evaluation,
    models,
)

from torch.utils.data import DataLoader
from datasets.arrow_dataset import Dataset

from transformer_document_embedding.baselines.experimental_model import (
    ExperimentalModel,
)
from transformer_document_embedding.tasks.imdb import IMDBClassification, IMDBData
import transformer_document_embedding.utils.sentence_transformers as tde_st_utils
import transformer_document_embedding.utils.torch as torch_utils


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
        transformer_model: str = "all-distilroberta-v1",
        batch_size: int = 37,
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
        embed_dim = transformer.get_sentence_embedding_dimension()
        modules.append(transformer)

        if cls_head_kwargs is None:
            cls_head_kwargs = {}

        cls_head_config = {
            "hidden_features": 25,
            "hidden_dropout": 0.5,
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
        modules.append(
            models.Dense(
                in_features=last_in_features,
                out_features=1,
                activation_function=torch.nn.Sigmoid(),
            )
        )

        self._model = SentenceTransformer(modules=modules, device=self._device)

        self._loss = tde_st_utils.losses.BCELoss(
            self._model, label_smoothing=label_smoothing
        )

    def train(
        self,
        task: IMDBClassification,
        *,
        log_dir: Optional[str] = None,
        save_best_path: Optional[str] = None,
        early_stopping: bool = False,  # Not supported by sentence_transformers
    ) -> None:
        # TODO: Why use sentence_transformers? Lets first test longformer
        # before doing large rewriting here.
        evaluator = None
        if log_dir is not None:
            # TODO: Could be done more efficiently, not to have the network encode
            # the same set of text twice.
            loss_evaluator = tde_st_utils.evaluation.LossEvaluator(
                task.train,
                log_dir,
                # TODO: Not the same loss as is used during training, due to
                # label_smoothing
                torch.nn.BCELoss(),
                batch_size=self._batch_size,
            )
            accuracy_evaluator = tde_st_utils.evaluation.AccuracyEvaluator(
                task.train, log_dir, batch_size=self._batch_size
            )
            vmem_evaluator = tde_st_utils.evaluation.VMemEvaluator(log_dir)
            evaluator = evaluation.SequentialEvaluator(
                [loss_evaluator, vmem_evaluator, accuracy_evaluator]
            )
            # TODO: Evaluator on validation data

        train_size = len(task.train)
        training_data = self._to_st_dataset(task.train)
        self._model.fit(
            train_objectives=[(training_data, self._loss)],
            epochs=self._epochs,
            warmup_steps=self._warmup_steps,
            evaluator=evaluator,
            evaluation_steps=train_size,
            save_best_model=save_best_path is not None,
            output_path=save_best_path,
            use_amp=True,
        )

    def predict(self, inputs: IMDBData) -> Iterable[np.ndarray]:
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

    def _to_st_dataset(self, data: IMDBData) -> DataLoader:
        ex_inputs = SBertIMDB.STDataset(data.with_format("torch"))
        return DataLoader(ex_inputs, batch_size=self._batch_size, shuffle=True)


Model = SBertIMDB
