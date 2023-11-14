from __future__ import annotations
from typing import TYPE_CHECKING
import os
import logging

import datasets
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.tensorboard.writer import SummaryWriter
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, WindowedBinaryAUROC
from torcheval.metrics.classification import BinaryRecall
from transformer_document_embedding.baselines.baseline import Baseline
from transformer_document_embedding.baselines.longformer.train import LongformerTrainer
from transformer_document_embedding.models.paragraph_vector import ParagraphVector
from transformer_document_embedding.models.torch.cls_head import ClsHead
from transformer_document_embedding.utils.gensim.data import PairedGensimCorpus
from transformer_document_embedding.utils.metrics import with_accessor
import transformer_document_embedding.utils.torch.training as train_utils
import torch


if TYPE_CHECKING:
    from transformer_document_embedding.tasks.experimental_task import ExperimentalTask
    from typing import Optional, Any, Iterable
    import numpy as np

logger = logging.getLogger(__name__)


class PairClassifier(Baseline):
    def __init__(
        self,
        dbow_kwargs: Optional[dict[str, Any]],
        dm_kwargs: Optional[dict[str, Any]],
        cls_head_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()

        self._pv = ParagraphVector(dbow_kwargs=dbow_kwargs, dm_kwargs=dm_kwargs)

        self._batch_size = cls_head_kwargs.pop("batch_size")

        self._cls_head = ClsHead(
            **cls_head_kwargs,
            in_features=2 * self._pv.vector_size,
            num_classes=2,
        )

    def train(
        self,
        task: ExperimentalTask,
        pv_epochs: int,
        cls_head_epochs: int,
        log_every_step: Optional[int],
        save_best: bool,
        log_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        **_,
    ) -> None:
        # TODO: Add `.splits() -> dict[str, Any]` to ExperimentalTask
        all_datasets = [task.train, task.validation, task.test]

        pv_train = datasets.combine.concatenate_datasets(all_datasets).shuffle()
        pv_train = PairedGensimCorpus(pv_train)

        for module in self._pv.modules:
            module.build_vocab(pv_train)
            module.train(pv_train, total_examples=module.corpus_count, epochs=pv_epochs)

        cls_head_train_data = self._create_features_dataloader(task.train)

        cls_head_val_data = None
        if task.validation is not None:
            cls_head_val_data = self._create_features_dataloader(
                task.validation, training=False
            )

        def _create_summary_writer(name: str) -> Optional[SummaryWriter]:
            if log_dir is None:
                return None

            return SummaryWriter(log_dir=os.path.join(log_dir, name))

        summary_writer = _create_summary_writer("train")
        val_summary_writer = _create_summary_writer("val")

        save_model_callback = None
        if save_best and model_dir is not None:

            def save_cb(_, total_step: int) -> None:
                self.save(os.path.join(model_dir, f"checkpoint_{total_step}"))

            save_model_callback = save_cb

        def _update_outputs_with_labels(metric, outputs, batch) -> None:
            metric.update(
                outputs["output"].round().type(torch.int32),
                batch["labels"].round().type(torch.int32),
            )

        # TODO: Move and call it Pytorch trainer
        optim = torch.optim.Adam(
            params=train_utils.get_optimizer_params(self._cls_head, 0), lr=3e-5
        )
        model = _ModelWithLoss(self._cls_head, torch.nn.BCELoss())
        trainer = LongformerTrainer(
            model=model,
            train_data=cls_head_train_data,
            val_data=cls_head_val_data,
            optimizer=optim,
            lr_scheduler=train_utils.get_cosine_lr_scheduler(
                optim, cls_head_epochs * len(cls_head_train_data)
            ),
            log_every_step=log_every_step,
            summary_writer=summary_writer,
            val_summary_writer=val_summary_writer,
            save_model_callback=save_model_callback,
            metrics={
                "accuracy": with_accessor(
                    BinaryAccuracy(), _update_outputs_with_labels
                ),
                "precision": with_accessor(
                    BinaryPrecision(), _update_outputs_with_labels
                ),
                "recall": with_accessor(BinaryRecall(), _update_outputs_with_labels),
                "auprc": with_accessor(
                    WindowedBinaryAUROC(), _update_outputs_with_labels
                ),
            },
        )

        trainer.train(epochs=cls_head_epochs)

    @torch.inference_mode()
    def predict(self, inputs: datasets.Dataset) -> Iterable[np.ndarray]:
        cls_head_test = self._create_features_dataloader(inputs, training=False)

        for batch in cls_head_test:
            yield self._cls_head(batch["embeddings"]).squeeze().numpy(
                force=True
            ).round()

    def _create_features_dataloader(
        self, dataset: datasets.Dataset, training: bool = True
    ) -> DataLoader:
        return DataLoader(
            FeaturesDataset(dataset, self._pv),
            batch_size=self._batch_size,
            shuffle=training,
        )

    @classmethod
    def _pv_save_dir(cls, dir_path: str) -> str:
        return os.path.join(dir_path, "pv")

    @classmethod
    def _cls_head_save_dir(cls, dir_path: str) -> str:
        return os.path.join(dir_path, "cls_head")

    def save(self, dir_path: str) -> None:
        self._pv.save(self._pv_save_dir(dir_path))
        torch.save(self._cls_head.state_dict(), self._cls_head_save_dir(dir_path))

    def load(self, dir_path: str) -> None:
        self._pv.load(self._pv_save_dir(dir_path))
        try:
            state_dict_path = self._cls_head_save_dir(dir_path)
            self._cls_head.load_state_dict(torch.load(state_dict_path))
        except Exception as e:
            logger.warn(
                "Error when loading classification head. Skipping. Error: %s", e
            )


class _ModelWithLoss(torch.nn.Module):
    def __init__(
        self,
        cls_head: torch.nn.Module,
        loss_fn: torch.nn.Module,
        # TODO: Label smoothing?
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cls_head = cls_head
        self.loss_fn = loss_fn

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor, **_
    ) -> dict[str, torch.Tensor]:
        output = self.cls_head(embeddings).squeeze()
        loss = self.loss_fn(output, labels)

        return {"output": output, "loss": loss}


class FeaturesDataset(TorchDataset):
    def __init__(
        self, text_dataset: datasets.Dataset, embedding_model: ParagraphVector
    ) -> None:
        super().__init__()

        self._text_dataset = text_dataset
        self._embed_model = embedding_model

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self._text_dataset[index]
        pair_id = item["id"]

        embeddings = []
        for pair_idx in range(2):
            key_embed = self._embed_model.get_vector(
                PairedGensimCorpus.transform_id(pair_id, pair_idx)
            )
            embeddings.append(torch.tensor(key_embed))

        features_item = {
            "embeddings": torch.cat(embeddings),
        }
        if "label" in item:
            features_item["labels"] = torch.tensor(item["label"], dtype=torch.float)

        return features_item

    def __len__(self) -> int:
        return len(self._text_dataset)
