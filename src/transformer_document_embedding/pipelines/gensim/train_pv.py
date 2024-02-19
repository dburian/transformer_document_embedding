from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterator, cast

from datasets import Dataset, concatenate_datasets
from gensim.models.callbacks import CallbackAny2Vec
from transformer_document_embedding.datasets import col
from transformer_document_embedding.pipelines.pipeline import (
    TrainPipeline,
)
import os

from transformer_document_embedding.utils.gensim import (
    GensimCorpus,
    TextPreProcessor,
    create_text_pre_processor,
)

if TYPE_CHECKING:
    from transformer_document_embedding.models.pv import ParagraphVector
    from transformer_document_embedding.datasets.document_dataset import DocumentDataset
    from typing import Optional


class CheckpointSave(CallbackAny2Vec):
    """Callback to periodically save the model."""

    def __init__(
        self,
        epoch_checkpoints: list[int],
        save_dir: str,
        paragraph_vector: ParagraphVector,
    ) -> None:
        self._epoch_checkpoints = epoch_checkpoints
        self._save_dir = save_dir
        self._epoch = 0
        self._pv = paragraph_vector

        os.makedirs(self._save_dir, exist_ok=True)

    def on_epoch_end(self, _) -> None:
        if self._epoch in self._epoch_checkpoints:
            model_path = os.path.join(self._save_dir, f"after_epoch_{self._epoch}")
            self._pv.save_weights(model_path)

        self._epoch += 1


def compute_alpha(
    total_epochs: int,
    cur_epoch: int,
    start_alpha: float = 0.025,
    end_alpha: float = 1e-4,
) -> float:
    progress = cur_epoch / total_epochs
    next_alpha = start_alpha - (start_alpha - end_alpha) * progress
    next_alpha = max(end_alpha, next_alpha)
    return next_alpha


class TrainPVPipeline(TrainPipeline):
    def __init__(
        self, start_at_epoch: Optional[int], save_at_epochs: Optional[list[int]]
    ) -> None:
        super().__init__()

        self._start_at_epoch = start_at_epoch
        self._save_at_epochs = save_at_epochs

        if self._start_at_epoch is not None and self._save_at_epochs is not None:
            self._save_at_epochs = [
                epoch - self._start_at_epoch for epoch in self._save_at_epochs
            ]

    def to_gensim_corpus(
        self, dataset: Dataset, model: ParagraphVector
    ) -> GensimCorpus:
        return GensimCorpus(
            dataset,
            text_pre_processor=create_text_pre_processor(model.text_pre_process),
            num_proc=model.workers,
        )

    def __call__(
        self,
        model: ParagraphVector,
        _,
        dataset: DocumentDataset,
        log_dir: Optional[str],
    ) -> None:
        all_splits = [
            split
            for split_name, split in dataset.splits.items()
            if split_name not in ["test", "validation"]
        ]

        train_data = self.to_gensim_corpus(
            concatenate_datasets(all_splits).shuffle(), model
        )
        callbacks = []

        if log_dir is not None and self._save_at_epochs is not None:
            callbacks.append(
                CheckpointSave(
                    epoch_checkpoints=self._save_at_epochs,
                    save_dir=os.path.join(log_dir, "checkpoints"),
                    paragraph_vector=model,
                )
            )

        if self._start_at_epoch is None:
            model.build_vocab(train_data)

        train_kwargs: dict[str, Any] = {
            "epochs": model.epochs,
            "callbacks": callbacks,
        }

        if self._start_at_epoch is not None:
            assert isinstance(model.epochs, int)

            train_kwargs["start_alpha"] = compute_alpha(
                total_epochs=model.epochs,
                cur_epoch=self._start_at_epoch,
            )
            train_kwargs["end_alpha"] = 1e-4
            train_kwargs["epochs"] -= self._start_at_epoch

        model.train(
            train_data,
            total_examples=model.corpus_count,
            **train_kwargs,
        )


class PairedGensimCorpus(GensimCorpus):
    """Gensim corpus for pairs of sentences as one item.

    E.g. for pairwise classification tasks, where PV needs each sentence separately."""

    def __init__(
        self,
        dataset: Dataset,
        text_pre_processor: TextPreProcessor,
        num_proc: int = 0,
    ) -> None:
        self._text_preprocessor = text_pre_processor
        self._dataset = Dataset.from_generator(
            self._unique_gensim_docs_iter,
            gen_kwargs={"pairs_dataset": dataset},
            num_proc=num_proc,
        )

    def _unique_gensim_docs_iter(
        self, pairs_dataset: Dataset
    ) -> Iterator[dict[str, Any]]:
        def _duplicates_iter() -> Iterator[dict[str, Any]]:
            for pair in pairs_dataset:
                pair = cast(dict[str, Any], pair)
                yield {
                    "words": self._text_preprocessor(pair[col.TEXT_0]),
                    "tag": pair[col.ID_0],
                }
                yield {
                    "words": self._text_preprocessor(pair[col.TEXT_1]),
                    "tag": pair[col.ID_1],
                }

        seen_tags = set()
        for doc in _duplicates_iter():
            if doc["tag"] in seen_tags:
                continue
            seen_tags.add(doc["tag"])
            yield doc


class TrainPairPVPipeline(TrainPVPipeline):
    def to_gensim_corpus(
        self,
        dataset: Dataset,
        model: ParagraphVector,
    ) -> GensimCorpus:
        return PairedGensimCorpus(
            dataset,
            text_pre_processor=create_text_pre_processor(model.text_pre_process),
            num_proc=model.workers,
        )
