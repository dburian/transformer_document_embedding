from typing import Any, Iterable, Optional

import numpy as np
from datasets.arrow_dataset import Dataset

from transformer_document_embedding.baselines.experimental_model import \
    ExperimentalModel
from transformer_document_embedding.models.paragraph_vector import \
    ParagraphVector
from transformer_document_embedding.tasks.experimental_task import \
    ExperimentalTask
from transformer_document_embedding.utils.gensim.data import GensimCorpus


class ParagraphVectorWinesGames(ExperimentalModel):
    def __init__(
        self,
        dm_kwargs: Optional[dict[str, Any]] = None,
        dbow_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self._pv = ParagraphVector(dm_kwargs=dm_kwargs, dbow_kwargs=dbow_kwargs)

    def train(
        self,
        task: ExperimentalTask,
        *,
        log_dir: Optional[str] = None,
        save_best_path: Optional[str] = None,
        early_stopping: bool = False
    ) -> None:
        train_data = GensimCorpus(task.unsupervised.shuffle())

        for module in self._pv.modules:
            module.build_vocab(train_data)
            module.train(
                train_data,
                total_examples=module.corpus_count,
                epochs=module.epochs,
            )

        if save_best_path:
            self._pv.save(save_best_path)

    def predict(self, inputs: Dataset) -> Iterable[np.ndarray]:
        for doc in inputs:
            yield self._pv.get_vector(doc["id"])

    def save(self, dir_path: str) -> None:
        self._pv.save(dir_path)

    def load(self, dir_path: str) -> None:
        self._pv.load(dir_path)
