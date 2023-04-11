from typing import Iterable

import numpy as np
import tensorflow as tf

from transformer_document_embedding.tasks.hf_task import HFTask
from transformer_document_embedding.utils.evaluation import aggregate_batches


class IMDBClassification(HFTask):
    """Classification task done using the IMDB dataset.

    `datasets.Dataset` of documents; by default with 'train', 'test' and
    'unsupervised' splits.

    Each document is dictionary with keys:
        - 'text' (str) - text of the document,
        - 'label' (int) - 1/0 sentiment class index; not present for the
          unsupervised split
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("imdb", add_ids=True, **kwargs)

    def evaluate(self, pred_batches: Iterable[np.ndarray]) -> dict[str, float]:
        metrics = [
            tf.keras.metrics.BinaryCrossentropy(),
            tf.keras.metrics.BinaryAccuracy(),
        ]

        for met in metrics:
            met.reset_state()

        for pred_batch, true_batch in aggregate_batches(
            pred_batches, iter(self.splits["test"]), lambda x: x["label"]
        ):
            for met in metrics:
                met.update_state(true_batch, pred_batch)

        return {met.name: met.result().numpy() for met in metrics}
