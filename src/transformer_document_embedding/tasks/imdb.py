from typing import Iterable

import numpy as np
import tensorflow as tf

from transformer_document_embedding.tasks.hf_task import HFTask


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

        true_iter = iter(self.dataset["test"])
        for pred_batch in pred_batches:
            true_batch = []
            while len(true_batch) < len(pred_batch):
                true_batch.append(next(true_iter)["label"])

            for met in metrics:
                met.update_state(true_batch, pred_batch)

        return {met.name: met.result().numpy() for met in metrics}
