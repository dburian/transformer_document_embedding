from typing import Any, Callable, Iterable, Iterator, Optional

import numpy as np


def aggregate_batches(
    pred_batches: Iterable[np.ndarray],
    true_iter: Iterator[Any],
    transform_true: Optional[Callable[[Any], np.ndarray]] = None,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    if transform_true is None:
        transform_true = lambda x: x

    for pred_batch in pred_batches:
        true_batch = []
        while len(true_batch) < len(pred_batch):
            true_batch.append(transform_true(next(true_iter)))

        yield pred_batch, np.array(true_batch)
