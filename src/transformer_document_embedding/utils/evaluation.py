from typing import Any, Callable, Iterable, Iterator, Optional

import numpy as np
from tqdm.auto import tqdm


def aggregate_batches(
    pred_batches: Iterable[np.ndarray],
    true_iter: Iterator[Any],
    transform_true: Optional[Callable[[Any], np.ndarray]] = None,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    def identity(x):
        return x

    if transform_true is None:
        transform_true = identity

    for pred_batch in pred_batches:
        true_batch = []
        while len(true_batch) < len(pred_batch):
            true_batch.append(transform_true(next(true_iter)))

        yield pred_batch, np.array(true_batch)


def smart_unbatch(
    iterable: Iterable[np.ndarray], dim_count: int
) -> Iterable[np.ndarray]:
    for batch in iterable:
        if len(batch.shape) > dim_count:
            yield from smart_unbatch(batch, dim_count)
        else:
            yield batch


def evaluate_ir_metrics(
    true_pred_ids_iterable: Iterable[tuple[list[int], list[int]]],
    *,
    hits_thresholds: list[int] | np.ndarray,
    iterable_length: Optional[int] = None,
    verbose: bool = False,
) -> dict[str, float]:
    hits_thresholds = np.array(hits_thresholds, dtype=np.int32)
    hit_percentages = [[] for _ in hits_thresholds]
    reciprocal_rank = 0
    percentile_ranks = []

    total_queries = 0

    for true_ids, pred_ids in tqdm(
        true_pred_ids_iterable,
        desc="Checking similarities",
        disable=not verbose,
        total=iterable_length,
    ):
        max_rank = len(pred_ids)
        first_hit_ind = max_rank
        query_hits = np.zeros(len(hits_thresholds))
        # For all predictions
        for i, pred_id in enumerate(pred_ids):
            # Skip those which are incorrect
            if pred_id not in true_ids:
                continue

            # Save the best-ranking correct prediction index
            if first_hit_ind > i:
                first_hit_ind = i

            percentile_ranks.append(i / max_rank)
            # For every correct prediction under a threshold we add 1
            query_hits += (i < hits_thresholds).astype("int32")

        # first_hit_ind could be zero if len(true_ids) == 0
        reciprocal_rank += 1 / (first_hit_ind if first_hit_ind > 0 else 1)
        total_queries += 1

        for perctanges, num_of_hits in zip(hit_percentages, query_hits, strict=True):
            perctanges.append(num_of_hits / len(true_ids))

    results = {
        "mean_reciprocal_rank": reciprocal_rank / total_queries,
        "mean_percentile_rank": np.mean(percentile_ranks),
    }

    for percentages, threshold in zip(hit_percentages, hits_thresholds, strict=True):
        results[f"hit_rate_at_{threshold}"] = np.mean(percentages)

    return results
