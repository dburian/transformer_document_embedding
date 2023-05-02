from typing import Any, Callable, Iterable, Iterator, Optional

import numpy as np
from tqdm.auto import tqdm


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


def smart_unbatch(
    iterable: Iterable[np.ndarray], dim_count: int
) -> Iterable[np.ndarray]:
    for batch in iterable:
        if len(batch.shape) > dim_count:
            for element in smart_unbatch(batch, dim_count):
                yield element
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
    hits = np.array([0 for _ in hits_thresholds])
    reciprocal_rank = 0
    percentile_ranks = []

    total_queries = 0

    for true_ids, pred_ids in tqdm(
        true_pred_ids_iterable,
        desc="Checking similarities",
        disable=not verbose,
        total=iterable_length,
    ):
        max_rank = len(pred_ids) - 1
        unordered_true = set(true_ids)

        def is_hit(target_id_with_rank: tuple[int, int]) -> bool:
            return target_id_with_rank[1] in unordered_true

        first_hit_ind = -1
        query_hits = np.zeros_like(hits)
        for i, _ in filter(is_hit, enumerate(pred_ids)):
            if first_hit_ind == -1:
                first_hit_ind = i

            percentile_ranks.append(i / max_rank)
            query_hits += i < hits_thresholds

        reciprocal_rank += 1 / (first_hit_ind + 1)
        total_queries += 1
        hits += np.clip(query_hits, None, 1)

    results = {
        "mean_reciprocal_rank": reciprocal_rank / total_queries,
        "mean_percentile_rank": np.mean(percentile_ranks),
    }

    for hit_count, threshold in zip(hits, hits_thresholds):
        results[f"hit_rate_at_{threshold}"] = hit_count / total_queries

    return results
