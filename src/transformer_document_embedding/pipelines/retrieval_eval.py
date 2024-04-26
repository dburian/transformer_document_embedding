from __future__ import annotations
from dataclasses import dataclass
from functools import reduce
from typing import Any, Iterable, Optional, TYPE_CHECKING, cast
import faiss
import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score
from tqdm.auto import tqdm

from transformer_document_embedding.datasets import col
from transformer_document_embedding.pipelines.classification_finetune import (
    get_default_features,
)
from transformer_document_embedding.pipelines.pipeline import EvalPipeline


if TYPE_CHECKING:
    import torch
    from datasets import Dataset
    from transformer_document_embedding.models.embedding_model import EmbeddingModel
    from transformer_document_embedding.datasets.document_dataset import DocumentDataset


@dataclass(kw_only=True)
class RetrievalEval(EvalPipeline):
    batch_size: int

    def _get_nearest_ids_from_faiss(
        self,
        dataset: Dataset,
        *,
        k: Optional[int] = None,
    ) -> Iterable[tuple[list[int], list[int]]]:
        def norm_embeddings(docs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            embeds = docs[col.EMBEDDING]
            normed = embeds / np.linalg.norm(embeds, axis=1).reshape(-1, 1)
            return {col.EMBEDDING: normed}

        faiss_dataset = dataset.with_format("numpy").map(
            norm_embeddings, batched=True, batch_size=256
        )
        faiss_dataset.add_faiss_index(
            col.EMBEDDING, metric_type=faiss.METRIC_INNER_PRODUCT
        )

        if k is None:
            k = len(faiss_dataset)

        for article in faiss_dataset:
            article = cast(dict[str, Any], article)

            if len(article[col.LABEL]) == 0:
                continue

            nearest_targets = faiss_dataset.get_nearest_examples(
                col.EMBEDDING,
                np.array(article[col.EMBEDDING]),
                k=k
                + 1,  # We're later removing the first hit, which is the query itself.
            )

            true_ids = [target_article[col.ID] for target_article in article[col.LABEL]]
            pred_ids = nearest_targets.examples[col.ID][1:]

            yield true_ids, pred_ids

    def _evaluate_ir_metrics(
        self,
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

        average_precisions = []
        ndcgs = []

        total_queries = 0

        for true_ids, pred_ids in tqdm(
            true_pred_ids_iterable,
            desc="Checking similarities",
            disable=not verbose,
            total=iterable_length,
        ):
            # We assume we go over queries with some true positives
            assert len(true_ids) > 0

            max_rank = len(pred_ids)
            first_hit_ind = max_rank
            query_hits = np.zeros(len(hits_thresholds))

            bin_true = np.isin(pred_ids, true_ids)
            # generate artificial scores, since only the order matters
            pred_score = np.linspace(1, 0, len(pred_ids))
            average_precisions.append(average_precision_score(bin_true, pred_score))
            ndcgs.append(
                ndcg_score(
                    bin_true.reshape((1, -1)),
                    pred_score.reshape((1, -1)),
                    ignore_ties=True,
                )
            )

            # For all predictions
            for i, pred_id in enumerate(pred_ids, start=1):
                # Skip those which are incorrect
                if pred_id not in true_ids:
                    continue

                # Save the best-ranking correct prediction index
                if first_hit_ind > i:
                    first_hit_ind = i

                # So that MPR is between 0 and 1
                percentile_ranks.append((i - 1) / (max_rank - 1))
                # For every correct prediction under a threshold we add 1
                query_hits += (i <= hits_thresholds).astype("int32")

            reciprocal_rank += 1 / first_hit_ind
            total_queries += 1

            for perctanges, num_of_hits in zip(
                hit_percentages, query_hits, strict=True
            ):
                perctanges.append(num_of_hits / len(true_ids))

        results = {
            "mean_reciprocal_rank": reciprocal_rank / total_queries,
            "mean_percentile_rank": np.mean(percentile_ranks).item(),
            "map": np.mean(average_precisions).item(),
            "ndcg": np.mean(ndcgs).item(),
        }

        for percentages, threshold in zip(
            hit_percentages, hits_thresholds, strict=True
        ):
            results[f"hit_rate_at_{threshold}"] = np.mean(percentages).item()

        return results

    def __call__(
        self,
        model: EmbeddingModel,
        _: Optional[torch.nn.Module],
        dataset: DocumentDataset,
    ) -> dict[str, float]:
        test_split = dataset.splits["test"]
        with_embeds = get_default_features(
            test_split, model, batch_size=self.batch_size
        )

        true_pred_ids_iter = self._get_nearest_ids_from_faiss(
            with_embeds,
            k=1000,
        )

        test_sims_total = reduce(
            lambda acc, doc: acc + int(len(doc[col.LABEL]) > 0),
            test_split,
            0,
        )

        return self._evaluate_ir_metrics(
            true_pred_ids_iter,
            hits_thresholds=[10, 100],
            iterable_length=test_sims_total,
            verbose=True,
        )
