import logging
from typing import Iterable, Optional

import faiss
import numpy as np
from datasets.arrow_dataset import Dataset

from transformer_document_embedding.baselines.paragraph_vector import \
    ParagraphVectorWinesGames
from transformer_document_embedding.tasks.wikipedia_wines import \
    WikipediaSimilarities

# %%
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

task = WikipediaSimilarities(dataset="wine", datasets_dir="../data")
model = ParagraphVectorWinesGames(
    dm_kwargs={
        "vector_size": 100,
        "min_count": 2,
        "epochs": 30,
        "negative": 5,
        "sample": 0,
    }
)
# %%
model.train(task)
# %% [markdown]
# ## Manual way of getting nearest ids
# %%
def get_nearest_ids(
    test: Dataset, pred_embeddings: Iterable[np.ndarray]
) -> Iterable[tuple[list[int], list[int]]]:
    ids = []
    embeddings = []
    true_sims = {}

    for doc, embedding in zip(test, pred_embeddings):
        ids.append(doc["id"])
        embeddings.append(embedding)
        if len(doc["label"]) > 0:
            true_sims[doc["id"]] = [target_ar["id"] for target_ar in doc["label"]]

    embeddings = np.array(embeddings)
    embeddings /= np.sqrt(np.sum(embeddings * embeddings, axis=1))[:, None]

    sim_matrix = embeddings @ embeddings.T
    preds = dict(zip(ids, sim_matrix))

    for source_id, true_target_ids in true_sims.items():
        pred_target_ids = preds[source_id]
        pred_target_ids = sorted(
            zip(pred_target_ids, ids), key=lambda tuple: tuple[0], reverse=True
        )
        pred_target_ids = list(map(lambda tuple: tuple[1], pred_target_ids))[1:]
        yield true_target_ids, pred_target_ids


# %%
hits_at_ten = 0
hits_at_hundred = 0
reciprocal_rank = 0

total_queries = 0

for true_ids, pred_ids in get_nearest_ids(
    task.splits["test"], model.predict(task.test)
):
    unordered_true = set(true_ids)
    first_hit_ind = -1
    for i, pred_id in enumerate(pred_ids):
        if pred_id in unordered_true:
            if i < 10:
                hits_at_ten += 1
            if i < 100:
                hits_at_hundred += 1

            if first_hit_ind == -1:
                first_hit_ind = i

    reciprocal_rank += 1 / (first_hit_ind + 1)
    total_queries += 1


hit_ratio_at_ten = hits_at_ten / (10 * total_queries)
hit_ratio_at_hundred = hits_at_hundred / (100 * total_queries)
reciprocal_rank = reciprocal_rank / total_queries
# %%
print(reciprocal_rank)
print(hit_ratio_at_ten)
print(hit_ratio_at_hundred)
# %% [markdown]
# ## Using FAISS index
# %%
def get_nearest_ids_from_faiss(
    true_dataset: Dataset,
    embeddings: Iterable[np.ndarray],
    *,
    k: Optional[int] = None,
) -> Iterable[tuple[list[int], list[int]]]:
    embed_column_name = "embedding"
    faiss_dataset = true_dataset.add_column(
        name=embed_column_name,
        column=map(lambda vec: vec / np.linalg.norm(vec), embeddings),
    )
    faiss_dataset.add_faiss_index(
        embed_column_name, metric_type=faiss.METRIC_INNER_PRODUCT
    )

    if k is None:
        k = len(faiss_dataset)

    for article in faiss_dataset:
        if len(article["label"]) == 0:
            continue

        nearest_targets = faiss_dataset.get_nearest_examples(
            embed_column_name,
            np.array(article[embed_column_name]),
            k=k + 1,  # We're later removing the first hit, which is the query itself.
        )

        true_ids = [target_article["id"] for target_article in article["label"]]
        pred_ids = nearest_targets.examples["id"][1:]

        yield true_ids, pred_ids


# %%
hits_thresholds = [10, 100]
hits = [0 for _ in hits_thresholds]
reciprocal_rank = 0
percentile_ranks = []

total_queries = 0

max_rank = len(test_with_embeds) - 2
for true_ids, pred_ids in get_nearest_ids_from_faiss(
    task.splits["test"], model.predict(task.test)
):
    unordered_true = set(true_ids)

    def is_hit(target_id_with_rank: tuple[int, int]) -> bool:
        return target_id_with_rank[1] in unordered_true

    first_hit_ind = -1
    for i, pred_id in filter(is_hit, enumerate(pred_ids)):
        if first_hit_ind == -1:
            first_hit_ind = i

        percentile_ranks.append(i / max_rank)
        for hit_ind, threshold in enumerate(hits_thresholds):
            if i < threshold:
                hits[hit_ind] += 1

    reciprocal_rank += 1 / (first_hit_ind + 1)
    total_queries += 1


hit_rates = [
    hit_count / (threshold * total_queries)
    for threshold, hit_count in zip(hits_thresholds, hits)
]
reciprocal_rank = reciprocal_rank / total_queries
mean_percentile_rank = np.mean(percentile_ranks)
# %%
print(reciprocal_rank)
print(mean_percentile_rank)
print(hit_rates)
# %%


def smart_unbatch(
    iterable: Iterable[np.ndarray], dim_count: int
) -> Iterable[np.ndarray]:
    for batch in iterable:
        print(len(batch.shape))
        if len(batch.shape) > dim_count:
            for element in smart_unbatch(batch, dim_count):
                yield element
        else:
            yield batch


# %%
for elem in smart_unbatch(
    np.array(
        [
            [
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
                [5, 5],
            ],
            [[4, 4], [5, 5], [6, 6], [6, 6], [9, 9]],
        ]
    ),
    1,
):
    print(elem)
    print()
