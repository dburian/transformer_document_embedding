# Similarity-based tasks

## Definition

Currently similarity-based tasks output all articles (meaning the things that we
compare using similarities) on all splits. However, for the test (and
validation) splits some articles will have 'label' key with list of all similar
articles.

### Data format

Below is an example of a format of single article:

```python
train_article = {
    'id': 42,
    'text': 'whole text of the article',
}

# Article with gold similarities
source_test_article = {
    'id': 42,
    'text': 'whole text of the article',
    'labels': [
        {
            'id': 43,
            'text': 'Another whole text of the article',
        },
        ...
    ]
}

# Article without gold similarities
non_source_test_article = {
    'id': 43,
    'text': 'Another whole text of the article',
}
```

## Evaluation

For evaluation the currently implemented approach is to:

- call function that iterates over predicted and gold similar articles
    - adds normalized predictinons (=embeddings) for each article into a hf dataset
    - adds faiss index for the embedding column with inner product as similarity
    - for each article with labels return the labels and the k top similar
      articles according to the index
- call function that takes the iterator and computes IR metrics

Currently there is only a single function which computes all metrics:

- Mean Reciprocal Rank -- mean of 1/first_hit_index of predicted documents for
  each query
- Mean Percentile Rank -- mean of hit_index/number_of_gold_docs for each predicted,
  gold document pair
- HitRatio@k -- total hits with hit_index <= k divided by the number of hits

## Why's

### Similarity or embedding based?

There are multiple ways how to define tasks, which are based on similarity of
documents:

1. make models output similarities of documents

This is the basic approach giving models the most amount of flexibility. Models
can e.g. process pairs of documents to create similarities and therefore compute
similarities directly.

2. make models output embeddings **Chosen**

This is more restrictive, but more applicable to my thesis. We are searching for
good document embedding, therefore it does not make sense to evaluate
similarities directly.

We can always have multiple tasks using the same data, only providing different
transformations of those data to suite given definition of similarity task.

### Why use faiss with HF dataset as vector database

Since we don't need anything super-scalable or fancy I went for the
quickest-to-implement approach. Since HF datasets offer interface for faiss for free
and faiss is very capable, there is not much to talk about.
