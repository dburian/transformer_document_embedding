# Similarity-based tasks

## Definition

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
