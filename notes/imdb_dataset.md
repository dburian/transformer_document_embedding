[article]: https://aclanthology.org/P11-1015
[download]: http://ai.stanford.edu/~amaas/data/sentiment/
[tf_dataset]: https://www.tensorflow.org/datasets/catalog/imdb_reviews
[hf_dataset]: https://huggingface.co/datasets/imdb
[d/imdb]: ../data/imdb.ipynb

# IMDB dataset

Dataset for sentiment analysis containing short movie reviews. The task is to
classify the review to one of two classes: `positive` or `negative`.

## Links

- [article][article] to by cited if used
- [Tensorflow Dataset][tf_datset]
- [Hugging face dataset][hf_dataset]

## Summary

Dataset contains 50k reviews split equally to train/test sets. Each review is
connected to a movie. There are no more than 30 reviews for a given movie. Train
and test sets have disjoint sets of movies.

Review with >=7 points (out of 10) is marked as positive, reviews with <=4 are
marked as negative.

Additional data:
- unlabeled reviews for unsupervised learning,
- BOW feature vectors with associated vocabulary,
- pairs (review_id, URL for given movie).


## Task

The goal is to guess positive/negative label for the whole set of 50k reviews.
The determining metric is **classification accuracy**.

### Paragraph vector

Embeddings dimensionality is 400 for both paragraphs and vectors. Sliding window
is 10 words. Special characters as ,.?! are treated as words, documents are
pre-pad with NULL word symbol to the required length of 9.

Learning paragraph and word embeddings on all 25k train and 50k unlabeled data.
Once the paragraph vectors are learned they are fed through a neural network
with single hidden layer with 50 units and a softmax layer used for
classification.

### Longformer

Single dense layer on top of the CLS token output with softmax activation and
Cross-Entropy loss. Additionally global attention is used on the CLS token.

## Data Analysis

[In notebook][d/imdb]


