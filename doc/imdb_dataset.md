[article]: https://aclanthology.org/P11-1015
[download]: http://ai.stanford.edu/~amaas/data/sentiment/
[tf_dataset]: https://www.tensorflow.org/datasets/catalog/imdb_reviews

# IMDB dataset

Dataset for sentiment analysis containing short movie reviews. The task is to
classify the review to one of two classes: `positive` or `negative`.

## Links

- [article][article] to by cited if used
- [download][download]
- [Tensorflow Dataset][tf_datset]

## Summary

Dataset contains 50k reviews split equally to train/test sets. Each review is
connected to a movie. There are no more than 30 reviews for a given movie. Train
and test sets have disjoint sets of movies.

Review with >=7 points (out of 10) is marked as positive, reviews with <=4 are
marked as negative.

Additional data:
- unlabaled reviews for unsupervised learning,
- BOW feature vectors with associated vocabulary,
- pairs (review_id, URL for given movie).


## Task

The goal is to guess positive/negative label for the whole set of 50k reviews.
The determining metric is **classification accuracy**.

## Data analysis

- size: 50k
- classes: 2
- train/test split:
- class distribution:

- min/mean + std/max wordpiece length:
- examples:
