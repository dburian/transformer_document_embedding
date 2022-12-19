
# Doc2Vec implementation

I chose to implement Doc2Vec using the `gensim` package. It seemed like
polished-enough code which copied the initial implementation.

## The missing loss problem

Out-of-the-box gensim's Doc2Vec does not return loss. It seems abnormal and
there have been lots of discussions around this. But the status is that there is
a PR, no ones working on it for some time now.

This means that there is no built-in way of telling how many epochs should one
do. I resolved the issue with my custom callback which for each training selects
randomly some number of documents and then computes cosine distances between
epochs. The idea is that once this number is close to zero we can stop learning.

In my first experiments this is achieved somewhere in the neighbourhood of 5-6
epochs.


## Hyperparameters

Although initially I though I would implement the model same as was described in
the paper, I've read
[here](https://groups.google.com/g/gensim/c/Ab4dcRaF9n8/m/XXl08mRiDgAJ) that it
is not advisable. Especially the following parameters:

- `dm_concat` - concatenation of context words rather then mean creates larger
  model that may not be worth it,
- `negative` - I remember that It was said that negative sampling is better
  choice then hierarchical. Though the original model used hierarchical softmax
  it may be beneficial in terms of speed to switch to negative sampling.
- `min_count` - maybe beneficial?
