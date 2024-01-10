# Paragraph Vector implementation

I chose to implement PV using the `gensim` package. It seemed like
polished-enough code which copied the initial implementation.

## Convergence

Out-of-the-box gensim's PV does not return loss. It seems abnormal and
there have been lots of discussions around this. But the status is that there is
a PR, no ones working on it for some time now. This means that there is no
built-in way of telling how many epochs should one do.

Initially I thought I resolved the issue by watching the change in document
embeddings of randomly chosen documents. If change to their embeddings was only
marginal between epochs I concluded the training converged.

Later I realised gensim uses linear decay on alpha, which meant that even
though embeddings should change (i.e. they caused the loss to be higher), they
did not get to change because alpha was small and therefore dampened the
gradient. Normalizing the change by currently employed alpha did not help
either. The expectation is that the change should be linearly decreasing as the
training progresses. Instead the changes suddenly diminishes very quickly at the
end of the training, no matter if it trained for 5 or 30 epochs.

The above is to say I probably need another convergence metric.

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
