[paper]: https://arxiv.org/abs/1908.10084
[i/sbert_pairs_architectures]: ./imgs/sbert_pairs_architectures.png

# SBert

Brief description of model introduced by [Reimers, Gurevych in 2019][paper].

It's interesting because conceptually they did what I'm trying to do:

1. take a transformer,
2. adjust the architecture so it produces nice embeddings.

As them I should focus on matching the performance of the original transformer,
while making it easier to find similar pieces of text.

## Architecture

SBert is composed of three layers:

- Bert-like transformer encoder (could also be RoBerta),
- some sort of pooling network over Bert's outputs (average was the best out of
  max and getting CLS's output and average),
- computation of loss for the model.

The third layer differs according to the data given. For pairs of inputs we
have siamese networks, where the loss is computed differently based on type of
gold labels:

- regression: computing the cosine similarity of the pooling layers' outputs
  with Mean Squared Error loss,
- classification (of pair of sentences): softmax layer from concatenation of the
  two pooling layers' outputs and their difference to Cross-Entropy loss.

![SBert Architecture for pairs of inputs][i/sbert_pairs_architectures]

For triplets -- anchor, positive and negative example -- we have triplet network
computing triplet loss.

## Learning

They trained on pairs of sentences classified into three classes:
*contradiction*, *entailment*, *neutral*.
