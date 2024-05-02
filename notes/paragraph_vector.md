[paper]: https://arxiv.org/abs/1405.4053
[imgs/pv-dm]: ./imgs/pv-dm.png
[imgs/pv-dbow]: ./imgs/pv-dbow.png

# Paragraph vector

Brief description of PV-DM and PV-DBOW models introduced by [Mikolov, Le in
2014][paper]. Then revisited by [Dai et al. (2015)](https://arxiv.org/pdf/1507.07998.pdf)

Each of the two models creates a dense vector representation of an arbitrarily
long piece of text. The two representations are usually concatenated, and thus
form a one single model. This model is often regarded as doc2vec or simply
Paragraph vector (as suggested by the authors).

## Distributed Memory model of Paragraph Vectors: PV-DM

PV-DM is similar to CBOW model of word2vec. Given context words and a context
paragraph, the model should predict a word appearing in that context.

There are two embedding matrices: $D$ for the paragraphs and $W$ for the words.
The model concatenates all representations, runs them through a single dense
layer as large as there are words with a softmax activation. In the simplest of
cases the loss is cross-entropy.

![PV-DM][imgs/pv-dm]

During training the weights of the hidden layer, $D$ and $W$ are learned. In
prediction $W$ and weights of the hidden layer are fixed, while new rows in $D$
are learned.

## Distributed Bag of Words version of Paragraph Vectors: PV-DBOW

Similar to skip-gram model of word2vec, PV-DBOW should given the paragraph
predict words in it.

PV-DBOW requires only a single embedding matrix $D$. The paragraph embedding is
fed through dense layer as large as there are words with a softmax activation.
The loss is again cross-entropy.

![PV-DBOW][imgs/pv-dbow]

During training the weights of the hidden layer are learned while in prediction
are fixed.

## Extensions

- [Gynsel et al. (2018) -- NVSM](https://dl.acm.org/doi/abs/10.1145/3196826)
    - embedding matrix for words & documents, projection from word space to document space
    - minimizing distance of projection of m-gram from document $i$ and the embedding of
  document $i$ while maximizing distances of embedding of document $i$ and
  embedding of all other documents in batch
    - evaluation on IR tasks
- [Ganesh et al. (2016) --
  Doc2Sent2Vec](http://audentia-gestion.fr/MICROSOFT/jawahar16_sigir.pdf)
    - 2 phase model, first PV-DM architecture learns sentence embedding
      (similarly as PV-DM learns document embeddings), then in similar fashion
      document embedding is learned from sentence embeddings
    - evaluation on two classification tasks, outscoring PV
- [Liu et al
  (2017)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7966147)
    - PV but with extra embedding for embedding of entities that is shared among
      several documents
    - the main goal of the paper was to do domain specific task with
      PV


## Follow-up papers

- [Ai et al. (2016)](https://dl.acm.org/doi/abs/10.1145/2970398.2970409) --
  analysed PV for IR tasks
- [Ai et al. (2016)](https://dl.acm.org/doi/abs/10.1145/2911451.2914688) --
  improved PV for IR tasks

- [Lau et al. (2016)](https://arxiv.org/pdf/1607.05368.pdf) -- proposed some
  hyperparameters for two sets of tasks
    -- based on this paper I used `dbow_words=1`
