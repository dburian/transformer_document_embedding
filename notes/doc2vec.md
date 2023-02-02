[paper]: https://arxiv.org/abs/1405.4053
[imgs/pv-dm]: ./imgs/pv-dm.png
[imgs/pv-dbow]: ./imgs/pv-dbow.png

# Doc2Vec

Brief description of PV-DM and PV-DBOW models introduced by [Mikolov, Le in
2014][paper].

Each of the two models creates a dense vector representation of an arbitrarly
long piece of text. The two representations are usually concatenated, and thus
form a one single model. This model is often regarded as doc2vec or simply
paragraph vector (as suggested by the authors).

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


