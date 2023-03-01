[d/bigbird]: ./bigbird.md
[d/longformer]: ./longformer.md
[d/sbert]: ./sbert.md
[d/doc2vec]: ./doc2vec.md
[d/datasets]: ./datasets.md

# Baselines

We'd like to test how our model does in the given [tasks][d/datasets] compare to
other older models. The idea is to asses whether our model brings something new
to the table regarding long document embeddings.

In the ideal world we would compare our model to the existing techniques that
can embed long document embeddings. It is also beneficial to test
State-Of-The-Art (SOTA) models for given task to get the context of what is
possible.

## BOW models

- (TF-IDF -- just for reference)
- [Doc2vec][d/doc2vec]

## Short document embeddings

- [SBert][d/sbert]

## Long document embeddings

- Averaging of Longformer's output layer (similarly to what was done with BERT
  before SBert according to SBert)

> Researchers have started to input individual sentences into BERT and to derive
> fixedsize sentence embeddings. The most commonly used approach is to average
> the BERT output layer (known as BERT embeddings) or by using the output of the
> first token (the [CLS] token). As we will show, this common practice yields
> rather bad sentence embeddings, often worse than averaging GloVe embeddings
> (Pennington et al., 2014).
> (for example: May et al. (2019); Zhang et al. (2019); Qiao et al. (2019)).

TODO: LASER

## SOTAs

- [Longformer][d/longformer]
- [BigBird][d/bigbird]
