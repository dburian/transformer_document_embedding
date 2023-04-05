[paragraph_vector]: paragraph_vector.md
[bigbird]: ./bigbird.md
[longformer]: ./longformer.md
[sbert]: ./sbert.md
[datasets]: ./datasets.md
[ginzburg_21]: https://arxiv.org/abs/2106.01186

# Baselines

We'd like to test how our model does in the given [tasks][datasets] compare to
other older models. The idea is to asses whether our model brings something new
to the table regarding long document embeddings.

In the ideal world we would compare our model to the existing techniques that
can embed long document embeddings. It is also beneficial to test
State-Of-The-Art (SOTA) models for given task to get the context of what is
possible.

## BOW models

- (TF-IDF -- just for reference)
- [Paragraph vector][paragraph_vector]

## Short document embeddings

- [SBert][sbert]

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

- SMITH
- CDLM

- [Longformer][longformer]
- [BigBird][bigbird]

# Unfit

TODO: move notes of the following paper into one note
- [Self-supervised Document Similarity Ranking (SDR)][ginzburg_21] -- RoBERTa
  backbone further trained with MLM and contrastive loss of positive and
  negative pairs drawn from the same and different document respectively. Also
  introduces special (and complex) inference technique of matching paragraphs.
  The model is only useful for similarities -- does not output embedding. **Good
  for comparisons in Wine and Game datasets, but I do not think it is a good
  baseline.**


TODO:
- SMASH-RNN
- LASER
- SPECTER
- HANs with RNN/Bert
