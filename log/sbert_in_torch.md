# SBERT (and other sentence transformers) in torch

I've been implementing the second sbert model and I thought I should consider
giving up `sentence_transformer` (abbreviated as `st`) code altogether and just
rely on `transformers` from hugging face.

## Advantages of dropping `st`

- no special code for metrics -- `st` use the princip of evaluators, to which I
  had to adapt metrics from `torcheval`
- no extra loop through data -- `st` metrics are only collected using
  evaluators, not during training
- no special network modules -- `st` has special modules, which are
  `torch.nn.Module`s but with something extra for saving and loading
- no special dataset -- `st` uses different feeding of data which requires
  special dataset
- no gradient accumulation steps -- `st` training doesn't allow gradient
  accumulation steps
- no early stopping -- unsupported by `st`

## Disadvantages of dropping `st`

- I'd have to fiddle with tokenizers for both training and prediction
- Prediction would be more complicates (but cca same code as is for longformer)
- saving done through pytorch -- does not really make any difference, as long as
  I initialize the to-be-loaded-to model equally as I did for the saved model

## Decision

Overall I think it is better to move away from `st`. I will drop a bunch of
code, maybe the code will start to resamble models based on longformer (and
other models from `transformers`) which could cause less code duplication.

**I've confirmed that using my 'mean' pooler on top of HF transformer equals
using `SentenceTransformer('model_name')`.**

## Difference from `longformer/*.py` and `sbert/*.py` and `bigbird/*.py`

Currently I have three different models for the same task for each architecture:
longformer, sbert, bigbird. There is the question what is the difference between
them aside from getting different transformer from `transformers`.

Differences:
- Longformer uses some of my custom `PreTrainedModels` that have inside
  classification head/pooler head/... If I would wanted this for SBERT as well,
  I would have to implement these models (when in fact their `forward` would be
  identical). Instead I would rather to move away from `PreTrainedModels`, going
  pure pytorch. The only **distadvantage is that I'd have to save/load all
  pieces of the model(transformer, pooler/classification head, loss, ...)
  seprately**.
