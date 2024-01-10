# Unifying all transformers into single model

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
