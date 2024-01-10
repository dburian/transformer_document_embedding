# My SBERT implementation

Initially I used dedicated package for SBERT-like models
`sentence-transformers`. The major advantages were quick and easy manipulation
with models using already tested code. However to do anything special, one
needed to write a lot of code. Additionally I later realised that I can unify
all code regarding transformer into a single model class.

## Thinking process

I've been implementing the second sbert model and I thought I should consider
giving up `sentence_transformer` (abbreviated as `st`) code altogether and just
rely on `transformers` from hugging face.

Advantages of dropping `st`:

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

Disadvantages of dropping `st`:

- I'd have to fiddle with tokenizers for both training and prediction
- Prediction would be more complicates (but cca same code as is for longformer)
- saving done through pytorch -- does not really make any difference, as long as
  I initialize the to-be-loaded-to model equally as I did for the saved model

### Decision

Overall I think it is better to move away from `st`. I will drop a bunch of
code, maybe the code will start to resamble models based on longformer (and
other models from `transformers`) which could cause less code duplication.

**I've confirmed that using my 'mean' pooler on top of HF transformer equals
using `SentenceTransformer('model_name')`.**
