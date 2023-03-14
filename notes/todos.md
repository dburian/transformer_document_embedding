# Next consultation

torch lightning and similar solutions
- weight decay for non layer-normalizations layers?

# TODOs

## Implementation

- [ ] Validation loss, early stopping, save best model

## Experiments

- [ ] Properly finetune SBERT on IMDB -- have best SBERT checkpoint
    - pull from repo
    - try training 3 variations until 10 epochs
        - best -- 25/0.1/0.5
        - no hidden -- 0/0.1/0
        - no label smoothing -- 25/0/0.5
- [ ] Finetune Longformer on IMDB -- have best Longformer checkpoint

## Research

- [ ] Define good document embeddings
- [ ] Based on the definition of good document embedding find other benchmarks
  and models
- [ ] STS and Spearman's correlation

## Cleanup

- [ ] Split pyproject installations to optional dependencies based on framework
- [ ] Transform `TFClassificationHead` into `tf.keras.Model`

