# Next consultation

torch lightning and similar solutions
- weight decay for non layer-normalizations layers?

# TODOs


## Experiments

- [ ] Properly finetune SBERT on IMDB -- have best SBERT checkpoint
    - try training 3 variations until 10 epochs
        - best -- 25/0.1/0.5
        - no hidden -- 0/0.1/0
        - no label smoothing -- 25/0/0.5
- [ ] Finetune Longformer on IMDB -- have best Longformer checkpoint

## Implementation

- `SBERT`
    - [ ] add sentence-transformers classification head
    - [x] rewrite metrics using `torcheval`
    - [x] maybe use different activation to use same loss in logging as in training
    - [x] try to implement validation split

## Research

- [ ] Define good document embeddings
- [ ] Based on the definition of good document embedding find other benchmarks
  and models
- [ ] STS and Spearman's correlation
- [ ] Cleanup baselines/related work notes --- move significant models into
  their own note

## Cleanup

- [ ] Split pyproject installations to optional dependencies based on framework

