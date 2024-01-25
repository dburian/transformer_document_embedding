[wiki_similarities_results]: wiki_similarities_results.md
[windowed_vs_normal_metrics]: windowed_vs_normal_metrics.md
[i/diffs_sklearn_mine_cca]: imgs/diffs_sklearn_mine_cca.png

# Teacher-student training logs

## Metrics

To validate the progress of training I decided to use metrics only and avoid
using tasks. Reasons:

- tasks should serve as 'test' split -- for reporting results; otherwise we
  would sacrifice some data/task and would need to look for another when
  comparing our model to others
- task's performance does not tell us a lot about the model -- especially why
  the model achieved such a score

The metrics which I'm going to watch will be:
- Contextual part:
    - MSE with SBERT -- How close exactly are the embeddings to SBERT?
    - normalized MSE with SBERT -- How close exactly are normalized embeddings
      to SBERT?
        - Since when training with cos, the norm of embeddings can be arbitrary
          large, it is worth watching MSE of normalized embeddings to reassure
          that by optimizing cosine we also optimize MSE of normalized
          embeddings.
    - COS with SBERT -- How similar are the embeddings to SBERT?
- Static part:
    - CCA of various dimensions -- How much the projected
      embedding correlate with projected DBOW embedding?

Since sometimes I will mask contextual loss because of length, there are these
additional metrics that monitor the masking and contextual metrics with the
masking considered:
- Mean contextual mask -- How often do we compare the embeddings to SBERT?
- MSE with SBERT only on non-masked inputs
- normalized MSE with SBERT only on non-masked inputs
- COS with SBERT only on non-masked inputs

### CCA metrics

- the most decisive metric should be measured on **model's outputs** not on the
  final projection layer
  - it holds:
    - good cca on previous layers -> probably good cca on next layers
    - good cca on next layers -> chance of having good cca on previous layers
- typically lower CCA dimensions than the dimension of the embedding reach full
  correlation (1 for each CCA dimension)
- lower CCA dimensions are closer to 0 entropy, everything is smoothed out,
  whereas CCA with more dimensions are more sensitive
- CCA with smaller [windows][windowed_vs_normal_metrics] are more sensitive to
  local (in terms of time of training) phenomena
- the most reliable is CCA measured on all dimensions of the embedding with
large window (I typically use 10x the dimension of CCA)
- to understand local phenomena use smaller window (typically i use 5x dim of
  CCA)

## Experiments done

Each aspect of the training has a separate file.

- [Depth experiments](./student_depth_experiments.md)
- [Breadth experiments](./student_breadth_experiments.md)
- [Both depth and breadth experiments](./student_depth_breadth_experiments.md)


### Batch sizes, grad. acc. steps, lr, lr scheduler

On Metacentrum I run:

- 8400 batches of 6 with 12 validation runs over 1280 batches for 10h 40min. So
  one batch takes ~4 secs.
- 5600 batches of 6 with 11 validation runs over 1280 batches for 5 - 8h. So one
  batch takes ~2.6 - 5.2 secs

Should probably try smaller datasets for grid searches.

- 6 batch size == 11.6GB VMem for only depth or only breadth

## Evaluations

### First round of evaluation of Wikipedia similarities

Results are in [wikipedia results log][wiki_similarities_results].

Observations:
- Optimizing cos with SBERT dramatically increases the scores. Even when used
  for all inputs, not just the ones with smaller length.
- Optimizing mse with SBERT worsens the results.
- Optimizing SoftCCA with DBOW worsens the results.
- As we measured CCA on the projections (which rose for training, a bit lowered
  for validation) we are not sure if CCA on outputs correlates with performance
  when only breadth model is used.
- Using both DBOW and SBERT improves the performance of the base model, but not
  as much as when using just SBERT. The most performing variant was when depth
  and breadth losses were somehow balanced.
