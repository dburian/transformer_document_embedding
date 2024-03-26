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

- [Structural experiments](./student_structural_experiments.md)
- [Contextual experiments](./student_contextual_experiments.md)
- [Both structural and contextual experiments](./student_structural_contextual_experiments.md)


### Validation settings

These are now the go-to settings.
- results in about 3-4h.
- takes around 11GB of VMem

```yaml
model:
    kwargs:
        pooler_type: mean # Structural experiments showed `cls` had worse scores

train_pipeline:
    kwargs:
        grad accumulation steps: 1
        batch size: 6
        epochs: 1
        warmup steps: 250 # 10% of all batches
        learning rate: 1e-4
        learning rate scheduler: cos
        weight decay: 0.01
        max grad norm: 1.0
        fp16: True
        # We'll play with pooling/glb. attention in 'ablation' experiments
        global_attention_type: cls
        # Turns out consistent is quicker (6h40m vs 8h per training)
        dataloader_sampling: consistent
        metric_window_size_mult: 5
        # Increase for less frequent metrics and faster training
        metric_window_shift_frac: 0.1
        sampler_kwargs:
            bucket_limits:
                - 512
                - 1024
                - 1536
                - 2048
                - 2560
                - 3072
                - 3584
                - 4096

train split size: 15000
validation split size: 7680
```


## Evaluations

### First round of evaluation of Wikipedia similarities (noob round)

Results are in [wikipedia results log][wiki_similarities_results].

Observations:
- Optimizing cos with SBERT dramatically increases the scores. Even when used
  for all inputs, not just the ones with smaller length.
- Optimizing mse with SBERT worsens the results.
- Optimizing SoftCCA with DBOW worsens the results.
- As we measured CCA on the projections (which rose for training, a bit lowered
  for validation) we are not sure if CCA on outputs correlates with performance
  when only contextual model is used.
- Using both DBOW and SBERT improves the performance of the base model, but not
  as much as when using just SBERT. The most performing variant was when
  structural and contextual losses were somehow balanced.

## Plan

The plan is to:
- GS structural & write it down
- GS PV & write it down
- GS structural and contextual & write it down
- In case it would be interesting we can do ablation study on structural, i.e.
  having contextual only

The reason behind not doing first contextual on its own and then both of them at
the same time is that
- best contextual loss might not be the best when used together with structural
- we might find out that the whole thing doesn't work i.e. contextual&structural
  is worse then just structural, in that case it is better to find sooner rather
  then later

While we wait for PV we should do explaratory GS to find any low-hanging fruits
we will take advantage of when we have good PV.
