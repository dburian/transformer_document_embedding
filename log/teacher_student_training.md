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

## Grid searches

Each training should run like 40min, validation every 10min for 5mins. This
translates to limits:
- train: 5600 batches, batch_size = 3
- validation: every 1400 batches, batch_size = 3, 1400 batches


### Just contextual loss

Questions:
- Will optimizing cos help mse and vice versa?
- Will more gradient accumulation steps help the model with robustness?
- Will optimizing on long inputs bring the loss up?
- Will contextualization loss help to lower static loss?

Different masking:
    - only small inputs should output loss (`contextual_max_len = 384`)
    - all inputs output loss (`contextual_max_len = inf`)

I need two datasets for this in order for the two to be comparable --
incompatible grid searches.

Each gs:
- loss:
    - mse
    - cos
- grad_accumulation_steps:
    - 8
    - 32

#### Results

- Optimizing mse also optimizes cos, not the other way around.
    - Minimizing cos_dist lead to lower normalized mse and but higher mse, which
      only started to lower after 1.5k steps (for 8 grad. acc. steps).
    - Minimizing mse lead to slightly lower normalized mse and slightly lower
      cos.
    - Optimization of cos must initially lead to the norm of embedding to grow.
      Only after some time this effect seems to lessen and turnaround causing
      the mse to also lower.
- Larger grad accumulation steps do not seem to improve the result.
    - Overall the training was severely slower, due to the diminishing learning
      rate, stagnated before reaching the performance of less grad. acc. steps.
    - This suggests that there is obviously more capacity to the network (since
      the model with lower grad. acc. steps did not have time to overfit).
- Validation loss when optimizing cos was much closer to training loss.
    - This could suggest either the model was more overfitted or that optimizing
      cos is more robust.
    - Since the val loss did not climb above train loss I'd guess the latter is
      the more probable explanation.
    - Another (also probable) explanation is that the range of cos_dist is much
      smaller than of mse. So the val loss appears to be closer to the train
      loss, when in fact it is relatively +- the same distance as with mse. (mse
      was - 0.1-0.5, cos was - 0.001-0.02)
- Learning embeddings for longer context than SBERT was able to handle did not
  seem to make it harder for the model to learn.
    - When optimizing cos, the differences between train losses were
      indistinguishable, while for val they got closer as the training progressed
      (the differences were quite pronounced) with longer context with higher
      losses.
    - For mse, the training loss for longer contexts was lower, yet the
      validation lass was higher. This almost suggests that the model was more
      overfitted when training with longer contexts.

### Searching for optimal `soft_cca_lam`



### Only static losses

- loss type:
    - mse
    - cos_dist
    - soft_cca
- grad. acc. steps:
    - 8
    - 32

- Non-limitting projection layers and cca dimension
- No extra bells and whistles
- No layer or norm projection normalization

#### Results

- SoftCCA has the worst CCA scores. Best are produced with contrastive types of
  losses, particularly for cos and smaller grad. accum. steps.
- The model has difficulty with optimizing contrastive loss. In some sections of
  training it gets smaller, but

### Combining contextual and static loss

Questions:
- Will static loss help to lower contextualization loss?
- Which combination of static and contextual loss yield the best results?
- How to achieve good

Answers:

- Static loss has no effect on contextualization loss -- no magic here.
- Large grad. accum. step (64, batch size 6) lead to overfitting of sdl1 loss

### Start fine-tuning

Down the road, planned questions...

Questions:
- What is the most efficient projection to optimize for given loss?
- Are the projections same for different losses?
- Does some combination of static and contextual loss perform better in terms of metrics than another?
- How does the distribution of lengths influence performances on contextual and
  on static loss?

Projection net:
- projection layers:
    - as minimal as possible -- small output
    - bottle neck -- medium output
    - tube -- medium output
    - beefy -- large output
- static loss type:
    - mse
    - cos

Losses:
- static loss type:
    - mse
    - cos
- contextual loss type:
    - mse
    - cos

Length distribution:
- sampling of dataset
    - original distribution
    - more towards shorter sentences
    - more towards longer sentences

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
