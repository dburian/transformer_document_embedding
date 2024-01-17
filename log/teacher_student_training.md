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

## Grid searches

On Metacentrum I run:

- 8400 batches of 6 with 12 validation runs over 1280 batches for 10h 40min. So
  one batch takes ~4 secs.
- 5600 batches of 6 with 11 validation runs over 1280 batches for 5 - 8h. So one
  batch takes ~2.6 - 5.2 secs

Should probably try smaller datasets for grid searches.

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

> SoftCCA = lam * SDL1 + lam * SDL2 + L2 norm

- `soft_cca_lam`:
    - 0.03
    - 0.015
    - 0.005

#### Results

I did a grid search on `soft_cca_lam` while watching CCA on the final
projections. Since I wasn't logging CCA of the embeddings themselves, I cannot
tell anything about optimal values. What I can deduce that:

- increasing the weight of SDL to 0.03 effectively reduced the correlation (even
  for validation split) on the final projections by 25% compared to correlation
  for 0.015 or 0.005 (which were roughly equal). It should be noted that the
  correlation also wasn't logged properly so I'd have to repeat the experiment
  to make sure it is as so.

### Searching for optimal `soft_cca_sdl_alpha`

> The forgetting parameter in SDL: alpha * old_cov_mat + (1-alpha) * new_cov

- `soft_cca_sdl_alpha`:
    - 0.999
    - 0.99
    - 0.95
    - ... lower values seemed worse (had smaller CCA on final projection layers)

#### Results:

- 0.95 managed to lower SDL1 the most, had the best train and val CCA on final
  layers
- 0.99 had the lowest correlation on final projection layer (though by only 0.7%
  below 0.95, which was 7% above)
- CCA of outputs is unclear as the validation CCA was more or less the same, if
  we leave out occasional dips which happen to all cases

### SoftCCA projection layers

#### Observations

L2 and SDL losses:
- with few static layers the model has problems with L2
- bottleneck-like projections have higher with SDL
- more neurons with net shape like <|> (64x256x64) have smaller SDL
- with few projection layers the SDL is a bit higher, more layers give the
  projection net more oportunities to lower SDL. Though there might be caveat:
  more transformer layers had higher SDL1, but it might be only to a unfortunate
  initialization -- to many things could have gone bad.
- transformer layers have no effect on SDL2, static layers have effect on
  SDL1
- by using more beefy static network we significantly lower L2 loss -- for L2
  loss static projection seems to be the bottleneck. (Though we get some small
  decrease for increase of transformer projection net as well.)
- it seems that the beefier the nets the smaller L2
- even for SDL1 static projection seems to be the bottleneck and larger gains
  can be won there.

Metrics (if not mentioned it is a *train* metric) and gradients:
- Low L2 and SDL doesn't have to mean low train CCA on outputs
- More static layers meant lower train CCA on outputs, meanwhile this drop could
  be compensated with more transformer layers. Nevertheless it seems to be more
  effective to have small number of static layers than large number of
  transformer layers.
- [H] It might be that if the static net is more voluminous the majority of
  changes happens there to get better SDL and L2 and vice-versa: if the
  transformer net is more voluminous the majority of changes happens there. If
  the static net changes, there are no changes to the outputs however, so output
  CCA is the same or worse. However if the transformer changes, gradients
  propagate to the transformer itself, output changes and CCA rises.
    - comparison of gradients after accounting for the difference in loss:
        - for more voluminous transformer net: net1 gradients lower, net2
          gradients higher, lower transformer gradients
        - for more voluminous static net: net1 gradients higher, net2 gradients
          lower, higher transformer gradients
        - for more voluminous static net and small transformer net transformer
          gradients are very jumpy
    - so the gradients do not support the hypothesis, rather, they are against
      it -- if we suppose higher gradients == more changes
        - but it may be that higher gradients == larger jumps are necessary for
          slight lowering of the loss
        - so higher gradients show us "the bottlenecks" -- the places where
          there are less parameters which have to move more than if there were
          more parameters which can move just a bit for the combined function to
          move a lot
        - in that sense gradients tell us that if transformer net is voluminous
          static net is the bottleneck, if the static net is voluminous
          transformer and its net are the bottlenecks
    - comparison of minimal transformer net (`t_p_l=[768]`) and minimal static
      net (`s_p_l=[768]`):
        - hypothesis says that min. static net should be better -- more changes
          in transformer net and transformer itself
        - gradients say that min. transformer net should be better so it is the
          bottleneck and the gradients there are higher
        - cca says min. static net is better
        - losses say min. transformer net is better
- [H] Another hypothesis that would explain larger CCA for small static or large
  transformer nets is that by using larger static nets we give the model the
  opportunity to lower loss without changing outputs of the model which would
  increase the CCA
    - this seems to be supported by the facts that for fixed transformer layers:
        - CCA lowers as the static layers get more layers/neurons
        - losses lowers as the static layers get more layers/neurons
    - and the gradients also kind of support this: for fixed transformer layers
      the net1 and transformer gradients lower as the static net gets more
      neurons (however the loss also lowers so, the gradients may lower because
      loss does)
- higher cross-correlation but also higher SDL means lower CCA. Otherwise
  cross-correlations on final projeciton correspond +- to cross-correlation and
  cca on outputs.
- If we compare CCA on final projection vs on outputs, order does +- correspond
  but the tendencies do not. While the CCA on outputs rises, on projections it
  declines.

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
