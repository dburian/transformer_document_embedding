# Student depth experiments

The depth part of [teacher-student training](./teacher_student_training.md).

## Hyperparameter tuning

### 12.12. First loss training

Relevant path: `hp_searches/transformer_student_only_contextual`

Goal:
- Find out the best loss for SBERT mse and cos.
- Find out if one loss is more general than the other (mse helps cos or cos
  helps mse).
- Find out if increased grad. acc. steps help the model with robustness.
- Find out if optimizing on longer inputs will be somehow harder.
- Find out if contrastive losses are more effective than their normal versions.

There were two grid searches one with `max_depth_length` set one without.
Each searched:
- loss:
    - `mse`
    - `cos_dist`
    - `contrastive_mse`
    - `contrastive_cos_dist`
- grad_accumulation_steps:
    - 8
    - 32

Results:
- Later I found out that I computed SBERT cos and mse incorrectly. So I'm
  leaving out any observations of that.
- Also my contrastive loss had errors so I'm leaving out that as well.
- Learning embeddings for longer context than SBERT was able to handle did not
  seem to make it harder for the model to learn.
- Larger grad accumulation steps do not seem to improve the result. The only
  effect was that training was slower.

### 12.12. Contrastive losses (after fix)

Relevant path: `hp_searches/contrastive_test`

Goals:
- Find out if fixing contrastive loss helped.
- Find out if contrastive losses are more effective than their normal versions.

Hyperparameters searched:
- loss:
    - `contrastive_mse`
    - `contrastive_cos_dist`

Results:
- SBERT cos and mse was still broken in this one, so no comments on that.
- `contrastive_mse` does not work. Presumably it is because very hard negatives,
  which push against the positive side of the loss. This causes the negative
  loss and the positive "to meet in the middle". Negative decreases a bit,
  positive increases a bit, then they stay at a balance.


### 23.1. Losses (with fixed SBERT cos and mse metrics)

Relevant path:
- `hp_searches/depth_loss` -- normal inputs
- `hp_searches/depth_loss_short` -- shortened inputs

Goals:
- Find out the best loss for SBERT mse and cos.
- Find out if one loss is more general than the other (mse helps cos or cos
  helps mse).
- Confirm that optimizing on longer inputs is not harder.
- Find out if contrastive losses are more effective than their normal versions.

Two same grid searches: one with `max_depth_length` set to the maximum context
length of the teacher (labeled as short), one without `max_depth_length`.

Hyperparameters of both searches:
- `loss_type`:
    - `mse`
    - `contrastive_mse`
    - `cos_dist`
    - `contrastive_cos_dist`

Context:
- `contrastive_lam` = 1

Results:
- the training on short inputs behaves almost exactly the same as the training
  on not shortened inputs in all losses/metrics
- For SBERT metrics the best loss is always corresponding loss.
- Optimizing `mse` seemed more general as all versions lowered SBERT cos, but
  only `mse` lowered SBERT mse during training (though in validation all
  versions except `contrastive_mse` lowered SBERT mse).
- `contrastive_mse` indeed doesn't work and the same behaviour as was described
  in [previous experiment](##1212-contrastive-losses-after-fix) was observed. It
  was even the worst version for SBERT mse.
- Contrastive losses aren't more effective. Even for SBERT cos, `cos_dist` was
  still better then its contrastive version. This could be because the negatives
  were always almost perpendicular, so it didn't exert useful pressure on loss.

## Evaluations

### 10.1. Losses (with incorrect cos and mse SBERT metrics) on Wikipedia Similarities

> Not really worth to record anything as the decisive metrics were broken.

### 26.1. Losses with short versions on Wikipedia Similarities

Evaluation of all models from [Loss grid
search](#231-losses-with-fixed-sbert-cos-and-mse-metrics). Scores are [wikipedia
similarities results.](./wiki_similarities_results.md#second-evaluation-round).

Variants (combinations of):
- `max_depth_length`:
    - 384
    - null
- `loss_type`:
    - `mse`
    - `contrastive_mse`
    - `cos_dist`
    - `contrastive_cos_dist`

Results:
- shortened inputs **always** worsen the scores
- `contrastive_cos_dist` is the best, `cos_dist` is right behind
- there are notable differences between the datasets:
    - in wines `mse` was the worst option, while in games it always competed for
    second best
    - `contrastive_mse` was close third in wines but in games it was the worst
      option by far
    - in games there is much more articles to choose from so there might be more
      easy negatives
    - `contrastive_cos_dist` was a bit worse in games than in wines (in wines
      clearly first, in games not as clearly)
    - so contrastive losses might be a bit finnicky when it comes to the tested
      dataset, and `cos_dist` might be the best option overall -- to test this
      I should do evaluation on *more datasets*
