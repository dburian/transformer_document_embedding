# Student structural experiments

The structural part of [teacher-student training](./teacher_student_training.md).

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

There were two grid searches one with `max_structural_length` set one without.
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

Two same grid searches: one with `max_structural_length` set to the maximum context
length of the teacher (labeled as short), one without `max_structural_length`.

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

### 5.3. Structural Validation experiments only with wikipedia

Relevant files:
- `hp_searches/old_structural_basic_loss`
- `hp_searches/old_structural_max_marginals_loss`

Hyperparameters:
- basic losses:
    - mse
    - contrastive
    - cos_dist
- max marginals:
    - losses:
        - cos_dist
        - mse
    - lambda:
        - 0.1
        - 0.5
        - 1
        - 1.5

Results:
- 1.5 lambda is bad in every case
- constrastive behaves a lot like `max_marginals_cos_dist` with lambda 1, but
  has slightly better SBERT cos
- in their respective metrics (e.g. max_marginals_cos_dist corresponding to
  SBERT COS) lambdas 0.1 and 0.5 behave very similarly
- for mse the best were:
    - mse
    - max_marginals mse, lambdas 0.1, 0.5
    - a large gap
    - constrastive
    - cos_dist
- for cos the best were:
    - max marginals cos_dist, lambdas 0.1, 0.5
    - cos_dist
    - a gap
    - contrastive
    - max marginals cos_dist lambda 1

### 11.3. Structural validation experiments with CLS pooling only with wikipedia

Relevant files:
- `hp_searches/cls_structural_basic_loss`
- `hp_searches/cls_structural_max_marginals_loss`

Hyperparameters:
- the same as for [GS
  above](#53-structural-validation-experiments-only-with-wikipedia) but with CLS
  pooling

Results:
- all observations made to previous GS with mean pooling still hold even with
  CLS pooling
- compared to mean pooling CLS pooling performed worse by a noticeable bit

### 18.3. Mean pooling, global attention, RealNews & Wiki

Relevant files:
- `hp_searches/structural_basic_loss`
- `hp_searches/structural_max_marginals_loss`

Hyperparameters:
- the same as for [GS
  above](#53-structural-validation-experiments-only-with-wikipedia) but with
  proper training dataset

Results:
- all observations from the mentioned GS hold except
- for mse the best were:
    - mse
    - max_marginals mse, lambdas 0.1, 0.5
    - a large gap
    - max_marginals cos_dist, lambdas 0.5, 0.1
    - max_marginals cos_dist, lambda 1
    - cos_dist
    - constrastive
- Both SBERT COS and MSE are noticeably lower than for the mentioned GS with
  just wikipedia

### 20.3. MEAN pooling, no global attention, RealNews & Wiki

Relevant files:
- `hp_searches/cls_structural_basic_loss`
- `hp_searches/cls_structural_max_marginals_loss`

Hyperparameters:
- the same as for [GS
  above](#53-structural-validation-experiments-only-with-wikipedia) but withith
  global attention set to `none`

Results:
- all observations from the mentioned GS hold except
- for mse the best were:
    - mse
    - max_marginals mse, lambdas 0.1, 0.5
    - a large gap
    - max_marginals cos_dist, lambdas 0.5, 0.1
    - cos_dist
    - constrastive
- without the global attention both SBERT MSE and COS were slightly worse, but
  still noticeably better than the mentioned grid search (which trained only on
  wikipedia)
- the avoidance of global attention did not reduce memory ;(, but speeded up the
  training by 7 mins from 50 to 43 mins


## Evaluations

### 10.1. Losses (with incorrect cos and mse SBERT metrics) on Wikipedia Similarities

> Not really worth to record anything as the decisive metrics were broken.

### 26.1. Losses with short versions on validation evaluation

Evaluation of all models from [Loss grid
search](#231-losses-with-fixed-sbert-cos-and-mse-metrics).

Relevant files: `evaluations/student_eval_correct`

Variants (combinations of):
- `max_structural_length`:
    - 384
    - null
- `loss_type`:
    - `mse`
    - `contrastive_mse`
    - `cos_dist`
    - `contrastive_cos_dist`

Results:
- how the models stack up?
    - permutation testing (wins):
        1. `contrastive_cos_dist`
        2. `cos_dist`
        3. short `contrastive_cos_dist`
        4. sbert
        5. short `cos_dist`
        6. ...
    - permutation testing (won by percentage of the task, metric range):
        1. `cos_dist`
        2. `contrastive_cos_dist`
        3. short `contrastive_cos_dist`
        4. short `cos_dist`
        5. `contrastive_mse`
        6. sbert
        7. ...
- to shorten or not to shorten:
    - when comparing the shorten and not shorten versions it is not clear if it
      is better to shorten or not
        - the difference in scores is around +-0.01 for classification, +-0.02
          for wikipedia similarities except MRR where they are around +0.03 and
          `contrastive_mse` 0.06
        - so the differences are very small, even negligible
        - if anything training with longer instances has better performance on
          retrieval tasks, but worse on classification tasks (except `mse`)
    - however from permutation testing we know the better models are those who
      do not shorten
- which loss type is the best?
    - `contrastive_cos_dist`
    - `cos_dist`
    - `contrastive_mse`
    - `mse`
- what are the differences in classification vs retrieval tasks?
    - this is important as the retrieval tasks do not have a validation split
      and we cannot include them in validation
    - sbert shines in retrieval tasks, and is on-par with `contrastive_cos_dist`
      and `cos_dist` in won matches and slightly better in 'won-by' (the
      difference of percentage ranges)
    - as noted above, short versions are very weak in retrieval tasks
    - in 'won-by', `cos_dist` is still the best but `contrastive_cos_dist` long
      is fourth, so if we want to choose `contrastive_cos_dist` it will be hard
      to defend
- its important to highlight that `contrastive_cos_dist` did not have cos SBERT
  as good as `cos_dist` yet it is clearly on par or better
    - this suggests that `cos` with SBERT may not be the decisive metric

### 13.3. Comparing cls and mean pooling of structural gs only with wikipedia


Relevant files:
- `evaluations/old_structural_eval` -- [mean
  GS](#53-structural-validation-experiments-only-with-wikipedia)
- `evaluations/cls_structural_eval` -- [cls
  GS](#113-structural-validation-experiments-with-cls-pooling-only-with-wikipedia)

Results:
- max-marginals cos_dist with lambda 1 is the best out of max marginals
- max-marginals mse was significantly worse overall
- best models (by normalized accuracy)
    - max marginals cos_dist lambda 1
    - cos_dist
    - contrastive
    - sbert
    - longformer
    - looking at just `value`: all first 4  models are within 0.01 of each other
      (cos_dist equal to sbert pretty much)
- Cls pooling is overall worse than mean pooling, especially all loss variants
  that contain mse except for max marginals with lambda 1 (which is for some
  reason the best model)
- if we look at the top models from CLS pooling and the best model from mean
  pooling we see that mean pooling is always better except for IMDB and S2ORC.

### 20.3. Mean pooling, glb attention, RealNews & Wiki

Relevant files:
- `evaluations/structural_eval` -- from [GS from
  18.3.](#183-mean-pooling-global-attention-realnews-wiki)

Results:
- for whatever reason the results dramatically changed from [previous
  evaluation](#133-comparing-cls-and-mean-pooling-of-structural-gs-only-with-wikipedia)
- best max-marginals:
    - mse with lambda 1
    - cos_dist with lambda 1.5
    - cos_dist with lambda 0.1
- best models overall:
    - max marginals mse with lambda 1
    - max marginals cos_dist with lambda 1.5
    - sbert
    - contrastive
    - max marginals cos_dist with lambda 0.1
    - cos_dist
    - max marginals cost_dist with lambda 1
- so for whatever reason max marginals mse with lambda 1 is suddenly good, even
  though other mse variants are the worst 3 models
- also cosine is suddenly worse than sbert (mm lambda 1, cos_dist, and
  contrastive)
- this raises a question whether to use mm mse lambda 1, even though the loss
  actually does something different than what our goal was
  - the decision is to continue with both variants (best basic loss = cos_dist
    and mm mse lam 1) and see how the situation will look once we add contextual
    teacher


### 25.3. Comparing cls glb attn. and no glb attn., MEAN pooling, RealNews & Wiki

Relevant files:
- `evaluations/structural_eval`
- `evaluations/glb_structural_eval` -- [GS from
  ](#203-mean-pooling-no-global-attention-realnews-wiki)

Results:
- just without glb attn:
    - best mm loss:
        - mse, lambda 1
        - cos_dist, lam 0.5
        - cos_dist, lam 1
    - best overall:
        - mm mse, lam 1
        - mm, cos_dist, lam 0.5
        - cos_dist
        - mm cos_dist, lam 1
        - contrastive,
        - mm cos_dist, lam 0.1
        - mm cos_dist, lam 1.5
        - sbert
        - longformer
    - again mm mse lambda 1 was clearly the best
    - cos_dist very close third
    - mm mse, lam 1 had very good score for pan, which launched it to the first
      place (the same didn't happen for glb attn.)

- comparing w/ and w/o glb attn.:
    - total order:
        - no glb., mm mse, lam 1
        - no glb., mm cos_dist, lam 0.5
        - no glb., cos_dist (very close)
        - no glb., mm cos_dist, lam 1
        - no glb., contrastive
        - no glb., mm cos_dist, lam 0.1
        - glb., mm mse, lam 1
        - no glb., mm cos_dist, lam 1.5
        - sbert
    - clearly no glb. is better significantly better
    - ergo we can avoid using global attention, and do the following experiments
      with:
        - no glb., mm mse, lam 1 -- best overall
        - no glb., cos_dist -- best basic loss
