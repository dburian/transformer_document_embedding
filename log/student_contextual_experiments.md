# Student contextual experiments

Experiments focused on contextual part of [teacher-student
training](./teacher_student_training.md).

## Final GS plan

How are we going to present the results?

### The shortest option

1. say we tried CCA and DCCA but the training just diverged (optionally try
   again to see what exactly is happening)
2. say we use normalizations and dropouts by default
3. grid search projections dimensions with SoftCCA
4. optionally ablate SDL part of SoftCCA (which results into using just MSE)

Chosen since we also need to grid search weighting of the two losses which might
be also complicated.

### More types of losses

1. say we tried CCA and DCCA but the training just diverged (optionally try
   again to see what exactly is happening)
2. say we use normalizations and dropouts by default
3. grid search projections dimensions with SoftCCA
3. grid search projections dimensions with MSE
3. grid search projections dimensions with COS?

### Projection in detail

1. say we tried CCA and DCCA but the training just diverged (optionally try
   again to see what exactly is happening)
2. grid search projections dimensions with SoftCCA
3. grid search dropouts and normalizations with SoftCCA
4. optionally ablate SDL part of SoftCCA (which results into using just MSE)


## Hyperparameter tuning

### Earlier experiments

These I unfortunately did not log. But I remember there were a lot of [problems
with CCA metric](./cca_metric_problems.md). Once I started using CCA-Zoo it was
somehow better.

### 12.12. contextual losses

Relevant files: `hp_searches/transformer_student_only_static`

Goals:
- Find out how model reacts to each loss.
- Find out how grad. acc. steps influences training.

Hyperparameters searches:
- loss:
    - `mse`
    - `cos_dist`
    - `soft_cca`
    - `contrastive_mse`
    - `contrastive_cos_dist`
- grad. acc. steps:
    - 4
    - 64

Results:
- Grad. acc. steps only slowed down the training.
- CCA was measured on the last projection layer, not at the outputs.
- Since CCA decreased for all training splits and stayed the same for all
  validation splits, I decided to focus on SoftCCA to try to improve it.
- There was something weird about contrastive losses. Their loss was wavy, they
  dramatically decreased correlation on the last projection layer. This was
  (presumably) after contrastive losses were fixed, so new gs should be done.

### 12.12. Balancing SDL and L2

Relevant files: `hp_searches/transformer_student_soft_cca`

Goal:
- Increase CCA by balancing SDL and L2

Hyperparameters:
- `soft_cca_lam` (back then it used to multiply L2 loss)
    - 5
    - 10
    - 15
    - 30
    - 60

Bugs:
- SDL had minor bug that caused incorrect loss right after validation split.
- CCA was measured on the last projection layer, not at the outputs.

Results:
- Again all versions decreased train CCA and validation CCA remained the same.

### 12.12. Soft CCA with increased projections' correlations

Relevant files: `hp_searches/transformer_student_soft_cca_test`

Goal:
- Find out if how much SDL helps CCA.

Hyperparameters searched:
- two models:
    - with SoftCCA as L2 + SDL1 + SDL2
    - with SoftCCA as L2 - SDL1 - SDL2 (the negative one)

Bugs:
- SDL had minor bug that caused incorrect loss right after validation split.
- CCA was measured on the last projection layer, not at the outputs.

Results:
- No pressure from SDL increased cross correlation of final projection layers.
- It was interesting to see that in the initial parts of training SDL can be an
  obstacle, but in long term the two versions were pretty much the same.
- **Due to bugs no conclusions.**

### 14.12. Soft CCA with positive SDL

Relevant files: `hp_searches/positive_sdl`

Goals:
- Find out if CCA starts to increase if we increase the training time.

Context:
- Just normal Soft CCA trained for 8400 steps.

Bugs:
- SDL had minor bug that caused incorrect loss right after validation split.
- CCA was measured on the last projection layer, not at the outputs.

Results:
- CCA still decreased, even for validation.

### 3.1. Initial search for SDL alpha

Relevant files: `hp_searches/sdl_alpha`

Goals:
- Find out if we can increase CCA by playing with SDL alpha parameter.

Context:
- SDL alpha parameter controls forgetting of covariance matrices. The larger
  the value the less SDL forgot the old covariance matrix.

Hyperparameters:
- `soft_cca_sdl_alpha`:
    - 0.1
    - 0.25
    - 0.5
    - 0.75
    - 0.9
    - 0.95
    - 0.99

Bugs:
- SDL had minor bug that caused incorrect loss right after validation split.
- CCA was measured on the last projection layer, not at the outputs.

Results:
- The higher the value the better.
- Nevertheless only 0.9, 0.95 and 0.99 variants increased train CCA and no
  variant increased validation CCA.
- While differences in train CCA were large, in validation all versions were
  pretty close to each other. This suggests that the model cannot generalize
  well.

### 9.1. Soft CCA lambda -- Balancing SDL and L2

Relevant files: `hp_searches/soft_cca_lam`

Goal:
- Find out the ideal weighting of SDL and L2.
- Find out if weighting has any effect on CCA on final projections.

Context:
- Soft CCA = L2 + lambda * SDL1 + lambda * SDL2

Hyperparameters:
- `soft_cca_lam`:
    - 0.005
    - 0.015 (balanced L2 and SDL)
    - 0.03

Bugs:
- SDL had minor bug that caused incorrect loss right after validation split.
- CCA was measured on the last projection layer, not at the outputs.

Results:
- L2 loss seemed unaffected. Due to bug in SDL, SDL was unreadable.
- No dramatic affect on CCA though 0.015 was the best and 0.005 the worst.
- The more weight on SDL the smaller cross-correlation on final layer was.

### 9.1. SDL alpha with CCA on outputs measured

Relevant files: `hp_searches/sdl_alpha_new`

Goal:
- See what SDL alpha is ideal for CCA on *outputs*.

Context:
- SDL alpha parameter controls forgetting of covariance matrices. The larger
  the value the less SDL forgot the old covariance matrix.

Hyperparameters:
- `soft_cca_sdl_alpha`:
    - 0.95
    - 0.99
    - 0.999

Results:
- Validation CCA on outputs was unreadable. It jumped up and down with no
  tendency whatsoever.
- Train CCA on outputs followed train and validation CCAs on final projection
  layers.
- The best alpha was 0.95 (both for training and validation).
- **For the first time validation CCA increased on final projection.** The only
  thing that changed between this and the previous experiment was that DeepNet
  did not start with activation (ReLU). This makes sense as it essentially
  discarded every number in embedding below 0.

### 9.1. First Soft CCA projections gs

Relevant files: `hp_searches/student_transformer_projection_gs`

Goals:
- Find out the ideal projection net for both transformer and PV.
- Find out how much the projection influences CCA.
- Find out how shapes of the net influence CCA.

Hyperparameters (one-search):
- transformer projection:
     - 768
     - 128(relu)x768
     - 768(relu)x1024(relu)x768
- PV projection:
    - 768
    - 64(relu)x512(relu)x768
    - 256(relu)x1024(relu)x768

Results:
- Validation CCA on outputs still unreadable.
- Low L2 and SDL doesn't mean high train CCA on outputs. Hypothesis is that this
  hold even for validation CCA.
- More beefy projection net means lower losses.
- Hypothesis is that by increasing the size of transformer projection while
  decreasing the size of PV projection we force the model to make the majority
  of changes on the transformer side, which can back-propagate to outputs.
- Bottleneck-like architecture tend to have higher SDL.
- CCA on outputs doesn't have to have the same tendency as those on the final
  projection layers, though in this case they agreed on order (as does
  validation CCA on final projection layers).
- It was evident that changes on PV's projection layers do influence transformer
  projections whereas the opposite wasn't true.

### 23.1. Soft CCA projection gs on DBOW 100d

Relevant files: `hp_searches/soft_cca_projections_dbow_100`

Goal:
- Confirm the previous findings on projections -- large transformer projection
  combined with small PV projection.
- Try out LoRA-like nets with no activations in between two FF layers.
- Map what is happening in terms of cross and normal correlation throughout the
  projection layers.

Hyperparameters (combinations of):
- transformer projection:
    - 256(relu)x768
    - 768(relu)x1024(relu)x768
- PV projection:
    - 32(linear)x768
    - 128(linear)x768

Results:
- Validation CCA is now readable even though I don't know what so dramatic
  changed.
- Also this experiment seems to contradict many previous findings:
    - the main factor in optimizing SDL1 was transformer projection, not PV
      projection, though still PV projection plays far more important role in
      SDL1 than transformer projection in SDL2
    - large transformer projection combined with small PV projection is the
      *worst* not the best variant
- The best CCA was achieved by small transformer and bigger PV (though smaller
  PV was just behind)
- LoRA-like nets need more experiments to show when they are helpful and when
  they are not. *I suspect this is the thing that threw off the experiment.*
- Interpreting what goes on throughout the projection seems impossible (e.g.
  different windows) looking at the various correlations, cross correlations and
  CCAs.
- Cross-correlation == L2 as well as other relationships such as (L2 -
  Correlation == CCA or Correlation == SDL) do **not** hold.
- This experiment is very confusing and I need to come back once I've done more
  experiments like this (perhaps repeat it with some older **not as extreme**
  configurations).

### 23.1. Soft CCA projection gs on DBOW 768d

Relevant files: `hp_searches/soft_cca_projections_dbow_768`

Goal:
- Confirm previous hypotheses on larger embeddings.
- Find out if there are any differences between low-dimensional PV embeddings
  and high-dimensional embeddings.

Hyperparameters (grid search):
- `transformer_projection`:
    - 256(relu)x768
    - 768(relu)x1024(relu)x768
- `contextual_projection`:
    - []
    - 128(linear)x768

Results:
- This experiment seems to support hypotheses from [first projection
  experiment](#9.1.-first-soft-cca-projections-gs):
    - best CCA: large transformer, no PV projection
    - in the middle, very close to each other: small transformer and no PV
      projection, large transformer and bigger PV projection (the latter would
      probably get worse)
    - worst CCA: small transformer and bigger PV projeciton
    - Bottleneck-like architecture truly have higher SDL.
- SDL2 is lowest for no projections (even when considering the previous
  experiment). This suggests that the projections really increase correlations.
- Hypothesis: with larger DBOW embeddings it seems easier to have low
  correlation on the side of PV. Even if the architecture is bottlenecked it is
  still better than smaller DBOW dim.

### 26.2. Soft CCA projection gs on DBOW 100d

Not as extreme as the experiment from
[23.1.](#231-soft-cca-projection-gs-on-dbow-100d).

Relevant files: `hp_searches/soft_cca_projections_dbow_100`

Hyperparameters (combinations of):
- `transformer_projection`:
    - 256(relu)x768
    - 768(relu)x1024(relu)x768
- `contextual_projection`:
    - 100(linear)x768
    - 100(relu)x768
    - 256(relu)x768
    - 768

Results (comparing with results from 23.1. 100 DBOW gs):
- So the best option was small transformer and just 768 on DBOW, followed by
  large transformer, 768 DBOW.
    - then we have three very similar versions with small transformer:
      128(linear), 32(linear), 100(linear)
    - then the same with large transformer except 32(linear) was noticeably worse
- The hurdle of the previous experiment was that the weekest net pushed DBOW
  through bottleneck. Which apparently is not very good. What I cannot explain
  yet why big projection for larger DBOWs is good, but here it hurts.
- Big transformer projection hurts 768 PV projection same as
  100(linear),128(linear), 32(linear). For larger PV projections the effect
  flips -- small projection hurts.
- Interestingly 100(linear) vs 100(relu) are very different in CCA. With
  relu is one of the worst options -- second highest correlation (after
  32(linear)), bad cross-correlations, one of the best in L2, second worst in
  SDL2.
- 100(relu)x... is closer to 256(relu)x... than 100(linear)x...
- SDL2 $\approx$ corr projection[-1] breadth, doesn't hold for SDL1 and corr
  projection[-1] transformer
- Cross-correlation doesn't behave as L2 nor as CCA, also holds for L2 and CCA

### 15.3. 768 contextual projection for 768d DBOW

Relevant files: `hp_searches/soft_cca_projections_dbow_768`

Hyperparameters:
- Student projection:
    - 256(relu)x768
    - 768(relu)x1024(relu)x768
- PV projection:
    - 768

Results:
- Validation CCA jumps a lot, but the CCA is clearly worse than for the previous
  768d dbow projections


### 19.3. Smaller student projections with 768d DBOW

Relevant files: `hp_searches/soft_cca_projections_dbow_768`

Hyperparameters:
- student projection:
    - 768
    - 256(linear)x768

Results:
- Validation CCA is all over the place, but


### 15.- 25.3. Validation unreadable experiments

I had trouble replicating the consistency of validation CCA of experiments from
[23.1. with 100d dbow](#231-soft-cca-projection-gs-on-dbow-100d). I've done some
experiments to check what influences the validation CCA.

Relevant files:
    - `hp_searches/soft_cca_projections_dbow_100_gas` -- playing with grad.
      accumulation steps
    - `hp_searches/soft_cca_projections_dbow_100_sampling` -- testing online
      sampling
    - `hp_searches/soft_cca_projections_dbow_100_warmup` -- playing with warmup
      steps
    - `results/teacher_embedding:TeacherEmbedding/transformer:TransformerEmbedder/replicating_previous_run*`
      -- replicating the very same setup from the grid search mentioned above

Results:
- no real surprises -- validation CCA still oscillated by quite a bit and nothing
  seemed to helped it settle by a lot
- the replicating experiments showed that there the result is highly dependent
  on the random seed which shuffled (fix coming soon) the validation split, and
  the windowed CCA ignored the first inputs that didn't fit into the window
    - now we do sliding window mean so the shuffling shouldn't matter and CCA
      should be more trustworthy and consistent
- the replicating experiments had the same training CCA but different validation
  CCA, which may suggest that we might be better off overlooking generalization
  and just decide base on train CCA

## Evaluations

### 10.1. Some random models chosen

- wrong CCA was measured and the models were picked chaotically

### 26.2. With DBOW 100, 768 on validation evaluation

Relevant files: `evaluations/student_eval_correct`


Evaluated models:
- all from grid searches from [23.1.](#231-soft-cca-projection-gs-on-dbow-100d)
  till [15.3](#153-768-contextual-projection-for-768d-dbow)

Results:
- permutation testing (adjusted won-by on all tasks including non-validation task):
    - 100 DBOW (best to slightly worse):
        - 768(relu)x1024(relu)x768 Transformer, 768 PV
        - 256(relu)x768 Transformer,            768 PV
        - Longformer
        - 256(relu)x768 Transformer,            100(linear)x768 PV
        - 256(relu)x768 Transformer,            128(linear)x768 PV
        - 256(relu)x768 Transformer,            32(linear)x768 PV
        - dbow_100d
        - ...
    - 768 DBOW (best to worst)
        - Longformer
        - dbow_768d
        - 256(relu)x768 Transformer,            - PV
        - 768(relu)x1024(relu)x768 Transformer, - PV
        - 768(relu)x1024(relu)x768 Transformer, 128(linear)x768 PV
        - 768(relu)x1024(relu)x768 Transformer, 768 PV
        - 256(relu)x768 Transformer,            768 PV
        - 256(relu)x768 Transformer,            128(linear)x768 PV

- higher CCA means higher probability of performant model
    - there are several unexplicable reordering going from CCA to permutation
      testing results
    - CCA certainly doesn't correlate with performance, small differences in CCA
      can easily result in large or differences in performance
- Classification and regression metrics definitely value different things
    - for 100d only 768 PV variants are better then pure Longformer in
      classification tasks
    - for 768d, large T, - PV is somehow much better in retrieval than small T,
      - PV, even though their classificaiton performance is on par
- dbow_768d is slightly worse in permutation testing won-by than dbow_100d even
  though it won more matches
    - this small difference cannot explain why with 100d dbow contextual
      teacher, the student beat the teacher (and longformer) while 768d dbow
      couldn't do that
    - Seemingly teaching with more dimensions is harder
- 100d dbow is better in retrieval while 768d is better at classification

### 20.3. Smaller student projections for 768d DBOW on validation evaluation

Relevant files:
    - `evaluations/student_eval_correct`

Evaluation of [20.3.
GS](#203-smaller-student-projections-for-768d-dbow-on-validation-evaluation)

Results:
- I only evaluated the true validation tasks (only classification).
- The best models according to normalized accuracy versions were (all versions
  from `student_eval_correct` considered):
    - 768 Transformer,           - PV
    - Longformer
    - 256(linear)x768,           - PV
    - dbow_768d,                 - PV
    - 256(relu)x768,             - PV
    - 768(relu)x1024x(relu)x768, - PV
    - clearly making the Transformer projection smaller helped a lot
