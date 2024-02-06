# Student breadth experiments

Experiments focused on breadth part of [teacher-student
training](./teacher_student_training.md).

## Hyperparameter tuning

### Earlier experiments

These I unfortunately did not log. But I remember there were a lot of [problems
with CCA metric](./cca_metric_problems.md). Once I started using CCA-Zoo it was
somehow better.

### 12.12. Breadth losses

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
- `breadth_projection`:
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

## Evaluations

### 10.1. Some random models chosen

- wrong CCA was measured and the models were picked chaotically

### 26.1. With DBOW 100, 768 on Wikipedia similarities

Scores: [wikipedia similarities
results](./wiki_similarities_results.md#second-evaluation-round).

Evaluated models:
- with `dbow_30e_100d` and `dbow_30e_768d`
- the models were picked from projection gs on [100
  DBOW](#231-soft-cca-projection-gs-on-dbow-100d) and on [768
  DBOW](#231-soft-cca-projection-gs-on-dbow-768d).
    - I picked the best, second best and worst CCA for each dimensionality

Results:
- more often than not CCA ranking equals ranking in wikipedia similarities
- CCA probably doesn't correlate with wikipedia sim. scores as models with very
  close CCA behave as differently as the best and worst CCA models
- sometimes it there is inexplicable success of models with bad CCA
    - this success was in precision-like metric, but the model did badly in
      longer recall-like metric (HR@100) which together with high in-embedding
      correlation may suggest that this caused embeddings to have low variance
      (many dimensions seem dependent on others) which caused in a dataset, with
      higher positive/negative answers to a query, to have high precision
- higher dimensional PV embeddings may not be as good as lower dimensional
- good news is that we surpassed vanilla longformer on all but one metric (MRR)
