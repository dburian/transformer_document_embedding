# Tuning of Paragraph Vector

Plan:
- with separate models GS vector size, min count and pre-processing method
- do it again because I'm dumbass
- see if combination of both models gets higher score
- Explore `dm_concat` on best dm
- GS negative sampling hyperparameters

## 31.1-7.2. Initial GS on individual tasks with separate models

Relevant files:
    - `hp_searches/pv_imdb_dm_gs`
    - `hp_searches/pv_imdb_dbow_gs`
    - `hp_searches/pv_oc_dm_gs`
    - `hp_searches/pv_oc_dbow_gs`
    - `hp_searches/pv_aan_dm_gs`
    - `hp_searches/pv_aan_dbow_gs`
    - `hp_searches/pv_pan_dm_gs`
    - `hp_searches/pv_pan_dbow_gs`
    - `hp_searches/pv_s2orc_dm_gs`
    - `hp_searches/pv_s2orc_dbow_gs`
    - `hp_searches/pv_wines_dm_gs`
    - `hp_searches/pv_wines_dbow_gs`
    - `hp_searches/pv_games_dm_gs`
    - `hp_searches/pv_games_dbow_gs`

All grid searches explored the combination of the following parameters:
- `vector_size`:
    - 100
    - 768
    - 1024
- `pre_process`:
    - stem
    - lowercase
    - no preprocessing
- `min_count`:
    - 2
    - 1% of unique documents in given dataset

### Results:

Influence of individual parameters:
- Even though `dbow` is better for classification tasks, `dm` is better for
  similarity tasks.
- 100 is the best `vector_size`, then 768, then 1024. The difference between
  last two is small. It seems the larger the classification dataset the smaller
  differences there are between the vector sizes.
- There wasn't a clear winner/loser for `pre_process`
- 2 is clearly better for `min_count`

Best overall (looking at median since we are interested in consistent
performance and ignore outliers):
- For classification tasks the overall best was (dbow, lowercase, 100, 2) right
  after its variation with stemming and no pre-processing.
- Then it starts to get unclear but the following variations are appearing
  often:
    - (dbow, None, 768, 2)
    - (dbow, lowercase, 768, 2)
    - (dbow, lowercase, 1024, 2)
- For similarity tasks the overall best was (dm, stem , 100, 2) with (dm,
  lowercase, 100, 2) right after. For more precision oriented metrics like HR@10
  and MRR (dbow, None, 100, 1) was sometimes good, but otherwise trash. Also
  surprisingly (dm, stem, 100, 1%) achieved best MRR, with 3-4 place in other
  metrics, though not quite as good as previously mentioned variants.

Worst overall (again medians):
- For classification tasks it was (dm, lowercase/None, 768/1024, 1%) except for
  recall where the worst was (dm, lowercase/stem, 1024/768, 2)
- For similarity tasks it were (dbow, None/stem/lowercase, 1024/768, 1%).

Comparison with student's results:
- When comparing the performance of student whose teacher was PV and PV with the
  same setting we see that the student's performance does not correspond to the
  performance of PV. The possible reasons may be:
    - student training fails to pass the information from teacher to student
    - PV setting is dependent on training-dataset (PV for student was trained on
      Wikipedia corpus, PV for evaluation was trained on the given task). Most
      importantly the volume of training datasets dramatically differed.
    - The latter reason feels more probable since it is to be kind of expected.
      What we can do then is to train PV on Wikipedia corpus and use it on given
      tasks.

## 7.2.- GS with training on Wikipedia corpus, separate models

In this grid search I decided to do things as I should have had done before.
Train PV on the Wikipedia corpus, then do `infer_vector` on all tasks.

Relevant files:
    -

All grid searches explored the combination of the following parameters:
- `vector_size`:
    - 100
    - 768
    - 1024
- `pre_process`:
    - stem
    - lowercase
    - no preprocessing
- `min_count`:
    - 2
    - 1% of unique documents in given dataset
