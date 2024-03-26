# Tuning of Paragraph Vector

I need to find the best hyperparameters for PV. Since there is no loss logging
or any other metrics, we must resolve to evaluation on datasets. However, such
evaluation would be part of model selection (either of PV or the resulting
student transformer) and thus should use the 'validation' split or
cross-validation on the 'train' split.

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
    - `hp_searches/old_pv_dm_gs`
    - `hp_searches/old_pv_dm_big_gs`
    - `hp_searches/old_pv_dbow_gs`
    - `hp_searches/old_pv_dbow_big_gs`

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

Additionally I created checkpoints after 4 and 9 epoch to learn to better
estimate PV's convergence. (So 10 epochs in total)

All this was done twice. First with 150'000 docs in train and then with 450'000
docs in train ('big' variant). Again for my estimate what effects does enlarging
the corpus have on PV training.

## 18.3. GS with training on Wikipedia, RealNews

Relevant files:
- `hp_searches/pv_dm_gs`
- `hp_searches/pv_dbow_gs`

Hyperparameters:
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

I did only basic gridsearch since as I've learned in the previous model:
    - more data == better
    - more epochs (on average) == better

The training was done on the whole `val_corpus_500k`

# Evaluations

## 29.2. 2024 Evaluation of GS from 7.2. on Wikipedia corpus

Relevant files:
    - `evaluations/pv_dbow_gs_eval_correct`
    - `evaluations/pv_dm_gs_eval_correct`

### Permutation testing

#### About

Initially I didn't know where to look -- lots of models, lots of tasks and more
metrics per task. So I used permutation testing where models are faced against
each other for given task and metric and they collect 'wins'. I additionally
mapped the scale of each metric to 0-1 interval by collecting the reached
maximums and minimums for given each task, metric. The difference between such
normalized metrics of the two matched models was used as a 'won-by'.

#### Results

DBOW results:
- epoch 4 vs 9 -- How quick is the convergence?:
    - permutation testing -- each model against all the other versions of the
      same model trained on various amount of data (5/10 epochs, small/big train
      split)
        - on average as the number of training iterations grows the models
          become more successful
        - however its is more complicated.
            - different metrics behave differently, even the same metric behave
              differently across tasks
            - it is very messy even if we focus just on the top half of models
- Is it better to train more epochs or have larger training split?
    - based on the same permutation testing as above it is better to have fewer
      epochs and larger train split, since it results in more training iters
    - permutation testing between small train split but 10 epochs and large
      train split but 5 epochs confirms that
- large vs small training set -- Do they behave the same? (looking at epoch 9):
    - I did two separate permutation testing runs (one for small and one for big
      training set) and compared how often the same two models for given task
      and metric had same 'won'/'won_by' between the big and small training set.
        - ~83% of matches had the same result
        - in ~80% of cases the 'won_by' differed at max by 0.2
        - correlation of 'won_by' between runs is 0.91
    - The top 9 models are the same, but in different order.
        - However while large training split favours 1024, smaller train split
          favours 768
- the best models (looking at big training set, epoch 9):
    - Since there are some differences, we unfortunately have to resort to
      taking the results from the GS with large training split and use that size
      in future GS as well. The reasoning is that:
        - the results depend on size of the training split
        - we cannot afford to go any larger (we set an artificial limit)
        - therefore the results from the big GS are the closest to the reality
    - universally good/bad features:
        - min count 2 is better than 1%
        - stemming is worse than lowercasing and no pre-processing
    - best models (big training split):
        - lowercase, 1024, 2
        - None, 1024, 2
        - lowercase, 768, 2
        - None, 768, 2
        - stem, 1024, 2
        - stem, 768, 2
    - best presentable models (big training split, only cls):
        - lowercase, 1024, 2
        - None, 1024, 2
        - None, 768, 2
        - stem, 1024, 2
        - lowercase, 768, 2
        - lowercase, 100, 2
        - smaller vector sizes are better at retrieval type of tasks
    - sorted according to normalized accuracy score:
        - None, 1024, 2
        - lowercase, 1024, 2
        - stem, 1024, 2
        - None, 768, 2
        - lowercase, 768, 2
        - stem, 768, 2
- How the best model converged?
    - The convergence of the 6 best models is again chaotic. For most of the 6
      models the performance didn't agree with the number of training
      iterations.

DM results:
- epoch 4 vs 9 -- Is it ok to look at epoch 9?:
    - train. iters permutation testing and epoch perm. testing
    - for small model, as expected epoch 10 is better
    - for larger train split on average epoch 5 is better, but the best models
      are trained on 10 epochs except for one metric
- Is it better to train more epochs or have larger train split?
    - Iterations permutation testing
    - clearly it is better to train with larger training split, for less epochs
      (though again this results in more iterations than the opposite)
- large vs small training split -- Do they behave the same?
    - From training split permutation testing (two runs, one for each size of
      train split)
    - again ~80% of matches with same result, above 80% won-by difference under
      0.2 and high correlation between won-by (0.93)
    - the best 4 models are completely the same and in same order, top 6 and 9
      models are the same but different order
- the best models -- Which model are we going with?
    - permutation testing on long train split & epoch 10
    - on all tasks:
        - stem, 1024, 2
        - stem, 768, 2
        - lowercase, 1024, 2
        - lowercase, 768, 2
        - None, 1024, 2
        - None, 768, 2
        - stem, 100, 2
        - stem, 1024, 4500
    - on classification tasks:
        - dtto except on 2 last positions: first (stem, 1024, 4500), then (stem,
          100, 2)

Best DMs vs Best DBOWs:
- we take top 6 (on all tasks) from each and permutation testing together:
    - best were DMs:
        - dm, stem, 1024, 2
        - dm, lowercase, 1024, 2
        - dm, stem, 768, 2
        - dbow, lowercase, 1024, 2
        - dbow, None, 1024, 2
        - dm, lowercase, 768, 2
    - if look at each task type sepparately:
        - classif
- if we take top 6 just from classifications:
    - all tasks considered:
        - dm, stem, 1024, 2
        - dm, stem, 768, 2
        - dm, lowercase, 1024, 2
        - dbow, lowercase, 1024, 2
        - dbow, None, 1024, 2
        - dm, lowercase, 768, 2
    - just classification tasks:
        - dbow, lowercase, 1024, 2
        - dm, stem, 1024, 2
        - dbow, None, 1024, 2
        - dm, lowercase, 1024, 2
        - dm, stem, 768, 2
    - just retrieval tasks:
        - dbow, lowercase, 100, 2
        - dm, stem, 768, 2
        - dm, lowercase, 1024, 2
        - dm, lowercase, 768, 2 (very close to the previous)
        - dm, stem, 1024, 2 (very close to the previous)
        - dm, None, 768, 2 (very close to the previous)
- conclusions:
    - the architectures are pretty even, dms slightly better
    - dms are better for retrieval, dbows are better at classifications

### Just accuracy

When I thought about how to present the results it occurred to me that the
permutation results are problematic:

- they put same weight to precision, recall, f1, accuracy, yet f1 is aggregation
  of precision and recall so they are considered "twice"
- accuracy is actually what we want, we don't care about recall or precision

So I only looked at accuracy and ignored everything else. This doesn't mean the
above results are invalid though. They gave me the base understanding of what is
going on and allowed me to compare models across different kinds of tasks.

Also I added arxiv. To save time I used the above observations and just
evaluated the PVs trained on large training split for 10 epochs.

#### Results

DBOW:
- best presentable models:
    1. None, 1024, 2,
    2. lowercase, 1024, 2,
    3. stem, 1024, 2
    4. lowercase, 768, 2
    5. None, 768, 2
    6. stem, 768, 2
DM:
- best presentable models:
    1. stem, 1024, 2,
    2. stem, 768, 2
    3. lowercase, 1024, 2
    4. lowercase, 768, 2
    5. None, 768, 2
    6. None, 1024, 2
- pan is weird -- the best models are very bad at pan, the worst models are very
  good at it

- it seems that how beefy model is to some degree is a predictor of its
  performance for given classification task:
    - for arxiv (long document & lots of them) the best models do stemming, have
      large vector size and even have min_count > 2
    - for pan (small validation split & medium length documents) best models are
      have 100 vector size, preferably do as little preprocessing as possible
      and have min_count set to 2
    - however imdb is very similar to arxiv even though the validation split is
      small and the documents are shorter than for pan

Combination of DMs and DBOWs
- combine the best 3 presentable results of DMs and DBOWs (so 3x3=9 models) and
see who is better (with each PV as a baseline as well)


## 25.3. Evaluation of GS from 18.3. (Wiki + RealNews)

Relevant files:
    - `evaluations/pv_dm_gs_eval`
    - `evaluations/pv_dbow_gs_eval`

Context:
- only true classification tasks
- only 500k corpus and 10 epochs

Results:
- DBOW:
    - best models:
        - None, 1024, 2
        - lowercase, 1024, 2
        - stem, 1024, 2
        - lowercase, 768, 2
        - None, 768, 2
        - stem, 768, 2
        - lowercase, 1024, 5000
        - ...
    - 100d is the absolute worst, then 5000 min count
    - 100d is that bad because of arxive and imdb, on other tasks they are
      comparable to the best variants, on pan they are even (far) better
- DM:
    - best models:
        - stem, 1024, 2
        - stem, 768, 2
        - lowercase, 768, 2
        - lowercase, 1024, 2
        - None, 1024, 2
        - None, 768, 2
        - stem, 1024, 5000,
        - ...
    - very similar order: 100d worst, then 5000 min count
    - again 100d is bad because of arxive, imdb, but again pan is far better
      than the best versions
    - its more obvious than in DBOW that arxive benefits from 5000 min count

- Comparison of the best 3 models:
    - DBOW is clearly better overall
    - the architectures are comparable in aan, s2orc, oc
    - DBOW is better at imdb, pan
    - DM is better at arxiv
