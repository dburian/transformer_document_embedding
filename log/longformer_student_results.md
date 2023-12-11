[cca_implementation]: cca_implementation.md
[i/diffs_sklearn_mine_cca]: imgs/diffs_sklearn_mine_cca.png
[contrastive_loss_with_embeddings]: contrastive_loss_with_embeddings.md
# LongformerStudent results

For cca implementation look at [log about CCA][cca_implementation].

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
    - normalized MSE with SBERT -- How close exactly are normalized embeddings to SBERT?
        - Since when training with cos, the norm of embeddings can be arbitrary
        large, it is worth watching MSE of normalized embeddings to reassure that
        by optimizing cosine we also optimize MSE of normalized embeddings.
    - COS with SBERT -- How similar are the embeddings to SBERT?
- Static part:
    - CCA of various dimensions -- How much the projected embedding correlate
      with projected DBOW embedding?

Since sometimes I will mask contextual loss because of length, there are these
additional metrics that monitor the masking and contextual metrics with the
masking considered:
- Mean contextual mask -- How often do we compare the embeddings to SBERT?
- MSE with SBERT only on non-masked inputs
- normalized MSE with SBERT only on non-masked inputs
- COS with SBERT only on non-masked inputs


## Questions

- Will frozen contextually-trained model with trainable projection be as good
  (in terms of metrics) as when trained together?
- Will [contrastive loss][contrastive_loss_with_embeddings] with embeddings
  bring better results than just mse/cos_dist?

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

### Combining contextual and static loss

Questions:
- Will static loss help to lower contextualization loss?
- Which combination of static and contextual loss yield the best results?
- How to achieve good

#### Problems

Same as with previous grid search, we seem to lower CCA during training to a
complete minimization of it.

- [x] try to compute sklearn cca with the trained model, see if it matches
    - [x] more exhaustive analysis of why and when sklearn differs from mine
    - [] compute sklearn cca every now and then... (every validation?)
- [x] hp search not propagating all params to config

#### Answers

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


## SentEvals

### Without any finetuning

```python
{
'SICKEntailment': {'acc': 77.69, 'devacc': 78.8, 'ndev': 500, 'ntest': 4927},
'STS12': {'MSRpar': {'nsamples': 750,
                    'pearson': PearsonRResult(statistic=0.11664723683824556, pvalue=0.0013737120789646784),
                    'spearman': SignificanceResult(statistic=0.20729644705629185, pvalue=1.0047142674517849e-08)},
        'MSRvid': {'nsamples': 750,
                    'pearson': PearsonRResult(statistic=0.07457026052617348, pvalue=0.041188486191284235),
                    'spearman': SignificanceResult(statistic=0.23779078715671953, pvalue=4.2197889829499694e-11)},
        'SMTeuroparl': {'nsamples': 459,
                        'pearson': PearsonRResult(statistic=0.27652948461077254, pvalue=1.6786196372717908e-09),
                        'spearman': SignificanceResult(statistic=0.41228688974735883, pvalue=2.90883420706498e-20)},
        'all': {'pearson': {'mean': 0.25126948873305077,
                            'wmean': 0.22535092319124045},
                'spearman': {'mean': 0.3632975393909439,
                            'wmean': 0.35046792501633656}},
        'surprise.OnWN': {'nsamples': 750,
                            'pearson': PearsonRResult(statistic=0.32877187636575994, pvalue=2.3077556143717178e-20),
                            'spearman': SignificanceResult(statistic=0.5228287386673285, pvalue=7.596914283156041e-54)},
        'surprise.SMTnews': {'nsamples': 399,
                            'pearson': PearsonRResult(statistic=0.4598285853243022, pvalue=2.8651567901698143e-22),
                            'spearman': SignificanceResult(statistic=0.43628483432702075, pvalue=5.696345320772476e-20)}},
'STS13': {'FNWN': {'nsamples': 189,
                'pearson': PearsonRResult(statistic=0.1357666264754312, pvalue=0.062499627180934984),
                'spearman': SignificanceResult(statistic=0.1494285202747337, pvalue=0.040149175382559556)},
        'OnWN': {'nsamples': 561,
                'pearson': PearsonRResult(statistic=-0.12983602291712515, pvalue=0.0020601821738974345),
                'spearman': SignificanceResult(statistic=0.02855262454223317, pvalue=0.4997325464454796)},
        'all': {'pearson': {'mean': 0.07154559358194719,
                            'wmean': 0.0729010109586673},
                'spearman': {'mean': 0.15983374785471577,
                            'wmean': 0.18026672450700187}},
        'headlines': {'nsamples': 750,
                        'pearson': PearsonRResult(statistic=0.2087061771875355, pvalue=7.937720985131654e-09),
                        'spearman': SignificanceResult(statistic=0.30152009874718044, pvalue=3.148553176051126e-17)}},
'STS14': {'OnWN': {'nsamples': 750,
                'pearson': PearsonRResult(statistic=-0.04715165127424858, pvalue=0.1970976014709609),
                'spearman': SignificanceResult(statistic=0.17330117008115492, pvalue=1.8033155885891656e-06)},
        'all': {'pearson': {'mean': 0.1154558397304342,
                            'wmean': 0.12536423622697473},
                'spearman': {'mean': 0.21935505987493864,
                            'wmean': 0.23594393235341812}},
        'deft-forum': {'nsamples': 450,
                        'pearson': PearsonRResult(statistic=-0.05770179289532586, pvalue=0.22183870381345747),
                        'spearman': SignificanceResult(statistic=-0.04046582263082787, pvalue=0.39179148917752904)},
        'deft-news': {'nsamples': 300,
                        'pearson': PearsonRResult(statistic=0.14832429067643652, pvalue=0.01009459059981554),
                        'spearman': SignificanceResult(statistic=0.25432837755812077, pvalue=8.185911244509942e-06)},
        'headlines': {'nsamples': 750,
                        'pearson': PearsonRResult(statistic=0.24926355472852263, pvalue=4.373127484130719e-12),
                        'spearman': SignificanceResult(statistic=0.3111418252598729, pvalue=2.6802169847879418e-18)},
        'images': {'nsamples': 750,
                    'pearson': PearsonRResult(statistic=0.18957560004941637, pvalue=1.68974391916119e-07),
                    'spearman': SignificanceResult(statistic=0.3698595571432963, pvalue=9.991662149383863e-26)},
        'tweet-news': {'nsamples': 750,
                        'pearson': PearsonRResult(statistic=0.2104250370978041, pvalue=5.941997242415464e-09),
                        'spearman': SignificanceResult(statistic=0.24796525183801496, pvalue=5.685041835384986e-12)}},
'STS15': {'all': {'pearson': {'mean': 0.207605450561821,
                            'wmean': 0.25014462380213576},
                'spearman': {'mean': 0.3003258040272945,
                            'wmean': 0.3511049101135229}},
        'answers-forums': {'nsamples': 375,
                            'pearson': PearsonRResult(statistic=0.056648224456546936, pvalue=0.2738651881312189),
                            'spearman': SignificanceResult(statistic=0.08270799037861247, pvalue=0.10981459681069837)},
        'answers-students': {'nsamples': 750,
                            'pearson': PearsonRResult(statistic=0.3691191404397579, pvalue=1.2688633021170766e-25),
                            'spearman': SignificanceResult(statistic=0.47604828590889886, pvalue=1.106072067153269e-43)},
        'belief': {'nsamples': 375,
                    'pearson': PearsonRResult(statistic=0.018249290744576915, pvalue=0.7246541305752427),
                    'spearman': SignificanceResult(statistic=0.11171076898614876, pvalue=0.03055490241215939)},
        'headlines': {'nsamples': 750,
                        'pearson': PearsonRResult(statistic=0.2875036258209556, pvalue=9.696933177628198e-16),
                        'spearman': SignificanceResult(statistic=0.3909664553191652, pvalue=8.429385256155668e-29)},
        'images': {'nsamples': 750,
                    'pearson': PearsonRResult(statistic=0.3065069713472676, pvalue=8.882199563050604e-18),
                    'spearman': SignificanceResult(statistic=0.440195519543647, pvalue=6.824147345471706e-37)}},
'STS16': {'all': {'pearson': {'mean': 0.29203417585870894,
                            'wmean': 0.30179663045910327},
                'spearman': {'mean': 0.42352522442367324,
                            'wmean': 0.43446842607952296}},
        'answer-answer': {'nsamples': 254,
                            'pearson': PearsonRResult(statistic=0.28034370926485974, pvalue=5.700118544900645e-06),
                            'spearman': SignificanceResult(statistic=0.42064641516808016, pvalue=2.5906734882564412e-12)},
        'headlines': {'nsamples': 249,
                        'pearson': PearsonRResult(statistic=0.371882932799205, pvalue=1.3840479041718705e-09),
                        'spearman': SignificanceResult(statistic=0.4895635988579728, pvalue=2.0585408955587531e-16)},
        'plagiarism': {'nsamples': 230,
                        'pearson': PearsonRResult(statistic=0.31748975611064595, pvalue=8.80904933620672e-07),
                        'spearman': SignificanceResult(statistic=0.49378219434576465, pvalue=1.5363984058422652e-15)},
        'postediting': {'nsamples': 244,
                        'pearson': PearsonRResult(statistic=0.5313434527124337, pvalue=3.523690368071605e-19),
                        'spearman': SignificanceResult(statistic=0.680418145412349, pvalue=1.6021510724274856e-34)},
        'question-question': {'nsamples': 209,
                                'pearson': PearsonRResult(statistic=-0.04088897159359977, pvalue=0.5566492850671636),
                                'spearman': SignificanceResult(statistic=0.03321576833419938, pvalue=0.6330440956262368)}},
'STSBenchmark': {'devpearson': 0.7455742398381552,
                'mse': 1.5225744064542188,
                'ndev': 1500,
                'ntest': 1379,
                'pearson': 0.6832664839031531,
                'spearman': 0.67259009671133,
                'yhat': array([1.47347094, 1.83003215, 2.48299285, ..., 3.91128775, 3.80678544,
    3.49845378])}
}
```
