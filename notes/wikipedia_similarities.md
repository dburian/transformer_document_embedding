[paper]: https://arxiv.org/pdf/2106.01186.pdf
[metrics_implementation]: https://github.com/microsoft/SDR/blob/main/models/reco/wiki_recos_eval/eval_metrics.py

# Wikipedia similarities

Introduced by [Ginzburg et al. (SDR paper)][paper]. This dataset has two
sub-datasets for wines and games.

## Metrics for comparison with baseline

For each sample (an article) we are given two lists (gold and predicted) of
recommendations ordered according to similarity to the sample. We denote single
such gold recommendation as $g$, pred recommendation as $p$.

I implemented the metrics as close to the [original
code][metrics_implementation] as possible. Caveats:

- ranks are indices
- max rank is the count of predicted recommendations

### Mean reciprocal rank (MRR)

Defined as mean of reciprocal ranks for each sample. When computing reciprocal
rank for a sample we

- take the is the smallest rank (defined by predicted recommendations) of a gold
  recommendation
- compute its multiplicative inverse (1/x)

> The bigger MRR the more precise the model is.

### Mean percentile rank (MPR)

Defined as mean of all (for each recommendation of each sample) percentile
ranks. For each gold recommendation $g$ we define percentile rank as the rank
the model gave to $g$ divided by number of samples in the dataset.

> The smaller MPR the better the recall the model has.

### Hit rate at k (HR@k)

HR@k is defined as the percentage of gold recommendations in the top k predicted
recommendations. Note that the percentage is computed as part of the possible
gold recommendations, not part of the top k predictions. E.g. we have 4 out of 6
possible gold recommendations in the top 10 predictions. So HR@10 would be
$4/6 * 100\%$

We consider mean over all samples.


## Wines
### Description

Articles from Wikipedia parsed and split to sections. Each article has a title,
each section has a title and parsed text. The articles are accompanied by
expertly assessed similarities. There are about 1660 articles, 89 source
articles each with ~10 similar articles.

Upon closer look, the articles are sometimes **related only through external
knowledge**. E.g. Moët & Chandon winery is 'similar' to Chardonnay. When
looking at the text of Moët & Chandon on Wikipedia term 'Chardonnay' appears
only once.

I also found that in more than half of the considered source-target pairs the
target text does not contain any words from the source's title. It is true also
the other way around. **We should really try TF-IDF on this one.**
