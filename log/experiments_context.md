# Experiments context

The settings and decisions that are common among all experiments mentioned in
the 'Experiments' chapter in the thesis.


### Training data

I use english split from [Wikipedia dataset from
HF](https://huggingface.co/datasets/wikipedia).

In total there is:
- about 6.3x10e9 docs
- which amounts to about 21GB, with my embeddings over 40GB

#### Do we use just wikipedia

Longformer's training set includes also other datasets, but we use only
Wikipedia. Why?
    - I already got it and did a number of experiments on it
    - I couldn't find the original Stories and book corpus datasets, only their
      variants
    - RealNews has 120GB and is downloaded from google plus we would still need
      to filter the data.
    - Wikipedia is already far too large. Estimated time for 1 epoch is 130h.
      Which is about twice what I could do.
    - On the other hand the variants could be probably easy to get (they are on
      HF).
    - **Since I'm behind schedule, I will use only Wikipedia, unless my
      supervisor tells me its a big problem.**

That's what I though until I realized the Wikipedia documents are breathtakingly
short (607 tokens). So I added [RealNews
dataset](https://github.com/rowanz/grover/tree/master/realnews) and following
Longformer filtered-out documents with below 1200 Longformer's tokens. Because I
didn't know what dataset is of higher quality I used a half-to-half mixture. I
don't know how much data I'm going to use so I decided just to randomly sample
both dataset to get enough data for validation (which is 500'000 train, 10'000
validation) and if I need more just do the sampling again.

While wikipedia doesn't have a validation split, Realnews has. So the validation
split is a result of sampling realnews val. split and wikipedia's train split,
where it is guaranteed the same document is not going to be sampled twice
(either twice in a single split or once in train and once in validation).

#### Resample or don't resample

We can resample the dataset so that each effective batch is sorted in lengths
and has the same distribution of lengths (More info about resampling at [dataset
sampling](./dataset_sampling_based_on_length.md)). The decision is whether we do
it or just shuffle the dataset.

At the end of the day, small resampling is just an implementation detail. So I
don't even have to mention it in my thesis. But it is faster to use is than to
not use it.

Also it doesn't make sense to resample beforehand because we still shuffle the
dataset which makes any any benefits of resampled datasets disappear. Also by
shuffling the dataset before creating the validation split, we already make the
length distribution quite even so resampling does very little.

### Evaluation

We use validation config (at `../experiments/evaluate_validation.yaml`), but use
only the classification results as the retrieval metrics don't have validation
split.

### Task selection

Currently we have the following tasks:

- Classification
    - Arxiv
    - AAN
    - OC
    - S2ORC
    - PAN
    - IMDB
- Retrieval
    - Wikipedia Wines
    - Wikipedia Games

I was considering of adding another for two reasons:
- only two retrieval tasks and
- only two classification tasks with high percentage of long documents

However, at the end I decided against adding new task (at least for validation),
because:

- the classification tasks cover the spectrum of lengths quite well
- there are no easily implementable (downloading, preprocessing needed e.g.
  wikipedia or Semantic Corpus) tasks from the related papers
- I want to be done with it.

For evaluation I might add another retrieval task (e.g. Relish from
ScirepEval).


#### Classification tasks

For classification tasks we care about accuracy. It measures how often was the
classifier correct, which is what we essentially want. For binary classification
tasks we use binary accuracy and for multiclass classification tasks we use
micro averaging.

To compare the accuracies between tasks we map the scale of values to 0-1
interval based on maximums and minimums that were reached for given task and
metric.
