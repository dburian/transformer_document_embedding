[d/approach]: doc/approach.md
[d/cord_dataset]: doc/cord_dataset.md
[d/msmarco_dataset]: doc/msmarco_dataset.md
[d/rcv1_dataset]: doc/rcv1_dataset.md
[d/relish_dataset]: doc/relish_dataset.md
[d/imdb_dataset]: doc/imdb_dataset.md
[wang_20]: http://proceedings.mlr.press/v119/wang20k.html
[jian_22]: https://arxiv.org/pdf/2209.09433.pdf
[mullenbach_18]: https://aclanthology.org/N18-1100.pdf
[codec]: https://github.com/grill-lab/CODEC
[dai_22]: https://arxiv.org/abs/2204.06683

[msmarco_v1paper]: https://arxiv.org/abs/1611.09268
[trec-robust04]: https://trec.nist.gov/data/t13_robust.html
[trec-robust05]: https://trec.nist.gov/data/t14_robust.html
[trec-car]: http://trec-car.cs.unh.edu/datareleases
[s2orc]: https://github.com/allenai/s2orc
[orcas]: https://microsoft.github.io/msmarco/ORCAS
[reuters]: https://www.kaggle.com/datasets/nltkdata/reuters
[cisi]: https://www.kaggle.com/datasets/dmaso01dsta/cisi-a-dataset-for-information-retrieval
[trec_wp]: https://trec.nist.gov/data/wapost/
[gutenberg]: https://www.gutenberg.org/ebooks/offline_catalogs.html#the-project-gutenberg-catalog-metadata-in-machine-readable-format
[oshumed]: https://huggingface.co/datasets/ohsumed
[scidocs]: https://github.com/allenai/scidocs
[beir]: https://github.com/beir-cellar/beir
[mldoc]: https://github.com/facebookresearch/MLDoc
[enwiki8]: https://huggingface.co/datasets/enwik8
[ecthr]: https://archive.org/details/ECtHR-NAACL2021
[hyperpartisan]: https://aclanthology.org/S19-2145/
[20news]: https://kdd.ics.uci.edu/databases/20newsgroups/20newsgroups.data.html
[ldd_github]: https://github.com/LiqunW/Long-document-dataset
[ldd_hd]: https://huggingface.co/datasets/ccdv/arxiv-classification
[ldd_paper]: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8675939
[deeppatent]: https://link.springer.com/article/10.1007/s11192-018-2905-5
[patentbert]: https://www.sciencedirect.com/science/article/abs/pii/S0172219019300742?via%3Dihub
[bigbird]: https://arxiv.org/abs/2007.14062
[lyle_19]: https://openreview.net/forum?id=ryxW804FPH
[yang_21]: https://ojs.aaai.org/index.php/AAAI/article/view/17673
[mbti]: https://www.kaggle.com/datasets/datasnaek/mbti-type
[pandora]: https://psy.takelab.fer.hr/datasets/all/pandora/
[mimic_homepage]: https://mimic.mit.edu
[lds_game]: https://zenodo.org/record/4812962
[lds_wine]: https://zenodo.org/record/4812960
[ginzburg_21]: https://arxiv.org/pdf/2106.01186.pdf
[dai_15]: https://arxiv.org/abs/1507.07998
[blog_auth_corpus]: https://www.aaai.org/Papers/Symposia/Spring/2006/SS-06-03/SS06-03-039.pdf
[baa_tf]: https://www.tensorflow.org/datasets/community_catalog/huggingface/blog_authorship_corpus
[baa_hf]: https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus
[baa_kagle]: https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus
[zhou_gitio]: https://xuhuizhou.github.io/Multilevel-Text-Alignment
[zhou_20]: https://aclanthology.org/2020.emnlp-main.407/

# Datasets

My quick research into suitable datasets for long document classification and
similarity/retrieval tasks.

## The perfect dataset

Following up on [the idea of the model][d/approach], the ultimate tests would
be:

- get similar articles

Properly identify the kind of document from multiple views.

- how this article is different from the rest, how it deviates.

I do not yet know how this would be measured. (The whether it deviates is easy
-- the Hyperpartisan dataset). The how it deviates is a problem.

- Get documents which are relevant to this one. Not similar, but somehow
  connected to the given document. This would require a special IR dataset.


### Classification

The goal with classification evaluation is to show our model is able to
differentiate documents across diverse set of attributes. There are different
"spaces" for sentiments, content, ...

#### Chosen

Get an idea how our model compares on shorter inputs:

- [IMDB][d/imdb_dataset] - avg. length of 300 wordpieces, sentiment
  classification

Proper longer inputs(before I realized Reuters articles are rather small):

- [RCV1][d/rcv1_dataset] - Reuters English news collection, 103 classes divided
  to 3 views of the data

- [Long document dataset (Github)][ldd_github] - 33k Arxiv articles classified
  into 11 topic classes; used by [BigBird][bigbird]; [dataset paper][ldd_paper],
  available on [Huggingface][ldd_hd]. Previous SOTA from [Adapting pretrained
  language models for long document classification][lyle_19], now BigBird with
  92.31 F1 micro-averaged score.
- [Blog Authorship corpus][blog_auth_corpus] - 600k of posts from 19k bloggers,
  each post is on average 207 words with label blogger id, age, gender, industry
  and astrological sign. There are on average 35 posts and 7250 words per
  blogger. Could be interesting to see the resulting embedding space for this
  one. Available on [tensorflow datasets][baa_tf], [huggingface][baa_hf] and
  [kaggle][baa_kagle].

#### Alternatives

Sorted according to relevance and good fit. Best are at the top.

- [MLDoc][mldoc] - multilingual Reuters news collection with uniform
  distribution accross languages, for the official original dataset an agreement
  is needed (in a TREC track)
- [Hyperpartisan][hyperpartisan] - binary classification, only ~600 documents
  long enough (which is about 54% of the dataset). Used by [BigBird][bigbird].
  Could be helpful but even BigBird does the experiment 5x and averages the
  results to produce meaningful number.
- [ECtHR][ecthr] - court hearings mapping allegations to articles that were
  allegedly violated
- [MIMIC-III/IV][mimic_homepage] - multi label classification, avg. length of
  710 tokens, but in [example usage][mullenbach_18] data were preprocessed to an
  average of 1485 tokens. License needed. Used in [Transformer for Long Document
  Classifications][dai_22]. Maybe the next best thing for classification. Maybe
  because I cannot get to the data because of licensing.
- [20news][20news] - 20 classes, only 20k docs of unknown length
- [Reuters-21578][reuters] - 90 classes, skewed, 10k docs, usually smaller size
- [OSHUMED][oshumed] - titles and abstracts of medical articles


### Document similarity

The goal with document similarity tasks is to show our model maps similar
documents close to each other in the feature space.

#### Chosen

High quality documents, domain specific dataset.

- [RELISH][d/relish_dataset] - another scientific article similarity benchmark
  annotated by experts, 3K/2K seed articles in total/eval set, each seed article
  compared to 60 other articles. I may need to send an email to get the data.


#### Alternatives

Sorted according to relevance and good fit. Best are at the top.

- [Document similarity triplets data][sim_triplets_wiki_arxiv] - Dataset of
  arxiv/Wikipedia links triplets, where the first two links are more similar to
  each other than to the third link. The dataset is described in [Paragraph
  vectors][dai_15]. There are:
    - 20k of Wikipedia links, where similarity is based on the category of the
      article
    - 172 hand crafted Wikipedia triplets and
    - 20k of Arxiv triplets, where similarity is based on shared subjects.


### Document retrieval

In IR setting we are testing whether the embedding captures the main thought of
the article.

#### Chosen

Low quality documents, many domains. Not really ideal since we get one query
question and corresponding document(s). It'd be better if query would be another
document. I also find an interesting (not sure how valid)
[article](https://www.kdnuggets.com/2020/04/ms-marco-evaluate-semantic-search.html)

- [MS MARCO][d/msmarco_dataset] - document retrieval based on one-sentence
  queries, used in TREC Deeplearning 2019, 2020, 2021, 2022. In contrast to ORCA
  was built by linking query to documents which contained answers to the given
  query. The documents were picked by humans. There is v1 version and a bigger
  v2 version. Paper for [v1][msmarco_v1paper]. 480 queries, 12M docs, 13K qrels.

TODO: alternative wanted...

#### Alternatives

- [TREC COVID-19][d/cord_dataset] - TREC track for CORD dataset, 50 topics
  (queries) each with ranked list of relevant articles, the average text length
  is 1.7k words, abstract length is 140 words. Available on `ir_datasets`.
- [S2ORC][s2orc] - scientific articles in a citation graph with full parsed text
  across multiple fields of science
- [Orcas][orcas] - click-based document-ranking dataset
- [CODEC][codec] - 42 queries (+relevant queries = 380 queries total) and 730K
  docs from CommonCrawl, with TREC-style qrels. Topics are related to economy,
  history and politics. Each topic has a narrative.
- [Istella22][istella22] - dataset combining query-document retrieval based on
  text and on feature vectors. Human assessed 5-grade qrels. Each document has
  `text` and `text_extra` fields, for each the median number of tokens is around
  250. Accessed through `ir_datasets`. May prove to be a good dataset, not sure
  about the length though.
- [TREC Wahington post][trec_wp] - news articles, license needed
- [BEIR][beir] - comprehensive IR benchmark on 18 datasets, not sure about the
  length of documents - they presented SBERT as an example

### Citation prediction and plagiarism detection

In citation prediction we are given a section and an article and we should
decide if the article is referenced by the section or not.

In plagiarism detection we are again given a pair of documents and we should
decide whether the source document was "copied" from target document or not.

Also not sure how document embeddings fit in all of this...

- [A Benchmark for Document Relation Prediction and Localization][zhou_gitio]
  introduced in [Multilevel Text Alignment with Cross-Document
  Attention][zhou_20]. The documents are rather small though (122, 190, 263,
  1569 words per pair).

### Generative task

It may be interesting what we could get from the embedding when used as an input
into a language model. Would the text resemble the original? What ideas, topics,
forms would be stored?


### Personality detection - could be interesting, but not researched in detail

- [MBTI dataset][mbti] and [Pandora dataset][pandora], both used in
  [Multi-document transformer for personality deteciton][yang_21]


### Study the embedding itself

- [Uniformity and Alignment][wang_20] - also discussed in [Jian 22][jian_22]

### Unlaballed documents

- [enwiki8][enwiki8] - The enwik8 dataset is the first 100,000,000 (100M) bytes
  of the English Wikipedia XML dump on Mar. 3, 2006 and is typically used to
  measure a model's ability to compress data.
- [Gutenberg project books][gutenberg] - books with metadata


## Found but unfit

- TREC blog and TREC terabyte look interesting, though I could not find links to
  the datasets or more information on the data
- TREC web - ClueWeb12(09 version too) dataset (filtered and preprocessed
  crawl), too large (in terabytes of data) and paid
- KILT from Meta AI - different tasks to what I need
- CSFCube - dataset for evaluation of IR methods based on faceted Query by
  example task in scientific paper. Basically an example scientific paper is
  given and the task it to find similar scientific papers based on {background,
  objective, method, result, other} part of the query document.
- [TREC Robust 04][trec-robust04] - 250 keyword-queries over 520K news articles
- [TREC Robust 05][trec-robust05] - 50 hard queries from Robust 04 over 1M docs
  from AQUAINT - license is needed and data must be paid for
- There is [wine dataset][lds_wine] and [video game dataset][lds_game] that were
  introduced by [Self-Supervised Document Similarity][ginzburg_21]. The datasets
  are parsed Wikipedia articles. However I could not find the annotations, the
  ground truth. Both datasets are pretty small with ~90 source articles, each
  with ~10 similar articles.
- [SciDocs][scidocs] - scientific article similarity benchmark based on
  citations, "may be overly optimistic", not from a single domain although 70%
  are Comp. Science, tasks include citations prediction, classification by
  topic, recommendation; Yet the similarity between articles should be assessed
  based on the abstract and title **only**.  Interesting thread in
  here: [not so good of a
  benchmark?](https://github.com/allenai/scidocs/issues/23). "Easy benchmark to
  beat."
- [USPTO 2M][uspto_2m] - patent classification dataset containing 2M patents
  classified to 663 classes. The dataset was introduced by [DeepPatent
  model][deeppatent], [PatentBert][patentbert] and finally by
  [BigBird][bigbird]. Unfortunately it seems the dataset is no longer
  available. Was at [](http://mleg.cse.sc.edu/DeepPatent/).
- [TREC CAR][trec-car] - mainly paragraph ordering task to explain query, short
  paragraphs
- [CISI][cisi] - short documents, questions as queries

