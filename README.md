[wang_20]: http://proceedings.mlr.press/v119/wang20k.html
[awesome_ir]: https://github.com/harpribot/awesome-information-retrieval
[google_doc_topic]: https://docs.google.com/document/d/13Yb34eyklpX6bGzaf3m0jlsFb8rF10KvLXh4DuY4SD0/edit#heading=h.k2zhq4p261n
[sbert]: https://arxiv.org/abs/1908.10084
[longformer]: https://arxiv.org/pdf/2004.05150v2.pdf
[reformer]: https://arxiv.org/pdf/2001.04451.pdf
[jian_22]: https://arxiv.org/pdf/2209.09433.pdf
[dai_22]: https://arxiv.org/pdf/2204.06683.pdf
[xiong_21]: https://arxiv.org/pdf/2112.07210.pdf
[mullenbach_18]: https://aclanthology.org/N18-1100.pdf
[luo_21]: https://arxiv.org/pdf/2103.14542.pdf
[awesome_ds]: https://github.com/malteos/awesome-document-similarity
[medic_22]: https://arxiv.org/pdf/2209.05452.pdf
[relish_article]: https://academic.oup.com/database/article/doi/10.1093/database/baz085/5608006?login=false
[xiong_20]: https://arxiv.org/abs/2007.00808

[msmarco_v1paper]: https://arxiv.org/abs/1611.09268
[msmarco]: https://microsoft.github.io/msmarco/
[trec-robust04]: https://trec.nist.gov/data/t13_robust.html
[trec-robust05]: https://trec.nist.gov/data/t14_robust.html
[trec-car]: http://trec-car.cs.unh.edu/datareleases
[trec-cord]: https://ir.nist.gov/trec-covid/data.html
[s2orc]: https://github.com/allenai/s2orc
[cord]: https://github.com/allenai/cord19
[orcas]: https://microsoft.github.io/msmarco/ORCAS
[rcv1]: https://jmlr.csail.mit.edu/papers/volume5/lewis04a/
[reuters]: https://www.kaggle.com/datasets/nltkdata/reuters
[ppdb]: http://paraphrase.org/#/download
[cisi]: https://www.kaggle.com/datasets/dmaso01dsta/cisi-a-dataset-for-information-retrieval
[trec_wp]: https://trec.nist.gov/data/wapost/
[gutenberg]: https://www.gutenberg.org/ebooks/offline_catalogs.html#the-project-gutenberg-catalog-metadata-in-machine-readable-format
[oshumed]: https://huggingface.co/datasets/ohsumed
[scidocs]: https://github.com/allenai/scidocs
[relish]: https://figshare.com/projects/RELISH-DB/60095
[beir]: https://github.com/beir-cellar/beir
[mldoc]: https://github.com/facebookresearch/MLDoc
[enwiki8]: https://huggingface.co/datasets/enwik8
[ecthr]: https://archive.org/details/ECtHR-NAACL2021
[imdb]: https://aclanthology.org/P11-1015
[hyperpartisan]: https://aclanthology.org/S19-2145/


# Diploma thesis

## Links

- [shared info w/ supervisor][google_doc_topic]

## Topic

In short combine [SBERT][sbert] and either [Longformer][longformer] or
[Reformer][reformer]. The goal is to create contextual embedding of the entire
document.

## Finetuning datsets

- unsupervised datset (Wikipedia) with contrastive loss

Worse than high-quality NLI(Natural Language Inference) datasets.


## Evaluation tasks

### Chosen

#### Classifiaction

The goal with classification evaluation is to show our model is able to
differentiate documents across diverse set of attributes. There are different
"spaces" for sentiments, content, ...

Get an idea how our model compares on shorter inputs:

- [IMDB][imdb] - avg. length of 300 wordpieces, sentiment classification
- [Hyperpartisan][hyperpartisan] - binary classification, only ~600 documents
  long enough

Proper longer inputs:

- [RCV1][rcv1] - Reuters English news collection, 103 classes divided to 3 views
  of the data

Alternativelly for multilingual classification:

- [MLDoc][mldoc] - multilingual Reuters news collection with uniform
  distribution accross languages, for the official original dataset an agreement
  is needed (in a TREC track)


#### Document similarity/retrieval

The goal with document retrieval tasks is to show our model maps similar
documents close to each other in the feature space.

High quality documents, domain specific dataset.

- [RELISH][relish] - another scientific article similarity benchmark annotated
  by experts, [article here][relish_article], 3K/2K seed articles in total/eval
  set, each seed article compared to 60 other articles. I may need to send an
  email to get the data.

Low quality documents, many domains.

- [MS MARCO][msmarco] - document retrieval based on one-sentence queries, used
  in TREC Deeplearning 2019, 2020, 2021, 2022. In contrast to ORCA was built by
  linking query to documents which contained answers to the given query. The
  documents were picked by humans. There is v1 version and a bigger v2 version.
  Paper for [v1][msmarco_v1paper]. 480 queries, 12M docs, 13K qrels.

#### Study the embedding itself

- [Uniformity and Alignment][wang_20] - also discussed in [Jian 22][jian_22]


### Looked through

Classification

- MIMIC-III - multilabel classification, avg. length of 710 tokens, see more at
  [example usage][mullenbach_18] used by BigBird
- 20news - 20 classes, only 20k docs of unknown length
- [Reuters-21578][reuters] - 90 classes, skewed, 10k docs, usually smaller size
- [OSHUMED][oshumed] - titles and abstracts of medical articles
- [ECtHR][ecthr] - court hearings mapping allegations to articles that were
  allegedly violated

Text similarity

- [PPDB][ppdb] - paraphrases, possibly too short, though I have not find any
  example
- [SciDocs][scidocs] - scientific article similarity benchmark based on
  citations, "may be overly optimistic", not from a single domain although 70%
  are Comp. Science

Document Retrieval

- [TREC COVID-19][trec-cord] - TREC track for CORD dataset, 50 topics (queries)
  each with ranked list of relevant articles
- [CISI][cisi] - short documents, questions as queries
- [BEIR][beir] - comprehensive IR benchmark on 18 datasets, not sure about the
  length of documents - they presented SBERT as an example
- [Orcas][orcas] - click-based document-ranking dataset
- [S2ORC][s2orc] - scientific articles in a citation graph with full parsed text
  across multiple fields of science
- [TREC CAR][trec-car] - mainly paragraph ordering task to explain query, short
  paragraphs
- [TREC Robust 04][trec-robust04] - 250 keyword-queries over 520K news articles
- [TREC Robust 05][trec-robust05] - 50 hard queries from Robust 04 over 1M docs
  from AQUAINT - license may be needed
- [CODEC][codec] - 42 queries (+relevant queries = 380 queries total) and 730K
  docs from CommonCrawl, with TREC-style qrels. Topics are related to economy,
  history and politics. Each topic has a narrative.
- [TREC Wahington post][trec_wp] - news articles, no goal attribute, license
  needed

Unlaballed documents

- [enwiki8][enwiki8] - The enwik8 dataset is the first 100,000,000 (100M) bytes
  of the English Wikipedia XML dump on Mar. 3, 2006 and is typically used to
  measure a model's ability to compress data.
- [Gutenberg project books][gutenberg] - books with metadata
- [CORD-19][cord] - collection of COVID articles - no metadata to evaluate on

#### Found but unfit

- TREC blog and TREC terabyte look interesting, though I could not find links to
  the datasets or more information on the data
- KILT from Meta AI - different tasks to what I need


## Sources

- [SBERT][sbert]
- [Longformer][longformer]
- [Reformer][reformer]

- [Contrastive learning of Sent. embed. using non-linguistic
  modalities][jian_22]

- current state-of-the-art sentence embeddings:

Tianyu Gao, Xingcheng Yao, and Danqi Chen. SimCSE: Simple contrastive learning
of sentence embeddings. In Empirical Methods in Natural Language Processing
(EMNLP), 2021.

- [comparison of transformers for long documents][dai_22]
- [comparison of attention types for longer documents][xiong_21]
- [Unsupervised Document Embedding via Contrastive Augmentation - 21 with
  doc2vecC as backbone][luo_21]

- [Awesome document similarity site][awesome_ds] covers all from methodology,
  models to benchmarks
- [Awesome information retrieval][awesome_ir]


- [testing article encoders for recommendation][medic_22]
- [Contrative learning for Dense Representations][xiong_20]

## Interesting snippets

- [Jian 2022][jian_22]:

> "we show that Transformer models can generalize better by learning a similar
> task (i.e., clustering) with multi-task losses using non-parallel examples
> from different modalities."



