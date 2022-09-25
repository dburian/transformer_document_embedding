[wang_20]: http://proceedings.mlr.press/v119/wang20k.html
[google_doc_topic]: https://docs.google.com/document/d/13Yb34eyklpX6bGzaf3m0jlsFb8rF10KvLXh4DuY4SD0/edit#heading=h.k2zhq4p261n
[sbert]: https://arxiv.org/pdf/1908.10084.pdf
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

[rcv1]: https://jmlr.csail.mit.edu/papers/volume5/lewis04a/
[reuters]: https://www.kaggle.com/datasets/nltkdata/reuters
[ppdb]: http://paraphrase.org/#/download
[cisi]: https://www.kaggle.com/datasets/dmaso01dsta/cisi-a-dataset-for-information-retrieval
[trec_wp]: https://trec.nist.gov/data/wapost/
[gutenberg]: https://www.gutenberg.org/ebooks/offline_catalogs.html#the-project-gutenberg-catalog-metadata-in-machine-readable-format
[oshumend]: https://huggingface.co/datasets/ohsumed
[scidocs]: https://github.com/allenai/scidocs
[relish]: https://figshare.com/projects/RELISH-DB/60095
[beir]: https://github.com/beir-cellar/beir
[mldoc]: https://github.com/facebookresearch/MLDoc
[enwiki8]: https://huggingface.co/datasets/enwik8
[encthr]: https://archive.org/details/ECtHR-NAACL2021
[imdb]: https://aclanthology.org/P11-1015.pdf
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

The problem is to find at least one dataset(task) on which I could show my model
embeds the documents well.

Ideas:

- dataset used in evaluations of Longformer - no
- dataset used in evaluations of BigBird
- dataset used in doc2vec, ... - small datasets?
- document retrieval tasks
- document classification tasks - not really what I want to teach the model
- SIGIR
- TREC - just pain to navigate that site

Thoughts:

What I really want to teach the model is similarity between documents. This 

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

TODO: What here?

#### Study the embedding itself

- [Uniformity and Alignment][wang_20] - also discussed in [Jian 22][jian_22]


### Looked through

#### Classification

- MIMIC-III - multilabel classification, avg. length of 710 tokens, see more at
  [example usage][mullenbach_18] used by BigBird
- 20news - 20 classes, only 20k docs of unknown length
- [Reuters-21578][reuters] - 90 classes, skewed, 10k docs, usually smaller size
- [OSHUMED][oshumed] - titles and abstracts of medical articles
- [ECtHR][ecthr] - court hearings mapping allegations to articles that were
  allegedly violated

#### Text similarity

- [PPDB][ppdb] - paraphrases, possibly too short, though I have not find any
  example
- [SciDocs][scidocs] - scientific article similarity benchmark based on
  citations, "may be overly optimistic", not from a single domain although 70%
  are Comp. Science
- [RELISH][relish] - another scientific article similarity benchmark annotated
  by experts, [article here][relish_article]

#### Information Retrieval

- [CISI][cisi] - short documents, questions as queries
- [BEIR][beir] - comprehensive IR benchmark on 18 datasets, not sure about the
  length of documents - they presented SBERT as an example


#### Unlaballed documents

- [enwiki8][enwiki8] - The enwik8 dataset is the first 100,000,000 (100M) bytes
  of the English Wikipedia XML dump on Mar. 3, 2006 and is typically used to
  measure a model's ability to compress data.
- [TREC Wahington post][trec_wp] - news articles, no goal attribute
- [Gutenberg project books][gutenberg] - books with metadata


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


- [testing article encoders for recommendation][medic_22]

## Interesting snippets

- [Jian 2022][jian_22]:

> "we show that Transformer models can generalize better by learning a similar
> task (i.e., clustering) with multi-task losses using non-parallel examples
> from different modalities."



