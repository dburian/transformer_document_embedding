[d/relish_dataset]: doc/relish_dataset.md
[d/imdb_dataset]: doc/imdb_dataset.md
[wang_20]: http://proceedings.mlr.press/v119/wang20k.html
[jian_22]: https://arxiv.org/pdf/2209.09433.pdf
[mullenbach_18]: https://aclanthology.org/N18-1100.pdf
[codec]: https://github.com/grill-lab/CODEC

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
[beir]: https://github.com/beir-cellar/beir
[mldoc]: https://github.com/facebookresearch/MLDoc
[enwiki8]: https://huggingface.co/datasets/enwik8
[ecthr]: https://archive.org/details/ECtHR-NAACL2021
[hyperpartisan]: https://aclanthology.org/S19-2145/
[20news]: https://kdd.ics.uci.edu/databases/20newsgroups/20newsgroups.data.html

# Datasets

todo:
- BigBird - Arxiv, Patent, Hyperpartisan datasets
- STS and Spearman's correlation
- write personality classification
- that other from "Multi-Document Transformer for Personality Detection" -
  Pandora
- Istella22
- look at trec covid, scidocs, csfcube
- MIMIC-III

My quick research into suitable datasets for long document classification and
similiarity/retrieval tasks.

## Chosen

#### Classifiaction

The goal with classification evaluation is to show our model is able to
differentiate documents across diverse set of attributes. There are different
"spaces" for sentiments, content, ...

Get an idea how our model compares on shorter inputs:

- [IMDB][d/imdb_dataset] - avg. length of 300 wordpieces, sentiment classification

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

- [RELISH][d/relish_dataset] - another scientific article similarity benchmark annotated
  by experts, 3K/2K seed articles in total/eval
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
- [Hyperpartisan][hyperpartisan] - binary classification, only ~600 documents
  long enough
- [20news][20news] - 20 classes, only 20k docs of unknown length
- [Reuters-21578][reuters] - 90 classes, skewed, 10k docs, usually smaller size
- [OSHUMED][oshumed] - titles and abstracts of medical articles
- [ECtHR][ecthr] - court hearings mapping allegations to articles that were
  allegedly violated
- [Blog Authorship corpus][blog_auth_corpus] - 600k of posts from 19k bloggers,
  each post is on average 207 words with label blogger id, age, gender, idustry
  and astrological sign. There are on average 35 posts and 7250 words per
  blogger. Could be interesting to see the resulting embedding space for this
  one.


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


---

### Articles I've searched through

- Longformer,
- Reformer,
- 
