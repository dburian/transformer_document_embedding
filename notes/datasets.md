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
[sim_triplets_wiki_arxiv]: http://cs.stanford.edu/~quocle/triplets-data.tar.gz

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
[wine_game_github_gt]: https://github.com/microsoft/SDR/tree/main/data/datasets
[jiang_19]: https://dl.acm.org/doi/pdf/10.1145/3308558.3313707
[yang_20]: https://dl.acm.org/doi/pdf/10.1145/3340531.3411908
[radev_13]: https://link.springer.com/article/10.1007/s10579-012-9211-2

# Datasets

My research into suitable datasets for testing long document embeddings.

Our focus is on **assessing similarity** on **longer documents**. Similarity
because thats where you can gain the most by having good document embeddings.
Longer documents because there are models which probably will produce something
better for shorter ones.

### Similarity

Testing whether the embeddings capture test similarity. Do not think about the
term 'similarity' too much, it loosely says the text have the same words,
structure, meaning and whatever else the embeddings capture.

#### Good fit

- [wine dataset][lds_wine] and [video game dataset][lds_game]
    - ground truths on [Github][wine_game_github_gt]
    - from [Self-Supervised Document Similarity][ginzburg_21]
    - parsed Wikipedia articles, with simililarity assessed by an expert
    - both are pretty small with ~90 source articles, each with ~10 similar
      articles

- PAN plagiarism detection 
    - introduced in [Multilevel Text Alignment with Cross-Document
      Attention][zhou_20]
    - downloadable from [A Benchmark for Document Relation Prediction and
      Localization][zhou_gitio]
    - pairs of section and document, binary classification whehter the section
      plagiarises the given document
    - positive pairs are assessed by human annotators,
    - negatives are constructed from positives, by changing the section to
      another section of the source document
    - 1569 words per pair.

- AAN citation recommendation
    - introduced in [Multilevel Text Alignment with Cross-Document
      Attention][zhou_20]
    - pairs of abstracts
    - positive pair is constructed given an abstract of source paper, and abstract of
paper cited by the source paper
    - available [on the article
      homepage](https://xuhuizhou.github.io/Multilevel-Text-Alignment/)

- OC citation recommendation
    - introduced in [Multilevel Text Alignment with Cross-Document
      Attention][zhou_20]
    - Again citations, similar to AAN but on Scholar Open Corpus
    - available [on the article
      homepage](https://xuhuizhou.github.io/Multilevel-Text-Alignment/)

- S2ORC citation recommendation
    - introduced in [Multilevel Text Alignment with Cross-Document
      Attention][zhou_20]
    - citations but on Semantic Scholar Open Corpus
    - available [on the article
      homepage](https://xuhuizhou.github.io/Multilevel-Text-Alignment/)

- Wiki65K
    - introduced in [Semantic Text Matching for Long-Form documents][jiang_19]
    - used again in [Siamese Multi-depth Transformer-based Hierarchical (SMITH)
      Encoder for Long-Form Document Matching][yang_20]
    - pairs of wikipedia articles which are either related or not
    - Related articles are those with Jackard similarity between outgoing links
      above 0.5.
    - available on [Google research
      github](https://github.com/google-research/google-research/tree/master/gwikimatch)

- AAN104K
    - introduced in [Semantic Text Matching for Long-Form documents][jiang_19]
    - adopts data from [The ACL anthology network corpus][radev_13] for *citation
      prediction*
    - used in [Siamese Multi-depth Transformer-based Hierarchical (SMITH)
      Encoder for Long-Form Document Matching][yang_20]
    - pairs of full paper texts
    - in a positive pair first text cites the second one
    - **unknown download**

- [Document similarity triplets data (download)][sim_triplets_wiki_arxiv]
    - Dataset of arxiv/Wikipedia links triplets, where the first two links are more
  similar to each other than to the third link
    - introduced in [Paragraph vectors][dai_15]
    - There are:
        - 20k of Wikipedia links, where similarity is based on the category of
          the article
        - 172 hand crafted Wikipedia triplets and
        - 20k of Arxiv triplets, where similarity is based on shared subjects.

#### Rejected

- [RELISH][d/relish_dataset]
    - scientific article similarity benchmark annotated by experts
    - 3K/2K seed articles in total/eval set
    - each seed article compared to 60 other articles
    - I may need to send an email to get the data
  - Reason of rejection: Only abstracts provided, difficult to get, may be too
    hard


### Classification

The goal with classification evaluation is to show our model is able to
differentiate documents across diverse set of attributes. There are different
"spaces" for sentiments, content, ...

#### Good fit


Get an idea how our model compares on shorter inputs:

- [IMDB][d/imdb_dataset] - avg. length of 300 wordpieces, sentiment
  classification
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
- [Hyperpartisan][hyperpartisan]
    - binary classification: hyperpartisan or not
    - labels by human annotators
    - only ~600 documents long enough (which is about 54% of the dataset)
    - used by [BigBird][bigbird]

#### Alternatives

- [ECtHR][ecthr]
    - court hearings mapping allegations to articles that were allegedly
      violated
- [20news][20news] - 20 classes, only 20k docs of unknown length

#### Rejected

- [Reuters-21578][reuters]
    - reuters articles,
    - 90 classes, 10k docs
    - Reason of rejection: smaller articles
- [OSHUMED][oshumed]
    - titles and abstracts of medical articles
    - Reason of rejection: smaller texts
- [MIMIC-III/IV][mimic_homepage]
    - multi label classification, avg. length of 710 tokens, but in [example
      usage][mullenbach_18] data were preprocessed to an average of 1485 tokens.
    - Used in [Transformer for Long Document Classifications][dai_22]
    - Reason of rejection: License needed
- [RCV1][d/rcv1_dataset]
    - Reuters English news collection, 103 classes divided to 3 views of the
      data
    - Reason of rejection: articles are too short
- [MLDoc][mldoc]
    - multilingual Reuters news collection
    - uniform distribution accross languages
    - for the official original dataset an agreement is needed (in a TREC track)
    - Reason of rejection: difficult to get


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

TODO: format this like above

Any document retrieval datasets (there are better models for that):
- [MS MARCO][d/msmarco_dataset] - document retrieval based on one-sentence
  queries, used in TREC Deeplearning 2019, 2020, 2021, 2022. In contrast to ORCA
  was built by linking query to documents which contained answers to the given
  query. The documents were picked by humans. There is v1 version and a bigger
  v2 version. Paper for [v1][msmarco_v1paper]. 480 queries, 12M docs, 13K qrels.
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


- TREC blog and TREC terabyte look interesting, though I could not find links to
  the datasets or more information on the data
- TREC web - ClueWeb12(09 version too) dataset (filtered and preprocessed
  crawl), too large (in terabytes of data) and paid
- KILT from Meta AI - different tasks to what I need
- CSFCube - dataset for evaluation of IR methods based on faceted Query by
  example task in scientific paper. Basically an example scientific paper is
  given and the task it to find similar scientific papers based on {background,
  objective, method, result, other} part of the query document. It is unfit due
  to the nature of faceted queries. I cannot ask an embedding model to embed
  different parts of the document separately. It cannot therefore do the IR
  task: "Give me all documents with background similar to this document's
  background."
- [TREC Robust 04][trec-robust04] - 250 keyword-queries over 520K news articles
- [TREC Robust 05][trec-robust05] - 50 hard queries from Robust 04 over 1M docs
  from AQUAINT - license is needed and data must be paid for
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
  available. Was at [http://mleg.cse.sc.edu/DeepPatent/](http://mleg.cse.sc.edu/DeepPatent/).
- [TREC CAR][trec-car] - mainly paragraph ordering task to explain query, short
  paragraphs
- [CISI][cisi] - short documents, questions as queries

