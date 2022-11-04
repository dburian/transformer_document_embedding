[article]: https://arxiv.org/abs/1611.09268
[github_io]: https://microsoft.github.io/msmarco/
[dl_overview21]: https://www.microsoft.com/en-us/research/uploads/prod/2022/05/trec2021-deeplearning-overview.pdf
[trec22]: https://microsoft.github.io/msmarco/TREC-Deep-Learning


# MS MARCO

## Links

- [Article][article]
- [Official website][github_io]

- [TREC DL 22][trec22]
- [overiview of TREC DL 2021 introducing v2 version of the
  dataset][dl_overview21] - TODO read about

## Tasks

There were two tasks:

- passage ranking - how much given passage answers given query,
- document ranking - how much passages in the given document answer given query,

each with two subtasks:

- full retrieval - retrieving the 100 most relevant passages/documents with
  ranking scores,
- top 100 reranking - reordering and reassigning ranking scores to 100 given
  passages/documents.

There were number of versions of this dataset:

- original **v1** - used for TREC DL 2019, 2020 and for the official MS MARCO
  leaderboard
- **v2** - used for TREC DL 2021, 2022. Created by adding more documents by a
  later crawl of the data (even bigger lag between passage crawl and document
  crawl viz. [problems](#problems)).


### Evaluation

TODO: MRR, NDCG, MSMARCO Qrels and NIST qrels


## Problems

Relevance of documents is inferred based on relevance of passages contained
within that document. However, documents were collected as an additional
processing step and therefore are likely to not contain the passages obtained
earlier or not even exist. This problem is thought of as being "realistic" in a
sence there is high probability of encountering broken data in the wild. Instead
of investing resources to completely new dataset, NIST decided to invest effort
into cleaning the dataset and reusing old labels. [source][dl_overview21]

## Data analysis

- #docs: 138M
- avg. number of words in first 200k docs: 1545

Example questions:

- "define extreme"
- "tattoo fixers how much does it cost"
- "what is a bank transit number"
- "what are the four major groups of elements"
- "blood clots in urine after menopause"
- "what is delta in2ition"
- "symptoms of an enlarged heart in dogs"
- "number of times congress voted to repeal aca"
- "how does a firefly light up"
- "what was introduced to the human diet in what year"
