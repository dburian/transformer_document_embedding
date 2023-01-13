[cord-nist]: https://ir.nist.gov/trec-covid/data.html
[cord-19]: https://arxiv.org/abs/2004.10706
[ir_datasets-cord]: https://ir-datasets.com/cord19.html

# CORD-19

Dataset of scientific articles on COVID-19 which served as corpus of ad-hoc
retireval TREC COVID task.

## Links

- original [paper on CORD-19][cord-19]
- [description of data provided by TREC][cord-nist]


## Description

There are multiple rounds which differ in documents used and therefore also in
qrels and queries. Same document may have different id in different rounds.

Final test set of 50 topics was made available as the 16. July 2020 release.
Also found on `ir_datasets` under ['cord19/trec-covid'][ir_datasets-cord].

All qrels are human assesed.

## Analysis

- #docs: 193K
- #queries: 50
- #qrels: 69K

- avg. length of text: 1.7k words
- avg. lenght of abstract: 140 words
