[./d/models]: ./doc/models.md
[./d/related_work]: ./doc/related_work.md
[d/timeline]: doc/timeline.md
[d/approach]: doc/approach.md
[d/datasets]: doc/datasets.md
[awesome_ir]: https://github.com/harpribot/awesome-information-retrieval
[google_doc_topic]: https://docs.google.com/document/d/13Yb34eyklpX6bGzaf3m0jlsFb8rF10KvLXh4DuY4SD0/edit#heading=h.k2zhq4p261n
[reformer]: https://arxiv.org/pdf/2001.04451.pdf
[xiong_21]: https://arxiv.org/pdf/2112.07210.pdf
[luo_21]: https://arxiv.org/pdf/2103.14542.pdf
[awesome_ds]: https://github.com/malteos/awesome-document-similarity
[medic_22]: https://arxiv.org/pdf/2209.05452.pdf
[xiong_20]: https://arxiv.org/abs/2007.00808
[ir_datasets]: https://ir-datasets.com/index.html
[hammerl_22]: https://aclanthology.org/2022.findings-acl.182.pdf
[timekey_21]: https://vansky.github.io/assets/pdf/timkey_vanschijndel-2021-emnlp.pdf
[rajaee_22]: https://aclanthology.org/2022.findings-acl.103.pdf
[openai_19]: https://arxiv.org/abs/1904.10509

# Transformer document embedding

This is a repo containing everything connected to my thesis. **Currently in
progress.**

## Documentation entry points

- [Datasets info][d/datasets]
- [Approach][d/approach]
- [Timeline][d/timeline]
- [Related work][./d/related_work] - collection of articles that might be useful
  in the future
- [Models][./d/models] - list of models models considered


## Links

- [Finalized topic](https://is.cuni.cz/studium/dipl_st/index.php?id=a91fb39f906ae7e035142a978450e151&tid=1&do=main&doo=detail&did=250786)


## Installation

This repo contains experiments with different models on different tasks. To run
these you need to install this python package. I recommend editable install to
be able to change the source files with:

```bash
pip install --editable .
```


## Structure of this repo

- `src` - source code for the experiments
- `notebooks` - jupyter notebooks either showcasing something or containing
  some really experimental code
- `paper` - latex source code of my thesis paper
- `notes` - my research notes,
- `log` - development logs (may contain some info but do not expect
  documentation)
