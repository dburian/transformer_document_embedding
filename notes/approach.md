[sbert]: https://arxiv.org/abs/1908.10084
[longformer]: https://arxiv.org/pdf/2004.05150v2.pdf
[reformer]: https://arxiv.org/pdf/2001.04451.pdf
[jian_22]: https://arxiv.org/pdf/2209.09433.pdf
[reimers_20]: https://aclanthology.org/2020.emnlp-main.365.pdf
[distilbert]: https://arxiv.org/pdf/1910.01108.pdf

# Approach

In short combine [SBERT][sbert] and either [Longformer][longformer] or
[Reformer][reformer]. The goal is to create contextual embedding of the entire
document.

## Ideas

### Knowledge distillation

Knowledge distillation is about mimicking other model's output on given inputs.
In our case we need to combine two aspects of embeddings:
    1. capturing meaning, similarity, structure (all the good things),
    2. working on large documents

We could do this by distilling SBERT (1.) and Paragraph Vector (2.). We could
also use different losses:
    - we should estimate SBERT embeddings more closely: *MSE*
    - capture only the structure of Paragrpah vector embeddings: *DCCA*

Inspirations:
- [Reimers et al. 2020][reimers_20] used distillation to make monolingual
  embeddings multilingual using parallel data.
- [DistilBert][distilbert] was trained using loss which combined LM,
  distillation and cosine-distance losses.


### Contrastive learning

We can finetune the model on unsupervised dataset with contrastive learning
like in [Jian 2022][jian_22]. It should be slightly worse than high-quality NLI
datasets, if there was one.

### Reinforcement learning

Can be reinforcement learning used in learning unsupervised document embeddings?
