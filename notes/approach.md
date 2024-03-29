[datasets]: datasets.md
[sbert]: https://arxiv.org/abs/1908.10084
[longformer]: https://arxiv.org/pdf/2004.05150v2.pdf
[reformer]: https://arxiv.org/pdf/2001.04451.pdf
[jian_22]: https://arxiv.org/pdf/2209.09433.pdf
[reimers_20]: https://aclanthology.org/2020.emnlp-main.365.pdf

# Approach

In short combine [SBERT][sbert] and either [Longformer][longformer] or
[Reformer][reformer]. The goal is to create contextual embedding of the entire
document.

We will evaluate the model on tasks described [here][datasets].

## Abstract goal

My idea of the model is that it'd allow you to see past the story or the topic.
Gain as much understanding not only about the text itself but also about it's
context. Utilize the broad spectrum of documents it trained on and see the
current one in the context of those.

## Ideas

### Knowledge distillation

The difficulty of the problem I aim to solve (train good embeddings from long
texts), boils down to the lack of data -- we do not have long documents with all
kinds of properties or true similarities. Instead let's try to mimic model able
to tell us something about the long document, while also adding some interesting
information about small texts (with the data we have). Training to mimic the
outputs of another model is called *knowledge distillation*.

This was already done by [Reimers 2020][reimers_20]

### Contrastive learning

We can finetune the model on unsupervised dataset with contrastive learning
like in [Jian 2022][jian_22]. It should be slightly worse than high-quality NLI
datasets, if there was one.

### Reinforcement learning

Can be reinforcement learning used in learning unsupervised document embeddings?
