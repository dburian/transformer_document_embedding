[sbert]: https://arxiv.org/abs/1908.10084
[longformer]: https://arxiv.org/pdf/2004.05150v2.pdf
[reformer]: https://arxiv.org/pdf/2001.04451.pdf
[jian_22]: https://arxiv.org/pdf/2209.09433.pdf
[reimers_20]: https://aclanthology.org/2020.emnlp-main.365.pdf
[distilbert]: https://arxiv.org/pdf/1910.01108.pdf
[llama_long]: https://arxiv.org/pdf/2309.16039.pdf
[flash_attention]: https://github.com/Dao-AILab/flash-attention
[sim_cse]: https://arxiv.org/pdf/2104.08821.pdf
[openai_embed]: https://arxiv.org/abs/2201.10005

# Goals and plan how to reach them

The goal of this thesis is to be able to semantically embed long pieces of
continuous text.

## Paths to goal

To reach the ultimate goal described above we need to fulfill the following:
1. have transformer encoder capable of encoding long sequences
2. fine-tune the encoder to generate semantic embeddings

### Long-context transformer

#### Use transformer with shorter context

There are several approaches that use a transformer model that originally was
pre-trained with shorter context, and fine-tune it for longer context
(optionally with different attention mechanism):

- use full attention with [FlashAttention][flash_attention] and
  [xFromers](https://facebookresearch.github.io/xformers/) as suggested by the
  [LLama 2 Long paper][llama_long]
    - FlashAttention cleverly optimizes the computation using custom attention functions
    - xFromers implements different kinds of attention in pytorch utilizing
      implementation from [paper by google that claims to compute
      self-attention using only O(log n)
      memory](https://arxiv.org/pdf/2112.05682.pdf)
    - Their use can be observed in [Mistral
      model](https://mistral.ai/news/announcing-mistral-7b)
- use transformer pre-trained with sparse attention but double (quadruple) its
  context

For such approaches we would need to heavily fine-tune the model. This was done
e.g.

- In [Extending context window of large language models via position
  interpolation](https://arxiv.org/pdf/2306.15595.pdf) the authors extended
  context of learned models that use [Rotary Position
  embeddings](https://arxiv.org/pdf/2104.09864.pdf). They used 32 A100 GPUs to
  extend LLaMA 7B model from 2k to 8k context. Note that they pre-train only for
  1000 steps, but the gpus are required due to memory constraints. They used
  [FlashAttention][flash_attention].
- In [Processing Long Legal Documents with Pre-trained
  Transformers](https://arxiv.org/pdf/2211.00974.pdf) the authors took
  Longformer and doubled its context size. Pre-training, however, was avoided
  due to limited resources without explicitly saying how much resources would
  be roughly necessary.
- [Focused Transformer](https://arxiv.org/pdf/2307.03170.pdf) used 32 TPUv3 cores
  to increase context-length of classical transformer decoder with 184M
  parameters.

We, probably, lack both resources and time for such heavy training.

Some helpful links:

- [Landmark Attention](https://arxiv.org/pdf/2305.16300.pdf) uses retrieval-like
  attention to increase context size.
- [Transformer Language Models without Positional Encodings Still Learn
  Positional Information](https://arxiv.org/pdf/2203.16634.pdf) suggests that
  positional embeddings are unnecessary.

#### Use transformer pre-trained with long context and use different attention

We can use a pre-trained long-context transformer, and simply use different
(maybe more efficient) attention.

There is a technique called [Low Rank Adaptation
(LoRA)](https://arxiv.org/pdf/2106.09685.pdf) that is able to fine-tune LLMs with
significantly less memory.

LoRA was then explored for finetuning LLMs in order to increase their context
size. The result was [LongLoRA](https://arxiv.org/abs/2309.12307). The paper
also introduces new kind of local attention that can be used only for
finetuning, but be switched off (e.g. model uses standard full attention) for
inference. This cannot be said for all type of attentions. The paper claims that
normal sparse attention cannot be used, once the model is pretrained with full
attention, because of "large gap" between the attention mechanisms.

#### Use transformer pre-trained with long context \[CHOSEN (for now)\]

This is the easiest option as it doesn't require any changes to the model. This
would mean the thesis would reduce to the problem of finding training suitable
for producing semantic embedding.

This is THE option to try since we should be able to reach some results quickly
as this is the simplest of options.

### Fine-tuning for semantic embeddings


#### Knowledge distillation

Knowledge distillation is about mimicking other model's output on given inputs.
In our case we need to combine two aspects of embeddings:
    1. Capturing meaning, similarity, structure (all the good things),
    2. Working on large documents

We could do this by distilling SBERT (1.) and Paragraph Vector (2.). We could
also use different losses:
    - we should estimate SBERT embeddings more closely: *MSE*
    - capture only the structure of Paragraph vector embeddings: *DCCA*

Inspirations:
- [Reimers et al. 2020][reimers_20] used distillation to make monolingual
  embeddings multilingual using parallel data.
- [DistilBert][distilbert] was trained using loss which combined LM,
  distillation and cosine-distance losses.


#### Contrastive learning

We can finetune the model on unsupervised dataset with contrastive learning
like in [Jian 2022][jian_22]. It should be slightly worse than high-quality NLI
datasets, if there was one.

As suggested by [LLama 2 Long][llama_long] pre-training on longer sequences is
not really essential. More important is the quality of the resources. Also
[LLama 2 Long paper][llama_long] suggests there are diminishing returns (in
terms of scores) when increasing context length. The smaller the model the
smaller the returns.

The key decision is how to construct positive pairs, since negatives are those
documents with different ids.

[OpenAI's embedding model][openai_embed] used neighbouring
paragraphs on the internet as positive pairs.

[SimCSE][sim_cse] (SOTA in sentence embeddings) used the same inputs with
different dropout applied.

TODO:
- [] [Focused Transformer](https://arxiv.org/pdf/2307.03170.pdf) used contrastive loss. Links to
  other papers it linked to are below.
- [] [SimCLR](http://proceedings.mlr.press/v119/chen20j/chen20j.pdf) used
  contrastive loss to train visual representations.
- [] [Scaling Deep Contrastive Learning Batch Size under Memory Limited
  Setup](https://arxiv.org/pdf/2101.06983.pdf) introduces some optimizations
  to quickly compute in-batch negatives.
- [] [SimCSE][sim_cse] uses dropout as data augmentation to produce positive
  examples.
- [] Look at links leading from [OpenAI's embedding paper][openai_embed].
