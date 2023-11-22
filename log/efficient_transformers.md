[flash_attn]: https://github.com/Dao-AILab/flash-attention
[xformers]: https://facebookresearch.github.io/xformers

# Efficient Tranformers

The situation AFAIK is quite complicated. I've split up notes from various
places to two concrete tasks: optimizing only the self-attention and optimizing
the model as a whole.

## Efficient Attention

### Fused kernels

There are custom kernels that fuse together the operations, namely:
- Memory efficient kernel introduced in [paper by
    google](https://arxiv.org/pdf/2112.05682.pdf) and implemented by [xFormers
    library][xformers]
- [Flash Attention][flash_attn] v1 and v2

Flash attention does not support masking -- no pad tokens, sources: [HF
documentation](https://huggingface.co/docs/transformers/perf_infer_gpu_one#expected-speedups),
pytorch GitHub comments:
[here](https://github.com/pytorch/pytorch/issues/107884#issuecomment-1719388335),
[here](https://github.com/pytorch/pytorch/issues/96099#issuecomment-1560307181).
The possible solution would be to use `NestedTensor` instead of padding,
described in [a pytorch
tutorial](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#nestedtensor-and-dense-tensor-support)

xFormers's implementation of memory efficient kernel does support attention
masking (see `AttentionBias` in [xFormers API
docs](https://facebookresearch.github.io/xformers/components/ops.html#module-xformers.ops),
but it suggests that using custom mask will make the computation slower.

### SDPA function

Pytorch (2.0) added new function to `torch.nn.functional` that encompasses
self-attention known as Scaled-Dot-Product Attention (SDPA) that supports
selecting the best kernel out of Mem. eff. attention, Flash Attention and
simple cpu rewrite of the classical operations. The inclusion of xFormers mem.
eff. kernel with custom masking is scheduled for 2.0.2
([source](https://github.com/pytorch/pytorch/issues/96099#issuecomment-1490675039)).


## Eficient models

### `torch.compile(model)`

Since 2.0 Torch supports 'compiling' a model, which significantly decreases
the overhead caused by a framework. Note that it is only supported for python
<=2.10 (support for 2.11 is included in Nithly release).

### BetterTransformer

In 1.12 PyTorch added a `BetterTransformer` interface, which speeds up models
composed of `torch.nn.Tranformer*` modules. [HF Optimum
library](https://huggingface.co/docs/optimum/index) includes this support also
for HF models. However, it also replaces self-attention with SDPA pytorch
function
([source](https://huggingface.co/docs/optimum/bettertransformer/overview)).
This currently means that, since SDPA function does not support custom masks,
the increase in efficiency is not as large as it could be.
