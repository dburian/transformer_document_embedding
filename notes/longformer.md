[paper]: https://arxiv.org/abs/2004.05150
[hf_longformer]: https://huggingface.co/allenai/longformer-base-4096
[i/longformer_attention]: ./imgs/longformer_attention.png

# Longformer

Introduced by [Beltagy, Peters, Cohan in 2020][paper].


Longformer is one of the transformer models, whose aim is to alleviate
transformers of quadratic complexity in length of input.

The premise of Longformer is to provide a solution for NLP tasks with longer
inputs in a way that avoids task-specific solutions and supports transfer
learning. Consequently Longformer was designed so that fine-tuned transformer
weights (such as those from RoBERTa) can be used also for Longformer.

## Attention types

Longformer designs new-types of self-attention which scale linearly in input
length:

- local sliding attention - every token receives information from it's local
  window (usually 512 tokens in total). Receptive field of given attention is
  linear function of both the window width and the attention's layer.

- dilated local sliding window - instead of the local window being dense, it can
  look at only k-th token meaning the window will have gaps. This increases the
  attention's receptive field while leaving out some details. This type of
  attention is only possible with custom CUDA kernel.

- global attention - chosen tokens (such as the CLS token) may receive attention
  from all other tokens. This makes self-attention in certain predefined
  locations powerful, so the resulting output can be used to asses the whole
  input. To obtain global attention's projections different weights are learned,
  yet to support transfer learning from models without such projections, they
  are initiated with normal projections.

![Longformer attention types][i/longformer_attention]

## Pre-trained weights

The model is available on [HuggingFace][hf_longformer] though I am not sure how
to use pre-trained RoBERTa weights...

Limitation could be positional embeddings which were learned using the original
BERT approach. Longformer copies the learned embeddings 8 times to accommodate
up to 4096 tokens on the input. However, I either:

1. need the learned positional embeddings
2. fine tune them on my own -- 65k gradient updates at most,
3. try-out the computed ones.
