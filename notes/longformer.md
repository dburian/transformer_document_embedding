[paper]: https://arxiv.org/abs/2004.05150
[hf_longformer]: https://huggingface.co/allenai/longformer-base-4096
[i/longformer_attention]: ./imgs/longformer_attention.png
[transformer_xl]: https://arxiv.org/abs/1901.02860
[sukhbaatar_19]: https://arxiv.org/abs/1905.07799 

# Longformer

Introduced by [Beltagy, Peters, Cohan in 2020][paper].


Longformer is one of the transformer models, whose aim is to alleviate
transformers of quadratic complexity in length of input.

The premise of Longformer is to provide a solution for NLP tasks with longer
inputs in a way that avoids task-specific solutions and supports transfer
learning. Consequently Longformer was designed so that fine-tuned transformer
weights (such as those from RoBERTa) can be used also for Longformer.

## Attention

### Types of attention

Longformer designs new-types of self-attention which scale linearly in input
length. First let's remember the original attention equation:
$$
Attention(Q, K , V) = softmax(\frac{QK^T}{\sqrt d_v}) V
$$

The problem of course is the matrix multiplications $QK^T$, where $Q \in
\mathcal{R}^{N\times E}$ and $T\in \mathcal{R}^{N\times E}$ and therefore $QK^T
\in \mathcal{R}^{N\times N}$ ($E$ is the embedding dimension, $N$ size of
input). This is the cause of quadratic complexity of full attention (and of
course later the same $N\times E$ matrix multiplication of the softmax result
and values $V \in \mathcal{R}^{N \times E}$.

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
  input. To give the model more flexibility different matricies $Q_g, K_g$ and
  $V_g$ are used for the global attention. These are initialized to the normal,
  pretrained matricies when transfering from transformer with dense attention.

We can think about the following images as visualizations of the $QK^T$ matrix
-- green cells are non-zero, white cells are zero.

![Longformer attention types][i/longformer_attention]

### Implementation

There are three implementations:

- `loop` -- PyTorch loop; extremely slow, though capable of supporting all three
  types of attention,
- `chunks` -- CUDA; splits the matrix into overlapping chunks, consumes
  quadratic amount of memory, does not support dilatation,
- `cuda` -- custom CUDA kernel; supports all types of attention, as fast as full
  attention --- to get any faster, one would need to know the ins and outs of a
  particular GPU/TPU. This implementation was used for autoregressive language
  modelling task as the memory complexity would otherwise be infeasible.


## Pretrained model


The authors supply a pretrained model. This model was:
- initialized with last RoBERTa checkpoint
- used RoBERTa's learned positional embeddings copied 8 times to support inputs
  up to 4096 tokens (16K tokens is possible with current GPUs).
- pretrained using MLM on long documents with 65K gradient steps.
- used with adaptive attention span: different attention span for different
  layers according to [Adaptive attention span in transformers][sukhbaatar_19]
- used with different dilatation for different heads.

The model is available on [HuggingFace][hf_longformer] with a set of pretrained
weights.


## Beoynd the 4096 tokens -- autoregressive character LM


The authors show how Longformer's attention can process really long sequences,
by:
- reusing [Transformer XL (2019)][transformer_xl] implementation,
- using relative sinusoidal positional embeddings
- staging learning: each stages doubles the window size and the sequence length
  and halves learning rate going from 2K to 23K tokens.

