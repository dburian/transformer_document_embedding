[paper]: https://arxiv.org/abs/2004.05150
[hf_longformer]: https://huggingface.co/allenai/longformer-base-4096
[i/longformer_attention]: ./imgs/longformer_attention.png
[transformer_xl]: https://arxiv.org/abs/1901.02860
[sukhbaatar_19]: https://arxiv.org/abs/1905.07799
[open_book_corpus_hf]: https://huggingface.co/datasets/bookcorpusopen
[book_corpus_hf]: https://huggingface.co/datasets/bookcorpus
[realnews_download]: https://www.google.com/url?q=https://storage.googleapis.com/grover-models/realnews.tar.gz&sa=D&source=editors&ust=1682090288933504&usg=AOvVaw34ItQejlxsu3JQmfi4fSuu
[stories_cc]: https://huggingface.co/datasets/spacemanidol/cc-stories
[realnews_gh]: https://github.com/rowanz/grover/tree/master/realnews


# Longformer

Introduced by [Beltagy, Peters, Cohan in 2020][paper]. Has cca 141M parameters.


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

Longformer uses the combination of local sliding attention, dilated local
sliding attention and global attention.

#### Local sliding attention

Every token receives information from it's local window (usually 512 tokens in
total). Receptive field of given attention is linear function of both the window
width and the attention's layer.

#### Dilated local sliding window

Instead of the local window being dense, it can look at only k-th token meaning
the window will have gaps. This increases the attention's receptive field while
leaving out some details. This type of attention is only possible with custom
CUDA kernel.

#### Global attention

Chosen tokens (such as the CLS token) may receive attention from all other
tokens. This makes self-attention in certain predefined locations powerful, so
the resulting output can be used to asses the whole input. To give the model
more flexibility different matricies $Q_g, K_g$ and $V_g$ are used for the
global attention. **These are trained per given task** and therefore are
initialized to the weights of the normal attention at the beginning of downstream
finetuning (i.e. global weights are not included in pre-trained weights). The
obvious disadvantage of separate global attention weights is **almost double the
memory footprint** of the model.

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


### Pretraining data

The authors collected corpus of long document to support long dependencies.

- Book corpus --- collection of freely available books collected on the Internet
    - **also in RoBERTa pretraining dataset**
    - the original dataset was used to train many models from BERT to GPT-N
      models
    - the original dataset is currently not available due to its many problems
      such as licensing
    - the replacement is *Open book corpus* available on
      [HuggingFace][open_book_corpus_hf]
    - the dataset is about 6 GB big with almost 18k books
    - also there is an [Home-made book corpus][home_made_book_corpus] --- a
      collection of books from smashwords.com, also available in
      [HuggingFace][book_corpus_hf]
- English Wikipedia
    - **also in RoBERTa pretraining dataset**
- RealNews dataset --- collection of news articles from Common Crawl
    - 120 GB of data
    - Available at [Google storage][realnews_download] (link from a [GitHub
      page][realnews_gh]
    - roughly third of the dataset was used with documents longer than 1200
      tokens
- Stories dataset --- texts from Common Crawl that bare similarity (though very
  small, but highest from documents in Common Crawl dataset) with Common sense
  dataset's questions
    - nearly 1M documents
    - there is a [HuggingFace almost identical variant][stories_cc]


## Beoynd the 4096 tokens -- autoregressive character LM


The authors show how Longformer's attention can process really long sequences,
by:
- reusing [Transformer XL (2019)][transformer_xl] implementation,
- using relative sinusoidal positional embeddings
- staging learning: each stages doubles the window size and the sequence length
  and halves learning rate going from 2K to 23K tokens.
