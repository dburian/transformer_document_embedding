[bigbird]: bigbird.md
[paragraph_vector]: paragraph_vector.md
[jian_22]: https://arxiv.org/pdf/2209.09433.pdf
[longformer]: longformer.md
[d/sbert]: doc/sbert.md
[dai_22]: https://arxiv.org/abs/2204.06683
[tian_20]: https://proceedings.neurips.cc/paper/2020/hash/4c2e5eaae9152079b9e95845750bb9ab-Abstract.html
[yang_21]: https://ojs.aaai.org/index.php/AAAI/article/view/17673
[cohan_20]: https://arxiv.org/abs/2004.07180
[cohan_19]: https://aclanthology.org/D19-1383/
[faiss]: https://faiss.ai
[weaviate]: https://weaviate.io
[milvus]: https://milvus.io
[tay_20]: https://arxiv.org/abs/2011.04006
[ppdb]: http://paraphrase.org/#/download
[cdlm]: https://aclanthology.org/2021.findings-emnlp.225
[ginzburg_21]: https://arxiv.org/pdf/2106.01186.pdf
[shroff_15]: https://arxiv.org/abs/1503.03832
[child_19]: https://arxiv.org/abs/1904.10509
[li_20]: https://arxiv.org/pdf/2008.08567.pdf
[laser]: https://arxiv.org/abs/1704.04154
[yang_16]: https://aclanthology.org/N16-1174.pdf
[pappagari_19]: https://ieeexplore.ieee.org/abstract/document/9003958
[zhou_20]: https://aclanthology.org/2020.emnlp-main.407.pdf
[gururangan_20]: https://aclanthology.org/2020.acl-main.740.pdf
[transformer_xl]: https://arxiv.org/abs/1901.02860
[yang_20]: https://dl.acm.org/doi/pdf/10.1145/3340531.3411908
[jiang_19]: https://dl.acm.org/doi/pdf/10.1145/3308558.3313707
[radev_13]: https://link.springer.com/article/10.1007/s10579-012-9211-2
[howard_18]: https://arxiv.org/pdf/1801.06146.pdf
[tay_22]: https://dl.acm.org/doi/full/10.1145/3530811
[kitaev_20]: https://arxiv.org/abs/2001.04451

# Related work

All the sources in the field to give me some inspiration with my thing.

## Efficient transformers

For overall picture see [Efficient transformes: A survey][tay_22] --
Summarization of efficient transformer models in recent years.

For comparison see [Long Range Arena][tay_20] -- systematic evaluation of
transformers for long sequences
  - there is task to see how well models compress information, testing on [AAN
    dataset][radev_13].

### Fixed patterns

- [BigBird 2021][bigbird]
- [Sparse transformers by OpenAI][child_19] --- focuses only on autoregressive
  models
- [Longformer (2020)][longformer]

### Learnable patterns

- [Reformer][kitaev_20]

### Recurrence

- [Transformer XL (2019)][transformer_xl] -- example of left-to-right transformer
  (rather than with sparse attention)

### Kernel tricks

- Performer

### Low-rank methods

- Linformer

### Sparse

> Updating only some parameters via Mixture of experts-like mechanisms.

- [Adaptively Sparse Transformers](https://arxiv.org/abs/1909.00015) -- maybe a
  relevant model

### Mics

- Encoder+Decoder [LongT5](https://arxiv.org/pdf/2112.07916.pdf)
- Distraction issue of long contexts: [Focused
  Transformer](https://arxiv.org/pdf/2307.03170.pdf)
- How to finetune attention span: [Adaptive Attention Span in
  Transformers](https://arxiv.org/abs/1905.07799) -- for sure relevant
  especially if we choose different attention spans for Longformer


## Document Embedding models

### Old-school neural networks

- [Paragraph vector][paragraph_vector] -- see extensions as well
- [Zamani et al. (2018)](https://dl.acm.org/doi/pdf/10.1145/3269206.3271800)
    - sparse representations designed solely for IR
    - word embeddings to sparse n-gram embeddings which are averaged
    - trained with hinge loss that gets embedding of positive, negative document
      both first multiplied with query embedding (all are sparse)

### Variation Autoencoders

The models where we obtain $\mu$ and $\sigma$ through FC layers, to create a
distribution (e.g. $\mathcal{N}(0, 1) \sigma + \mu$) from which we sample.

- [Holmer et al. (2018)](https://arxiv.org/pdf/1806.01620.pdf)
    - train two sets of word embeddings, one for global context, one for local
    - the global embeddings are used to create the distribution from which a
      document embedding is sampled, this is concatenated with local embedding
      computed from $1:k$ words to predict the $k+1$-th word
    - experiments were done on classification tasks (20Newsgroups, RCV1-v2,
      IMDB)
- [Wang et al.
  (2019)](https://proceedings.neurips.cc/paper_files/paper/2019/file/3a029f04d76d32e79367c4b3255dda4d-Paper.pdf)
    - extension of VAE, modelling not one document embedding but pairs of
      document embeddings
    - I just skimmed it, read into it if necessary

### Transformer

- [Open AI embedding with contrastive loss](https://arxiv.org/abs/2201.10005) --
  contrastive learning, training at scale, large batches, 300M-175B models
  - All related work regarding embedding models done
- [SPECTER2](https://arxiv.org/pdf/2211.13308.pdf) -- different embeddings for
  different tasks
  - All related work regarding embedding models done
- [SPECTER - scientific article embedding based on SciBert][cohan_20] -- SciBert
  (Bert trained on scientific documents) further trained using triplet loss and
  examples generated using citations (positives are papers sharing a citation).
  The embedding of an abstract is used as the document embedding.
  - evaluated on classification, citation prediction, visualization
  - All related work regarding embedding models done
- [Ostendorff et al. (2022)](https://arxiv.org/pdf/2202.06671.pdf)
    - follows SPECTER in model and methodology -- SciBert initialization, use
      with triplet loss
    - the difference is controlled sampling of negatives from citation graph
    - evaluation on SciDocs (outdated version of SciRepEval)
- [Izacard et al. (2022)](https://arxiv.org/pdf/2112.09118v4.pdf)
    - train a transformer using contrastive loss on unlabelled corpora
    - positive pairs are augmented versions of the same document
    - studies how term-frequency models (e.g. BM25) surpass dense vector
      representations on IR tasks, when few data is available
    - highlight: dense representations are good in multilingual retrieval (where
      term-frequency models are not transferable) and in scenarios where
      some training data exist (where term-frequency models cannot adjust their
      weights, but dense embedders can)
    - evaluation on
        - two question-answering datasets, where answers are picked from a
          collection of documents (e.g. wikipedia dump)
        - BEIR benchmark
        - few-shot retrieval scenario where only a few training data is
          available
        - multilingual retrieval

### Convolution architectures

- [Liu et al. (2018)](https://arxiv.org/pdf/1711.04168.pdf)
    - convolution above word embeddings
    - trained on predicting next words
    - evaluation on classification tasks

### Recurrent networks

- [van den Oord et al. (2019)](https://arxiv.org/pdf/1807.03748.pdf)
    - learn representations of multiple modalities using RNNs over embedded
      tokens of the modality (e.g. words for text)
    - the loss is a special one (read more)
    - evaluation on classification tasks

### Graph networks

- [Xu et al. (2021)](https://aclanthology.org/2021.findings-emnlp.327.pdf)
    - training document embeddings with graphs
    - I would need to read-up on this to know how graph networks work
    - evaluation on retrieval and classification tasks

### Mixed architectures

- [Siamese Multi-depth Transformer-based Hierarchical (SMITH) Encoder for
  Long-Form Document Matching][yang_20]
    - another hierarchical approach using Transformers. Sentences are greadily
      fed into transformer towers. Whose [CLS] tokens are then fed to another
      transformer whose first token will be used as the document representation.
    - uses similarity datasets Wiki65K (introduced in [Semantic Text Matching
      for Long-Form Documents][jiang_19]) and AAN104K. AAN104K uses citations as
      proxies for assuming similarity.
    - There is section on competing models, so look into that for more related
      work.

## Sentence embedding models

- [SBERT][d/sbert]
- [SimCSE](https://arxiv.org/pdf/2104.08821.pdf) -- contrastive learning on
  sentences

## Similarity models

- [Self-Supervised Document Similarity Ranking][ginzburg_21]
    - Really helpful related work, talks about all I should talk about as well.
    - Introduces *SDR* model w/ RoBERTa backbone. Trained in an unsupervised
      fashion with masked LM and contrastive loss -- from a set of documents
      positive and negative pairs of sentences are picked (positive are from the
      same document).
    - TODO: there are links to other useful papers in here
- [Semantic Text Matching for Long-Form Documents][jiang_19]
    - Hierarchical RNN with attention
    - the model computes representation of documents, but the authors then
      attach a head that computes their similarity. Moreover the authors
      highlight that using simple function like cosine to compute similarity
      from the two embeddings doesn't work. So similarity, not embedding.
    - introduces similarity Wikipedia dataset which uses links as proxi for
      assuming similarity

## Training

### Extending context length

- [Extending Context Window of Large Language Models via Positional
  Interpolation](https://arxiv.org/abs/2306.15595) -- finetuning to increase
  context length $\approx$ a way to train embeddings

### Misc

- [FaceNet][shroff_15] - Talks about creating an embedding of pictures for face
  recognition. It uses learning with triplets: anchor, positive, negative and
  explains how to generate triplets, such that the model doesn't diverge and
  converges quickly. Referenced by SBert.

- Siamese networks - Two identical networks (with tied weights) producing two
  outputs, both of which are needed for the network loss. Usually the network
  output some feature vector and the loss either pushes the vectors towards each
  other (positive pair) or farther apart (negative pair). Thus they are great
  fit for networks producing embeddings. They need smaller number of examples,
  but train for longer.


- Triplet networks - Similar idea to Siamese networks, only now we have three
  identical networks. They may use loss with an anchor example:

$$
    \mathcal{L}(y_a, y_p, y_n) =
    max(||y_a - y_p|| + \epsilon - ||y_a - y_n||, 0)
$$,

  where $y_x$ stands for embedding of anchor ($a$), positive ($p$) and negative
  examples ($n$). Such loss is caled triplet objective loss. Used by SBert.


- [Good views on data for contrastive learning][tian_20]
- [PPDB][ppdb] - paraphrases. Maybe to give our model more robustness?

- [Contrastive learning of Sent. embed. using non-linguistic
  modalities][jian_22]

> "we show that Transformer models can generalize better by learning a similar
> task (i.e., clustering) with multi-task losses using non-parallel examples
> from different modalities."

- Pre-training is important as documented in [Don’t Stop Pretraining: Adapt
  Language Models to Domains and Tasks][gururangan_20].

- [Universal Language Model fine-tuning][howard_18] - SOTA NLP classification
  finetuning approach

## Vector databases

At some point I'll need to quickly similar documents according to their
embeddings..Ran across this list of tools:

- [Faiss][faiss] - library in C++ with wrappers for Python for similarity search
  and clustering of dense vectors. Developed by Facebook AI.
- [Weaviate][weaviate] - complete and open source vector oriented search engine.
- [Milvus][milvus] - vector database built for scalable similarity search.

## Misc & TODO

- Attempts to make with the 512 tokens BERT has to offer, instead of creating a
  model without quadratic complexity in the input length -- could probably be a
  baseline. E.g. SpanBERT, ORQA, REALM, RAG (BigBird's related work).
- [Multi-document transformer for personality detection][yang_21]
- [Comparison of transformer-like models in classification][dai_22]
- [Compressive Transformers for Long-Range Sequence
  Modelling](https://arxiv.org/abs/1911.05507?ref=pragmatic.ml) -- relevant
  model as it uses different mechanisms to increase context length that I do not
  mention yet
- [Pretrained Language Models for Sequential Sentence Classification][cohan_19]
  - Transformer layers that "directly utilize contextual information from all
    words in all sentences"
- DUNNO? [Memorizing transformer](https://arxiv.org/pdf/2203.08913.pdf)
- [CDLM -- Cross-Document language modeling][cdlm] --- adaptation of Longformer
  for cross-document tasks. Interesting use of global attention and new masked
  LM objective forcing the longformer to query information (to predict the
  masked word) from other "similar" document. Evaluation done on cross-document
  tasks. Only comparable task is the one introduced by [Multilingual text
  alignment with Cross Document Attention (CDA)][zhou_20], where CDLM achieves
  SOTA performance on 3 out of 4 tasks.
- [Transformer based Multilingual document embedding model][li_20] - transformer
  version of LASER with a special distant constraint loss
- [Hierarchical Attention Network (HAN)][yang_16] -- built on the
  word/sentence/document hierarchy. First each word token is contextualized
  (using BERT or BiRNN), each sentence is aggregation of its contextualized word
  vectors and is after contextualized with other sentences (using transformer
  [Pappagari et al. 2019][pappagari_19] or again BiRNN), each document is an
  aggregation of its contextualized sentence representations.
    - Pappagari is focused on document classification, not embedding
    - Pappagari used 20Newsgroups and some automatically/manually generated
      conversation transcripts from audio. I cannot tell if the findings were
      anything to write home about.
    - HANs were augmented using a cross-document attention in [Multilingual text
      alignment with cross document attention (CDA)][zhou_20]
- [Multilingual text alignment with cross document attention (CDA)][zhou_20] --
  is HAN focused on downstream tasks (like binary classification) straightly,
  there isn't much talk about embeddings
