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

# Related work

All the sources in the field to give me some inspiration with my thing.

## Document embeddings wo/ Transformers

- [Paragraph vector][paragraph_vector]
- [LASER][laser] -- learning sentence embeddings using parallel texts and
  forcing the multilingual representations to be close to each other, while also
  describing the content of the sentence.

## Document embeddings w/ Transformers

- [Self-Supervised Document Similarity Ranking][ginzburg_21]
    - Really helpful related work, talks about all I should talk about as well.
    - Introduces *SDR* model w/ RoBERTa backbone. Trained in an unsupervised
      fashion with masked LM and contrastive loss -- from a set of documents
      positive and negative pairs of sentences are picked (positive are from the
      same document).
    - TODO: there are links to other useful papers in here
- [SPECTER - scientific article embedding based on SciBert][cohan_20] -- SciBert
  (Bert trained on scientific documents) further trained using triplet loss and
  examples generated using citations (positives are papers sharing a citation).
  The embedding of an abstract is used as the document embedding.
- [CDLM -- Cross-Document language modeling][cdlm] --- adaptation of Longformer
  for cross-document tasks. Interesting use of global attention and new masked
  LM objective forcing the longformer to query information (to predict the
  masked word) from other "similar" document. Evaluation done on cross-document
  tasks. Only comparable task is the one introduced by [Multilingual text
  alignment with Cross Document Attention (CDA)][zhou_20], where CDLM achieves
  SOTA performance on 3 out of 4 tasks.
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
- [Semantic Text Matching for Long-Form Documents][jiang_19]
    - Hierarchical RNN with attention
    - introduces similarity Wikipedia dataset which uses links as proxi for
      assuming similarity
- [Transformer based Multilingual document embedding model][li_20] - transformer
  version of LASER with a special distant constraint loss
- [Hierarchical Attention Network (HAN)][yang_16] -- built on the
  word/sentence/document hierarchy. First each word token is contextualized
  (using BERT or BiRNN), each sentence is aggregation of its contextualized word
  vectors and is after contextualized with other sentences (using transformer
  [Pappagari et al. 2019][pappagari_19] or again BiRNN), each document is an
  aggregation of its contextualized sentence representations.
    - Pappagari used 20Newsgroups and some automatically/manually generated
      conversation transcripts from audio. I cannot tell if the findings were
      anything to write home about.
    - HANs were augmented using a cross-document attention in [Multilingual text
      alignment with cross document attention (CDA)][zhou_20]


## Backbones

- [SBERT][d/sbert]
- [BigBird 2021][bigbird]
- [Sparse transformers by OpenAI][child_19] --- focuses only on autoregressive
  models
- [Longformer (2020)][longformer]
- [Reformer][kitaev_20]

- [Efficient transformes: A survey][tay_22] -- Summarization of efficient
  transformer models in recent years

TODO: Learn about these more - how are global attentions trained, initialized in
inference, what makes them different
- [Longformer (2020)][longformer]
- [BigBird (2021)][bigbird]
- [Transformer XL (2019)][transformer_xl] -- example of left-to-right transformer
  (rather than with sparse attention)
- [LongT5](https://arxiv.org/pdf/2112.07916.pdf)
- [Memorizing transformer](https://arxiv.org/pdf/2203.08913.pdf)
- [Focused Transformer](https://arxiv.org/pdf/2307.03170.pdf)


- [Multi-document tranformer for personality detection][yang_21]

- [Comparison of transformer-like models in classification][dai_22]
- [Pretrained Language Models for Sequential Sentence Classification][cohan_19]
  - Transformer layers that "directly utilize contextual information from all
    words in all sentences"

## Learning

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

## Evaluation

- [Long Range Arena][tay_20] -- systematic evaluation of transformers for long
  sequences
  - hard to say which is better, so here is a benchmark
  - there is task to see how well models compress information, testing on [AAN
    dataset][radev_13].

## Practical information

At some point I'll need to quickly similar documents according to their
embeddings..Ran across this list of tools:

- [Faiss][faiss] - library in C++ with wrappers for Python for similarity search
  and clustering of dense vectors. Developed by Facebook AI.
- [Weaviate][weaviate] - complete and open source vector oriented search engine.
- [Milvus][milvus] - vector database built for scalable similarity search.


I should dedicate some pages to efficiency. Checkout this [benchmark for efficient
transformers by Tay, Dehghani et al. in 2020][tay_20]

## TODO

- Attempts to make with the 512 tokens BERT has to offer, instead of creating a
  model without quadratic complexity in the input length -- could probably be a
  baseline. E.g. SpanBERT, ORQA, REALM, RAG (BigBird's related work).
