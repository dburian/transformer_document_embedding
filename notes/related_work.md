[doc2vec]: doc2vec.md
[jian_22]: https://arxiv.org/pdf/2209.09433.pdf
[d/longformer]: doc/longformer.md
[d/sbert]: doc/sbert.md
[bigbird]: https://arxiv.org/abs/2007.14062
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

# Related work

All the sources in the field to give me some inspiration with my thing.

## Document embeddings wo/ Transformers

- [Doc2Vec][doc2vec]
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
- [SPECTER - scientific article embedding based on SciBert][cohan_20]
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
      alignment with cross document attention][zhou_20]


## Backbones

- [SBERT][d/sbert]
- [Longformer][d/longformer]
- [BigBird 2021][bigbird] 

- [Multi-document tranformer for personality detection][yang_21]

- [Comparison of transformer-like models in classification][dai_22]
- [Pretrained Language Models for Sequential Sentence Classification][cohan_19]
  - Transformer layers that "directly utilize contextual information from all
    words in all sentences"
- [CDLM][cdlm] - Cross-Document Language Modeling - adaptation of Longformer for
  cross-document tasks

## Theory

- [BigBird 2021][bigbird]
- [Sparse transformers by OpenAI][child_19]

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


## Practical information

At some point I'll need to quickly similar documents according to their
embeddings..Ran across this list of tools:

- [Faiss][faiss] - library in C++ with wrappers for Python for similarity search
  and clustering of dense vectors. Developed by Facebook AI.
- [Weaviate][weaviate] - complete and open source vector oriented search engine.
- [Milvus][milvus] - vector database built for scalable similarity search.


I should dedicate some pages to efficency. Checkout this [benchmark for efficent
transformers by Tay, Dehghani et al. in 2020][tay_20]
