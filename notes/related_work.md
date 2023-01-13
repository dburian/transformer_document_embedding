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


# Related work

All the sources in the field to give me some inspiration with my thing.

## Models

- [SBERT][d/sbert]
- [Longformer][d/longformer]
- [BigBird 2021][bigbird] - transformer-like architecture for longer sequences
  developed by Google
- [Multi-document tranformer for personality detection][yang_21]

- [Comparison of transformer-like models in classification][dai_22]
- [SPECTER - scientific article embedding based on SciBert][cohan_20]
- [Pretrained Language Models for Sequential Sentence Classification][cohan_19]
  - Transformer layers that "directly utilize contextual information from all
    words in all sentences"
- [CDLM][cdlm] - Cross-Document Language Modeling - adaptation of Longformer for
  cross-document tasks
- [Self-Supervised Document Similarity][ginzburg_21]

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


## Practical information

At some point I'll need to quickly similar documents according to their
embeddings..Ran across this list of tools:

- [Faiss][faiss] - library in C++ with wrappers for Python for similarity search
  and clustering of dense vectors. Developed by Facebook AI.
- [Weaviate][weaviate] - complete and open source vector oriented search engine.
- [Milvus][milvus] - vector database built for scalable similarity search.


I should dedicate some pages to efficency. Checkout this [benchmark for efficent
transformers by Tay, Dehghani et al. in 2020][tay_20]
