[bigbird]: https://arxiv.org/abs/2007.14062
[dai_22]: https://arxiv.org/abs/2204.06683
[tian_20]: https://proceedings.neurips.cc/paper/2020/hash/4c2e5eaae9152079b9e95845750bb9ab-Abstract.html
[yang_21]: https://ojs.aaai.org/index.php/AAAI/article/view/17673
[cohan_20]: https://arxiv.org/abs/2004.07180
[cohan_19]: https://aclanthology.org/D19-1383/
[faiss]: https://faiss.ai
[weaviate]: https://weaviate.io
[milvus]: https://milvus.io
[ppdb]: http://paraphrase.org/#/download
[cdlm]: https://aclanthology.org/2021.findings-emnlp.225
[ginzburg_21]: https://arxiv.org/pdf/2106.01186.pdf


# Related work

All the sources in the field to give me some inspiration with my thing.

## Models

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

- [Good views on data for contrastive learning][tian_20]
- [PPDB][ppdb] - paraphrases. Maybe to give our model more robustness?

## Practical information

At some point I'll need to quickly similar documents according to their
embeddings..Ran across this list of tools:

- [Faiss][faiss] - library in C++ with wrappers for Python for similarity search
  and clustering of dense vectors. Developed by Facebook AI.
- [Weaviate][weaviate] - complete and open source vector oriented search engine.
- [Milvus][milvus] - vector database built for scalable similarity search.
