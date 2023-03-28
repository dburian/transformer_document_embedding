
# My SBERT implementation

Using `sentence-transformers`.

Seems that you pick pretrained model (i.e. transformer with pooling layer) or you
create your own using pretrained transformer and custom pytorch layers. Then a
loss is applied above.

Architectures:
- **for classification**: MLP on top of pooled hidden states (i.e. on top of
  embedding)

## Notes on implementation

Using `sentence-transformers` is possible via the package itself or just using
HF and pytorch Module.

There is a helper function `SentenceTransformer.smart_batching_collate` which
can prepare the correct features for the pytorch Module. Use `DataLoader` with
`InputExample`s and `SentenceTransformer.smart_batching_collate` as
`collate_fn`.


## Hardware requirements

As far as I know we are hitting the max performence of the GPUs in AIC.

- 64 batches were OOM
- 2 batches used around 25% memory

# Evaluation

## IMBD

The results of various configurations of the classification head were really
close.

| val accuracy | hidden dropout | hidden features | label smoothing |
| ------------ | -------------- | --------------- | --------------- |
| 0.9348       | 0.1            | 25              | 0.15            |
| 0.9344       | 0.5            | 50              | 0.15            |
| 0.9334       | 0.5            | 25              | 0.2             |
| 0.9324       | 0.5            | 25              | 0.15            |
| 0.932        | 0.5            | 0               | 0.15            |
| 0.93         | 0.5            | 25              | 0.1             |
| 0.929        | 0              | 25              | 0.15            |
| 0.9282       | 0.5            | 150             | 0.15            |

Observations:
- too many hidden features hurts the performence
- best results seem to have around 25 - 50 hidden features with .15 label
  smoothing and at least 0.1 dropout (though 0.5 does not hurt that much)
- after 10 epochs validation accuracy seem to have stabilized
