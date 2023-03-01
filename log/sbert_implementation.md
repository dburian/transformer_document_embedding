
# My SBERT implementation

Using `sentence-transformers`.

Seems that you pick pretrained model (i.e. transformer with pooling layer) or you
create your own using pretrained transformer and custom pytorch layers. Then a
loss is applied above.


## Classification

For classification I use single layer with 1 output with Sigmoid activation. I
should rewrite it to a network with hidden layer.


## Hardware requirements

As far as I know we are hitting the max performence of the GPUs in AIC.

- 64 batches were OOM
- 2 batches used around 25% memory
