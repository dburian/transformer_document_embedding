
# My SBERT implementation

Using `sentence-transformers`.

Seems that you pick pretrained model (i.e. transformer with pooling layer) or you
create your own using pretrained transformer and custom pytorch layers. Then a
loss is applied above.

Architectures:
- **for classification**: MLP on top of pooled hidden states (i.e. on top of
  embedding)



## Hardware requirements

As far as I know we are hitting the max performence of the GPUs in AIC.

- 64 batches were OOM
- 2 batches used around 25% memory

# Evaluation

## IMBD

The results of various configurations of the classification head were really
close and chaotic. I am suspicious that all configurations are more-or-less
equal and the difference is caused by different shuffling of the input dataset.
For example the following are the best results after 4 epochs and with
`batch_size = 6`:

| hidden_features | label_smoothing | hidden_dropout | binary_accuracy |
| --------------- | --------------- | -------------- | --------------- |
| 25.000          | 0.10000         | 0.50000        | 0.94212         |
| 0.0000          | 0.20000         | 0.25000        | 0.94212         |
| 75.000          | 0.10000         | 0.50000        | 0.94196         |
| 75.000          | 0.20000         | 0.50000        | 0.94176         |
| 25.000          | 0.10000         | 0.0000         | 0.94132         |
| 0.0000          | 0.15000         | 0.0000         | 0.94128         |
| 75.000          | 0.10000         | 0.0000         | 0.94124         |
| 75.000          | 0.20000         | 0.25000        | 0.94088         |

Nevertheless here are some observations:

- label smoothing helps (0, 0.1)
- dropout helps (0.5)
