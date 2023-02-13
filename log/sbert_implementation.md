
# My SBERT implementation

Using `sentence-transformers`.

Seems that you pick pretrained model (i.e. transformer with pooling layer) or you
create your own using pretrained transformer and custom pytorch layers. Then a
loss is applied above.


## Classification

For classification of individual texts exist several options:

- using softmax classification head on top of pooling

Probably the easiest option, though I would need to delve into pytorch.

- using `ContrastiveLoss` or `OnlineContrastiveLoss`

- using some kind of triplet loss

Requires tuning the `margin` parameter. Also I should read into it more.
