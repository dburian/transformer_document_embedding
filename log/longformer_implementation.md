# My Longformer implementation

## Implementation problems

### How to create custom models?

At the end I decided to rely solely on `torch.nn.Module`. This got rid of a lot
of HF code, which IMHO is great for storing the model, learning about it and
quickly using it but not for experimentation. This is because to write HF models
one needs to do a lot of copying and writing very verbose code.

So the steps are:
- download HF model
- put it inside custom `torch.nn.Module`
- work only with the custom module
- save only the state dict

# Evaluation

## IMDb classification

The smaller model consumed somewhere in the neighbourhood of 7600 MB of vmem.
The larger model ran out of memory (with bs=1 and gradient_checkpointing) almost
at the very beginning of training.

### Test set evaluations

Using the default CLS head, which is:

- FC, 768 dim
- dropout, 0.1
- tanh,
- dropout, 0.1
- FC

and the parameters:

```yaml
epochs: 10
label_smoothing: 0.1
warmup_steps: 0.1 * train_size
batch_size: 1
classifier_dropout: 0.1
```
**we achieved 0.9506 accuracy on test set** after 5-7 epochs (125k-175k steps
with grad_accumulation step set to 16).

Overfitting seemed to happen after 7 epochs, but the validation score was
fairly consistent throughout the training (above 0.9433).


### HP search

One search results (4 epochs):

- the differences of accuracies were really minor
- `relu` helps
- reducing hidden dimension of classification head is the only really bad
  decision
- smaller/larger dropout on the classifier caused peaks in achieved accuracy in
  2./3. epoch, but 0.1 was overall the best.

| label_smoothing  | classifier_dropout | classifier_activation | classifier_dim | val. accuracy | val. loss |
| ---------------- | ------------------ | --------------------- | -------------- | ------------- | --------- |
|                  |                    | relu                  |                | 0.95120       | 0.20479   |
| 0.1              | 0.1                | tanh                  | 768.0          | 0.95099       | 0.19709   |
|                  | 0.0                |                       |                | 0.94800       | 0.24426   |
|                  | 0.5                |                       |                | 0.94679       | 0.30623   |
|                  |                    | 250.0                 |                | 0.94459       | 0.25216   |
| .0               |                    |                       |                | 0.94440       | 0.27785   |
|                  |                    | 50.0                  |                | 0.94340       | 0.24950   |
