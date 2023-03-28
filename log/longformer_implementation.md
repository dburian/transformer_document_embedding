# My Longformer implementation

# Evaluation

## IMDb classification

The classification setup using only HF `transformers` package, where the
predictions are made using CLS head only, which has global attention. The net
above CLS head is as so:
- FC, 768 dim
- dropout, 0.1
- tanh,
- dropout, 0.1
- FC

With the following hparams we achieved 0.9506 on test set after 5-7 epochs (125k
- 175k steps with grad_accumulation step set to 16):

```yaml
epochs: 10
label_smoothing: 0.1
warmup_steps: 0.1 * train_size
batch_size: 1
classifier_dropout: 0.1
```

Overfitting seemed to happen after 7 epochs, but the validation score was
fairly consistent throughout the training (above 0.9433).
