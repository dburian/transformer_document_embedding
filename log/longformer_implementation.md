# My Longformer implementation

## Implementation problems

### How to create custom models?

It is better to create custom specialized models such as `LongformerFor...`:
- With custom Longformers the pretrained weights (saved as `state_dict`s) might
  not align.
- Put `LongformerModel` under `self.longformer`. The merging downloaded
  `state_dict` and current `state_dict` is in
  `PreTrainedModel._load_pretrained_model` where the key prefix 'longformer' is
  gotten from `LongformerPreTrainedModel.base_model_prefix`.


### PreTrainedConfig

Approaches:

- Stick to HF code: no autocompletion for config values, need to check values
  (for `None`s) twice (in baseline init and in config's init)
    1. Create custom class, inheriting from `LongformerConfig`.
    2. Call `.from_pretrained` on the custom class, with custom config values as
       kwargs.
    3. If those kwargs are attributes on the initiated custom config, the values
       get assigned. From `configuration_utils`:
```python
config = cls(**config_dict)
...
for key, value in kwargs.items():
    if hasattr(config, key):
        setattr(config, key, value)
```

- Brute force **BEST**: we're maybe ignoring what should happen in `from_dict`
```python
config = CustomConfig(
    my_custom_key=my_custom_value,
    **CustomConfig.get_config_dict('pretrained/config/path')[0]
)
```



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
