# LongformerStudent results

## Grid searches

### Just contextual loss

- loss:
    - mse
    - cos
- grad_accumulation_steps (incompatible grid searches):
    - 8
    - 32
- masking loss: (I need two datasets for this in order for the two to be
  comparable -- incompatible grid searches)
    - only small inputs should output loss (`contextual_max_len = 384`)
    - all intpus output loss (`contextual_max_len = inf`)

Each training should run like 3h, validation every 45mins for 10mins. This translates to
limits:
- train: 1400 batches, batch_size = 3
- validation: every 350 batches, batch_size = 3, (77.77~)80 batches

###
