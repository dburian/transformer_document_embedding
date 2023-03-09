[annotated_transformer]: http://nlp.seas.harvard.edu/annotated-transformer/
[hf_eff_gpu_train]: https://huggingface.co/docs/transformers/perf_train_gpu_one

# Training transformers manually

From deep learning course:
- when training from scratch:
    - Adam with $\beta_2 = 0.98$
    - learning rate $\alpha = \frac{1}{\sqrt{d_{model}}} min \Big(\frac{1}{\sqrt{step\_num}}, \frac{step\_num}{warmup\_steps} \frac{1}{\sqrt{warmup\_steps}} \Big)$
        - Linear warmup until $warmup\_steps$ to 1, then decay
- for finetuning freeze the base model for first few iterations
    - in pytorch with `torch.nn.Module.requires_grad_(False)`

My hunch that I will forget to:
- call `model.train(True)` before starting training and `model.train(False)` or
  equivalently `model.eval()` during evaluation.

From Trainer:
- `args.weight_decay` weight decay for all weights except biases and all weights
  of LayerNorms
- `args.optim` optimizier with `args.adam_beta1` and `args.adam_beta2`
- linear warmup to 1 then `args.lr_scheduler`
- call `gradient_checkpointing_enable` on model
- for each epoch do
    - `self.training_step()`
    - unscale gradients and clip the gradients to `args.max_grad_norm`
    - optimizer step
    - learning rate scheduler step if optimizer was run (due to gradient scaling
      it may not have run)
    - zero grad
- default `self.training_step()` is:
    - `model.train()`
    - move inputs to device
    - with context `torch.cuda.amp.autocast()` which enables mixed precision,
      compute loss -- model output
    - scale loss using `torch.cuda.amp.GradScaler`
    - `loss.backward()`
- default values:
    ```python
    args.weight_decay = 0
    args.optim = 'adamw_hf'
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999
    args.adam_epsilon = 1e-8
    args.max_grad_norm = 1
    args.lr_scheduler = 'linear'
    args.warmup_steps = 0
    ```

From Torch documentation on mixed precision -- gradient scaling and autocast:
- typical training step:
    ```python
    optimizer.zero_grad()

    with autocast(device_type='cuda', dtype=torch.float16):
        output = model(inputs)
        loss = loss_fn(output, target)

    # Increasing loss to avoid underflow (BTW scaler = GradScaler())
    scaler.scale(loss).backward()

    if gradient_clipping:
        # Unscales optimizer's params in-place
        scaler.unscale_(optimizer)
        troch.nn.utils.clip_grad_norm(model.parameters(), max_norm)

    # Decreasing gradients to equalize for increasing loss
    # If decreasing gradients causes infs or NaNs, optimizer is not run
    # scaler remembers if it unscaled_ gradients earlier
    scaler.step(optimizer)

    # Updates scaling for next iteration
    scaler.update()
    ```

- gradient accumulation w/ scaled gradients
    - the scaling should be done identically for every accumulated gradient
    - clip *accumulated* (unscaled) gradients
    ```python
    with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)
            loss = loss / iters_to_accumulate

    # Accumulate scaled gradients
    scaler.scale(loss).backward()

    if (i + 1) % iters_to_accumulate == 0:
        # may unscale_ here (e.g. for gradient clipping)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    ```

## Glossary

### Gradient accumulation

When training, for each input example we need to store all the activations to be
used in backward pass plus (obviously) the input example itself (or state of the
computation). Lowering batch size would also lower the memory consumption, but
it often also cases slower convergence -- small batches result in a lot of
costly adjustments (sometimes in the opposite directions) to parameters.

The solution is gradient accumulation -- we decrease the batch size (so it fits
into our GPU mem), but accumulate the gradients for several batches.

### Mixed precision

Idea of mixed precision is to lower the precision of certain computations in
order to increase spead and decrease memory consumption at the cost of lower
accuracy of the computation (which seems to not matter). The parameters are
still kept at float32 (to avoid large precision errors), but the computations
themsleves happen in float16.

### Gradient checkpointing

During the backward pass, the activations from the forward pass are needed for
differentiation:

$$
\frac{\partial f(g(x))}{\partial x} = \frac{\partial f(g(x))}{\partial g(x)}
\frac{\partial g(x)}{\partial x}
$$

So the optimizer needs to either recompute the forward pass multiple times
(computation too long) or remember the activations (memory heavy). The
compromise is to select checkpoints in the network where we will remember
activations and recompute the rest from those. Thus saving memory at the expense
of little bit more time.

## References

- `Trainer` implementation in HuggingFace's `transformers`
- Torch docs *'CUDA automatic mixed precision examples'*
- HuggingFace's guide on [Efficient Training on a Single GPU][hf_eff_gpu_train].
- article explaining implementation of a [transformer][annotated_transformer].
