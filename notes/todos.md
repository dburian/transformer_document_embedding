# Next consultation

torch lightning and similar solutions
- weight decay for non layer-normalizations layers?

# TODOs

## Implementation

- [ ] Validation loss, early stopping, save best model

## Experiments

- [ ] Properly finetune SBERT on IMDB -- have best SBERT checkpoint
    - pull from repo
    - try training 3 variations until 10 epochs
        - best -- 25/0.1/0.5
        - no hidden -- 0/0.1/0
        - no label smoothing -- 25/0/0.5
- [ ] Finetune Longformer on IMDB -- have best Longformer checkpoint

## Research

- [ ] STS and Spearman's correlation


## Cleanup

- [ ] Split pyproject installations to optional dependencies based on framework
- [ ] Transform `TFClassificationHead` into `tf.keras.Model`



---
# Problems

## problem: Longformer OOF

- what: current training causes OOM, even with batch_size = 1
- why: I should try to train Longformer with the hardware I have, maybe I will
  get better. Nevertheless I should try.
- status: implementation

<!---{{{1--->
### solution

#### fp16

Sets the dtype for layers created from here after. To be precise:
    - computations in float16
    - variables in float32

This speeds up computations and reduces storage on more recent NVIDIAs and TPUs.

```python
mixed_precision.set_global_policy('mixed_float16')
```

Loss scaling also happens to avoid underflow in backward pass (multiplying loss
and then dividing gradients).

When tried *caused errors*, probably the TF model is not ready for fp16.

#### gradient checkpoints

Untrivial: https://arxiv.org/pdf/1604.06174.pdf
Just use:
```python
TFAutoModelForSequenceClassification.from_pretrained("allenai/longformer-base-4096",
gradient_checkpointing=True)
```

If used as above it *does nothing*, because *gradient checkpointing is not
implemented in tf version of the model*.

*As far as I know, there is no way to make gradient checkpointing work out of
the box with TF*

*Ergo I really need to use PyTorch with Longformer*

[How to train large models efficiently with Torch and HF Trainer](https://huggingface.co/docs/transformers/perf_train_gpu_one)
Disadvantage: I do not know what is happening under the hood.

So we need to **use pytorch** to play with Longformer.

### implementation

The solution is to use custom training to have everything under control.

The custom training loop works, I just need to rewrite it to a
`ExperimentalModel`.

<!---1}}}--->


## problem: Finetune SBERT

- what: get best out of SBERT for IMDB
- why: we need to be able to say what were the best params

<!---{{{1--->
### solution

- implement logging

We would like to log loss, train metrics after every epoch.

We need to implement custom evaluators for each loss and train metric.

For accuracy I might find inspiration in `LabelAccuracyEvaluator`.

- write notebook to play with sbert, verify stuff
- REFACTOR, commit
- implement torch clas. head
- logging of vmem
- test vmem, batch_size
- test saving and loading
- prepare grid search
- submit aic job

### implementation

Problem of using custom pytorch training script using HF transformers or just
sentence-transformers package.

sentence-transformers package definitely makes some things easy -- model
construction, saving, loading, but also makes me implementing some things twice
(cls head, loss function). Let's look at HF transformers custom training loops
first though.

- won't HF Trainer learn faster?

<!---1}}}--->


## problem: PyTorch training in `transformers` package

- what: Learn how to train pytorch models inside `transformers` package
- why: Most probably, my own model will be pytorch model

<!---PyTorch training in `transformers` package {{{1!--->
### solution

There are several solutions:

- `transformers` Trainer API
- DIY simple loop, does all the things I want it to do
- PyTorch Lightning -- creates almost Tensorflow-like API above just PyTorch

Implementation of HF Trainer seems really complicated and I do not want to
attempt to recreate it.

The question is whether I can do a simple loop which does all I want without all
the necessary hussle caused by the introduction of another framework-like
library.

Let's create a notebook and play with it. Maybe then compare speeds of custom
loop and what can I do with Trainer.

My guess is that Trainer will be quicker both to implement and during runtime.

The more I read about the issue the more I realise the following:

- writing your custom training loop sucks -- You need to reinvent the wheel.
  Additionally you can easily deploy your model totally independently of PyTorch
  lightning (in pure PyTorch) i.e. use PL only for training.

- HF trainer is too complicated. It is hard to know what is going on and it is
  not documented anywhere.

- The only pro of DIY is I will understand pytorch more. However we need to take
  care of:
    - device
    - evaluations/logging callbacks
    - gradient checkpointing, accumulation, mixed precision
    - learning rate decay
    - freezing/unfreezing layers
    - padding and attention mask

### implementation

<implementation problems and solutions>
<!---1}}}--->


## problem: Define good document embeddings

- what: Define and describe what makes document embedding good
- why: This is the central point of my thesis. I will reference it throughout.

<!---{{{1--->
### solution

Supervises gave me already some ideas:

- embedding should be continuous vector -- we want a set of numbers with
  possibly infinite precision. Additionally to allow models to infer different
  meaning for different positions, we require the set to be numbered.

Somehow we would like to define that the embeddings should capture:

- structure
- semantics
- word choice

Ideas from other papers:
- LASER:
    - multilingual closeness: the representations of the same sentence for
      different languages should be as similar as possible;
    - semantic closeness: similar sentences should be also close in the
      embeddings space, ie. sentences conveying the same meaning, but not
      necessarily the syntactic structure and word choice;
    - preservation of content: sentence representations are usually used in the
      context of a task, eg. classification, multilingual NMT or semantic
      relatedness. This requires that enough information is preserved in the
      representations to perform the task;
    - scalability to many languages: it is desirable that the metric can be
      extended to many languages without important computational cost or need
      for human labeling of data

### implementation

<implementation problems and solutions>

<!---1}}}--->

## problem: Lack of validation data

- what: We miss validation data in training.
- why: To see if we are overfitting or not

<!---{{{1--->
### solution

Testing models requires us to get the best result we can get. Without us knowing
if we overfit or not we cannot know if the parameters are good and we just
trained too long or the parameters are bad.

I currently miss the following features:
- being able to track validation loss
- save the best model in terms of validation loss
- early stop when validation loss stops decreasing

These features cannot be implemented by the `run-experiment` script only. The
models need to take care of this. However, models are the repetetive part of my
code. We would like to minimize code there and move it to reusable parts
(`torch.nn.Module`s). One way would be to have different base classes to support
Tensorflow, PyTorch, ..? which would implement some of the functionality.

### implementation

<implementation problems and solutions>

<!---1}}}--->
