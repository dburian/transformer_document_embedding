[configuration_files]: configuration_files.md

# Basic interfaces

This document records the basic interfaces that are used in scripts. The idea is
to abstract away the peculiarities of each model and task such that they can be
combined almost arbitrarily.

There are two basic interfaces:

- `ExperimentalTask`, and
- `ExperimentalModel`.

Using these interfaces scripts can load up a dataset and a model, train the
model on the dataset, evaluate it and save everything.

Its clear that each task is going to have some type (e.g. classification,
retrieval based, classification of pairs, ...). This needs to be mirrored on the
side of the models (e.g. classification model cannot be used when generating
embeddings and vice versa). *This is not explicitly mentioned in code*.

## `ExperimentalTask`

Wraps a dataset and defines the method how it is evaluated.

## `ExperimentalModel`

Wraps a model and its training/prediction setup.

- `save_weights`/`load_weights` -- saves/loads model's weights in a manner such
  that the same model of different type will be able to reuse weights from shared
  architecture (e.g. a Longformer classifier should be able to reuse the weights
  of the Longformer, with weights of the classification head randomly
  initialized).

### Notes

- all arguments that are required only for the training should go into
  `train_kwargs` in [configuration files][configuration_files]

## General implementation advice

- do not define default values for arguments -- configuration files are then
  incomplete and one has to search back to the default value that was default
  when the configuration file was created. Instead leave all but helper
  arguments as obligatory to force the configuration file to include all values
  necessary


## Thought process

The initial though behind an evaluation environment was to unify models and
tasks so we can easily evaluate particular task with particular model. As the
work went on I learned:

1. I would have to write more code to achieve that.
2. Most models come with a somewhat user-friendly framework already, so the
   amount of code required to run each model is fairly small. This means there
   is small need for unification for the purposes of avoiding code duplication.

The result is we have some unification, but not much since we are trying to have
lots of experiments, not a really though-out framework for evaluating models.
The unification is there to have scripts that can operate on multiple model-task
pairs. There is still plenty of value in that:

1. we can move from a model implementation to trying it out really quickly,
2. easy arguments loading, which are also automatically saved
3. avoiding *some* code repetition (though not as much as initially planned)
4. automatic and systematic saving of experiment results and logging

### The Why-s

- **Why this does not matter that much.**

The main goal is to write experiments, not code. Do not over-optimize for stuff
that you'll never need.

- Separating pure models and experimental models

Pure models are easily portable. ExperimentalModels are not. This is the main
reason why the code is separated as so. Models can be written in pytorch or
tensorflow, ExperimentalModels also act as an adaptors.

- importing models and tasks by package path

Can be done differently (e.g. w/ dictionary), but this requires the least amount
of extra code. This is more flexible at the cost of being more error-prone.

- evaluation in model vs evaluation in task

I decided against evaluation in model, mainly because I want each model to be
evaluated equally. This also minimizes the code which should evaluate
predictions, which would have to be in every model. Also there could be problems
with getting the evaluations out of the models.

- all data given to model when training

This is from experience. Some models do unsupervised training on all inputs.

- in defence of HF dataset format

HF dataset is great because:
    - it has all kinds of useful transformations,
    - it can be loaded from all kinds of files,
    - it is sharable (so if some dataset is not in this format I can create it and
    publish it)
    - it should work with all transformers in HF out of the box
    - it has documentation (though hard to navigate)
    - caching

- Defining task's and model's type in code:

It is not explicitly mentioned in the code that each task and model has certain
type and that they need to match. This is because enforcing matching of types of
model and task currently wouldn't contribute to the result of this thesis and
would only add bulk. Instead it is up to the researcher that runs the scripts to
use to correct model for given task.
