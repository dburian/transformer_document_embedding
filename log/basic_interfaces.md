[configuration_files]: configuration_files.md

# Basic interfaces

I tried to avoid strict rules and interfaces as I've found I need the
flexibility, while avoiding having script for each single experiment. There are,
however, some things that are common for all the code base. There are:

- Embedding models -- models that produce an embedding for a text
- Heads -- models that transform embeddings into task-specific output and
  compute loss
- Pipeline -- sometimes configurable object that does stuff (training,
  finetuning, evaluation)
- Document Dataset -- object wrapping a HuggingFace dataset that is used for
  training/finetuning/evaluation

There are few scripts that operate with these objects:

- `train` -- trains a model on a dataset, evaluates it, optionally saves it
- `evaluate` -- loads saved embedding models, and with finetuned head
  evaluates it on multiple tasks
- `hp_search` -- does a grid search or one search over parameters. With each
  option it does pretty much the same as `train`
- `generate_emebeddings` -- trains a model on a dataset and generates embeddings
  for it

## Interfaces in detail

### Embedding models

Since I don't want to be limited by what I can do with a model I've kept the
interface minimal:

- `generate_embeddings` -- generates embeddings for given dataset
- `save_weights`/`load_weights` -- saving/loading the model

Everything else must be done via a specialized pipeline. This is almost as if
I had a script for each model, but much better since the relationship of model
to pipeline can be n-to-n. Plus the pipeline defines the only thing that really
changes and doesn't need to specify all the params and high-level logic.

The models should be as dumb as possible to allow pipelines to operate directly
on them.


### Heads

All the heads are just `torch.nn.Module`s that accept serializable arguments.

### Pipeline

Here I would rather not to state any strict rules or facts. But currently there
are

- training pipelines which are fully configurable,
- head-finetuning pipelines which are chosen based on dataset and have configurable parameters and
- evaluation pipelines which are also chosen based on the dataset, but do not
  have configurable parameters

The reasons are obvious:

- multiple trainings can be done with single model, maximal flexibility needed
- finetuning heads is largely the same once we know the type of task and have
  the embeddings of the model
- evaluation should be done in a same manner for given type of task

### Document dataset

Just a class that serves as a dataset creation class. Everywhere else
"dataset" means `Dataset` from hugging face `datasets` library.


### Configuration

Do not define default values for arguments -- configuration files are then
incomplete and one has to search back to the default value that was default when
the configuration file was created. Instead leave all but helper arguments as
obligatory to force the configuration file to include all values necessary


## History & process

The initial though behind an evaluation environment was to unify models and
tasks so we can easily evaluate particular task with particular model. As the
work went on I learned:

1. I would have to write more code to achieve that.
2. Most models come with a somewhat user-friendly framework already, so the
   amount of code required to run each model is fairly small. This means there
   is small need for unification for the purposes of avoiding code duplication.

The result was that each model had several subclasses (which lowered the amount
of duplicated code) each of which was suited for a given task. Ergo it had a
`train` method which assumed a given type of training data. This was nice in the
sense that training scripts could operate with any task x model combination, as
long as they matched each other. However, later it became clear that it was
unpractical to have `train` method bound with the model. It is because:

- multiple models are trained equally (e.g. heads), having a single "function"
  would dramatically save on code
- the same model can be trained multiple ways, while this could be achieved with
  having a different subclass, implementing an entire model just for different
  training technique is bad,
- separating heads and models became a necessity as I needed to do a bunch of
  operations on the level of the script with separate head or separate model.
  E.g. saving a model, loading a model with different head, finetuning a given
  head on some arbitrary model, ... To do this I would need subclasses of each
  model that would implement in its `train` method all these deviations from
  common training.

This is where we are now. We have pipelines that implement all the things we can
do with a model. We keep the model classes lean and the pipeline's interfaces
set but ready to change so we are not limited in any regard.
