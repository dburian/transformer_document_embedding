
# Experiment configuration and grid search configuration files

To make experiments fully repeatable I envisioned that each experiment will be
described by a YAML file **fully describing** the experiment being done.

While defining experimental setup using separate file is handy for defining the
context of given experiment, it would be cumbersome to use these files when
searching for best parameters. To help with this scenario, there are also grid
search configuration YAML files. These complement the experiment config files --
they cannot be used alone.

## Experiment config's syntax

Aims to fully describe experiment so that it is fully reproducible.

```YAML
tde_version: <transformer_document_embedding's version>

model:
    module: <Model's module name inside tde.models>
    kwargs: <will be parsed as a dict and given to the Model's constructor>

task:
    module: <Task's module name inside tde.tasks>
    kwargs: <will be parsed as a dict and given to the Task's constructor>
```

## Grid search config's syntax

Each grid search configuration specifies all values to try. From each
combination of values a new experiment configuration is created after updating
given experiment configuration with grid searched values.

```YAML
<path.to.value.inside.experiment.config>:
    - <value 1>
    - <value 2>
    - <value 3>
```

So for:

```YAML
model.kwargs.epochs:
    - 10
    - 50
model.kwargs.learning_rate:
    - 1.0e-3
    - 1.0e-4
```

The following configurations would be merged into the reference experiment
configs:

```YAML
model:
    kwargs:
        epochs: 10
        learning_rate: 1.0e-3
```

```YAML
model:
    kwargs:
        epochs: 50
        learning_rate: 1.0e-3
```

```YAML
model:
    kwargs:
        epochs: 10
        learning_rate: 1.0e-4
```

```YAML
model:
    kwargs:
        epochs: 50
        learning_rate: 1.0e-4
```
