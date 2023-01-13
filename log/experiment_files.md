
# Experiment files

To make experiments fully repeatable I envisioned that each experiment will be
described by a YAML file **fully describing** the experiment being done.

## Syntax

```YAML
tde_version: <transformer_document_embedding's version>

model:
    type: <Model's package name inside tde>
    kwargs: <will be parsed as a dict and given to the Model's constructor>

task:
    type: <Task's package name inside tde>
    kwargs: <will be parsed as a dict and given to the Task's constructor>
```
