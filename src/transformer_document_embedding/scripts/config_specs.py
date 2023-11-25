# No `from __future__ import annotatinos` ruins loading of dataclasses from
# dict and is the whole point of this module
from dataclasses import dataclass, field


from dacite import from_dict
from typing import Any, Self


@dataclass(kw_only=True)
class BaseValuesSpec:
    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> Self:
        try:
            return from_dict(cls, dct)
        except TypeError as e:
            raise TypeError("Configuration in invalid format.") from e


@dataclass(kw_only=True)
class ModuleSpec:
    module: str
    module_prefix: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class ExperimentalModelSpec(ModuleSpec):
    module_prefix: str = "transformer_document_embedding.models"
    train_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class ExperimentalTaskSpec(ModuleSpec):
    module_prefix: str = "transformer_document_embedding.tasks"


@dataclass(kw_only=True)
class ExperimentSpec(BaseValuesSpec):
    model: ExperimentalModelSpec
    task: ExperimentalTaskSpec


@dataclass(kw_only=True)
class NamedExperimentalTaskSpec(ExperimentalTaskSpec):
    name: str


@dataclass(kw_only=True)
class EvaluationSpec(BaseValuesSpec):
    model: ExperimentalModelSpec
    tasks: list[NamedExperimentalTaskSpec]
    evaluated_model_path: str
