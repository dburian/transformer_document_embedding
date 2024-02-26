# No `from __future__ import annotatinos` ruins loading of dataclasses from
# dict and is the whole point of this module
from abc import abstractmethod
from dataclasses import dataclass, field


from dacite import from_dict

from transformer_document_embedding.datasets.document_dataset import DocumentDataset

from transformer_document_embedding.models.embedding_model import EmbeddingModel
from transformer_document_embedding.scripts.utils import init_type
from typing import Any, Optional, TypeVar
from torch.nn.modules import Module

# For compatibility with python3.10
Self = TypeVar("Self", bound="BaseValuesSpec")


@dataclass(kw_only=True)
class BaseValuesSpec:
    @classmethod
    def from_dict(cls: type[Self], dct: dict[str, Any]) -> Self:
        try:
            return from_dict(cls, dct)
        except TypeError as e:
            raise TypeError("Configuration in invalid format.") from e


@dataclass(kw_only=True)
class ModuleSpec(BaseValuesSpec):
    module: str
    module_prefix: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def initialize(self) -> Any:
        pass


@dataclass(kw_only=True)
class EmbeddingModelSpec(ModuleSpec):
    module_prefix: str = "transformer_document_embedding.models"

    def initialize(self) -> EmbeddingModel:
        return init_type(self)


@dataclass(kw_only=True)
class DatasetSpec(ModuleSpec):
    module_prefix: str = "transformer_document_embedding.datasets"

    def initialize(self) -> DocumentDataset:
        return init_type(self)


@dataclass(kw_only=True)
class HeadSpec(ModuleSpec):
    module_prefix: str = "transformer_document_embedding.heads"

    def initialize(self) -> Module:
        return init_type(self)


@dataclass(kw_only=True)
class PipelineSpec(BaseValuesSpec):
    kind: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class ExperimentSpec(BaseValuesSpec):
    model: EmbeddingModelSpec
    head: Optional[HeadSpec] = None
    dataset: DatasetSpec
    train_pipeline: Optional[PipelineSpec] = None


@dataclass(kw_only=True)
class CrossValidateSpec(BaseValuesSpec):
    split: str
    num_folds: int


@dataclass(kw_only=True)
class EvaluationSpec(BaseValuesSpec):
    dataset: DatasetSpec
    head: Optional[HeadSpec] = None
    finetune_pipeline_kwargs: dict[str, Any] = field(default_factory=dict)
    cross_validate: Optional[CrossValidateSpec] = None


@dataclass(kw_only=True)
class EvaluationsSpec(BaseValuesSpec):
    evaluations: dict[str, EvaluationSpec]


@dataclass(kw_only=True)
class EvaluationInstanceSpec(EvaluationSpec):
    model: EmbeddingModelSpec
