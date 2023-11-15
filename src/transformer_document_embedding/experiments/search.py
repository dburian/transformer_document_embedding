from __future__ import annotations
from abc import abstractmethod

import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Iterable
from coolname import generate

import yaml

from transformer_document_embedding.experiments.config import (
    HPSearchExperimentConfig,
)

if TYPE_CHECKING:
    from typing import Optional


class HyperparameterSearch:
    @classmethod
    def from_yaml(
        cls, gs_config_path: str, output_base_path: str, **kwargs
    ) -> HyperparameterSearch:
        with open(gs_config_path, mode="r", encoding="utf8") as gs_file:
            return cls(yaml.safe_load(gs_file), output_base_path, **kwargs)

    def __init__(
        self,
        hparams: dict[str, list[Any]],
        output_base_path: str,
        name: Optional[str] = None,
    ) -> None:
        self.hparams = hparams
        self.output_base_path = output_base_path
        self.name = name if name is not None else "_".join(generate(2))

    @property
    def experiments_base_dir(self) -> str:
        return os.path.join(self.output_base_path, self.name)

    def _update_with_hparam(
        self, exp_config: dict[str, Any], param_key: str, param_value: Any
    ) -> None:
        field = exp_config
        path = param_key.split(".")
        path, last_field_name = path[:-1], path[-1]
        for next_field in path:
            if next_field not in field:
                field[next_field] = {}
            field = field[next_field]

        field[last_field_name] = param_value

    @abstractmethod
    def based_on(
        self, reference_values: dict[str, Any]
    ) -> Iterable[HPSearchExperimentConfig]:
        raise NotImplementedError()


class GridSearch(HyperparameterSearch):
    def based_on(
        self, reference_values: dict[str, Any]
    ) -> Iterable[HPSearchExperimentConfig]:
        for combination in self._all_combinations(self.hparams):
            new_values = deepcopy(reference_values)
            for gs_key, gs_value in combination.items():
                self._update_with_hparam(new_values, gs_key, gs_value)

            yield HPSearchExperimentConfig(
                new_values,
                self.experiments_base_dir,
                flatten_hparams=deepcopy(combination),
            )

    def _all_combinations(
        self, gs_values: dict[str, list[Any]]
    ) -> Iterable[dict[str, Any]]:
        key = next(iter(gs_values.keys()), None)
        if key is None:
            yield {}
            return

        values = gs_values.pop(key)
        for combination in self._all_combinations(gs_values):
            for value in values:
                combination[key] = value
                yield combination


class OneSearch(HyperparameterSearch):
    def based_on(
        self, reference_values: dict[str, Any]
    ) -> Iterable[HPSearchExperimentConfig]:
        for hparam_key, value_set in self.hparams.items():
            for hparam_value in value_set:
                new_values = deepcopy(reference_values)
                self._update_with_hparam(new_values, hparam_key, hparam_value)

                yield HPSearchExperimentConfig(
                    new_values,
                    self.experiments_base_dir,
                    flatten_hparams={hparam_key: hparam_value},
                )
