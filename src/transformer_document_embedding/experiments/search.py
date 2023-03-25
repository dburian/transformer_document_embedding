from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

import yaml

from transformer_document_embedding.experiments import ExperimentConfig


class HyperparameterSearch:
    @classmethod
    def from_yaml(cls, gs_config_path: str) -> HyperparameterSearch:
        with open(gs_config_path, mode="r", encoding="utf8") as gs_file:
            return cls(yaml.safe_load(gs_file))

    def __init__(self, hparams: dict[str, list[Any]]) -> None:
        self._hparams = hparams

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


class GridSearch(HyperparameterSearch):
    def based_on(
        self, reference_config: ExperimentConfig
    ) -> Iterable[ExperimentConfig]:
        for combination in self._all_combinations(self._hparams):
            new_values = deepcopy(reference_config.values)
            for gs_key, gs_value in combination.items():
                self._update_with_hparam(new_values, gs_key, gs_value)

            yield ExperimentConfig(new_values, reference_config.base_results_path)

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
        self, reference_config: ExperimentConfig
    ) -> Iterable[ExperimentConfig]:
        for hparam_key, value_set in self._hparams.items():
            for hparam_value in value_set:
                new_values = deepcopy(reference_config.values)
                self._update_with_hparam(new_values, hparam_key, hparam_value)

                yield ExperimentConfig(new_values, reference_config.base_results_path)
