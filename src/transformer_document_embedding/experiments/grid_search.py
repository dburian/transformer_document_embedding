from __future__ import annotations
import yaml
from copy import deepcopy
from transformer_document_embedding.experiments import ExperimentConfig

from typing import Any, Iterable


class GridSearch:
    @classmethod
    def from_yaml(cls, gs_config_path: str) -> GridSearch:
        with open(gs_config_path, mode="r", encoding="utf8") as gs_file:
            return cls(yaml.safe_load(gs_file))

    def __init__(self, values: dict[str, list[Any]]) -> None:
        self._values = values

    def based_on(
        self, reference_config: ExperimentConfig
    ) -> Iterable[ExperimentConfig]:
        for combination in self._all_combinations(self._values):
            new_values = deepcopy(reference_config.values)
            for gs_key, gs_value in combination.items():
                self._apply_gs_value(new_values, gs_key, gs_value)

            yield ExperimentConfig(new_values, reference_config.base_results_path)

    def _apply_gs_value(
        self, exp_config: dict[str, Any], gs_key: str, gs_value: Any
    ) -> None:
        field = exp_config
        path = gs_key.split(".")
        path, last_field_name = path[:-1], path[-1]
        for next_field in path:
            if next_field not in field:
                field[next_field] = {}
            field = field[next_field]

        field[last_field_name] = gs_value

    def _all_combinations(
        self, gs_values: dict[str, list[Any]]
    ) -> Iterable[dict[str, Any]]:
        if len(gs_values.keys()) == 0:
            yield {}
            return

        key = next(iter(gs_values.keys()))
        values = gs_values.pop(key)
        for combination in self._all_combinations(gs_values):
            for value in values:
                combination[key] = value
                yield combination
