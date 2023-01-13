from __future__ import annotations

import importlib
import logging
import os
from datetime import datetime
from typing import Any

import pkg_resources
import yaml

MODEL_MODULE_PREFIX = "transformer_document_embedding.models"
TASK_MODULE_PREFIX = "transformer_document_embedding.tasks"
CONF_REQUIRED_FIELDS = [
    (["tde_version"], "version of transformer_document_embedding package"),
    (
        ["model", "module"],
        f"model's module path (gets prepended with {MODEL_MODULE_PREFIX})",
    ),
    (
        ["task", "module"],
        f"task's module path (gets prepended with {TASK_MODULE_PREFIX})",
    ),
]


class ExperimentConfig:
    @staticmethod
    def parse(config_file: str, base_results_path: str) -> ExperimentConfig:
        def _check_field_exist(values: dict[str, Any], field_path: list[str]) -> bool:
            field = values
            for breadcrumb in field_path:
                if breadcrumb not in field:
                    return False
                field = field[breadcrumb]
            return True

        with open(config_file, mode="r", encoding="utf8") as file:
            values = yaml.safe_load(file)

            for field_path, desc in CONF_REQUIRED_FIELDS:
                if not _check_field_exist(values, field_path):
                    logging.error(
                        "Parsing of experiment file %s failed. Missing required field:"
                        " %s - %s.",
                        config_file,
                        ".".join(field_path),
                        desc,
                    )

            tde_version = pkg_resources.get_distribution(
                "transformer_document_embedding"
            ).version
            if values["tde_version"] != tde_version:
                logging.warning(
                    "Experiment %s designed for different version of"
                    " transformer_document_embedding, results might differ."
                    " Experiment's version: %s, current version: %s",
                    config_file,
                    values["version"],
                    tde_version,
                )

            return ExperimentConfig(values, base_results_path)

    def __init__(self, config_values: dict[str, Any], base_results_path: str) -> None:
        self.values = config_values
        self._exp_path = None
        self._model_path = None
        self.base_results_path = base_results_path

    @property
    def experiment_path(self) -> str:
        if self._exp_path is None:
            self._exp_path = os.path.join(
                self.base_results_path,
                self.values["task"]["module"],
                self.values["model"]["module"],
                f'{datetime.now().strftime("%Y-%m-%d_%H%M%S")}',
            )

            os.makedirs(self._exp_path, exist_ok=True)

        return self._exp_path

    @property
    def model_path(self) -> str:
        if self._model_path is None:
            self._model_path = os.path.join(self.experiment_path, "model")
            os.makedirs(self._model_path, exist_ok=True)

        return self._model_path

    def get_model_type(self) -> type:
        return importlib.import_module(
            MODEL_MODULE_PREFIX + "." + self.values["model"]["module"]
        ).Model

    def get_task_type(self) -> type:
        return importlib.import_module(
            TASK_MODULE_PREFIX + "." + self.values["task"]["module"]
        ).Task

    def save(self) -> None:
        save_path = os.path.join(self.experiment_path, "config.yaml")
        logging.info("Saving experiment config to %s.", save_path)
        with open(save_path, mode="w", encoding="utf8") as file:
            yaml.dump(self.values, file)
