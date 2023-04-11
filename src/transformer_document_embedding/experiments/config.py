from __future__ import annotations

import importlib
import logging
import os
from datetime import datetime
from typing import Any

import pkg_resources
import tensorflow as tf
import yaml
from tensorboard.plugins.hparams import api as hp

MODEL_MODULE_PREFIX = "transformer_document_embedding.baselines"
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
    def from_yaml(config_file: str, base_results_dir: str) -> ExperimentConfig:
        with open(config_file, mode="r", encoding="utf8") as file:
            values = yaml.safe_load(file)

            return ExperimentConfig(values, base_results_dir)

    def __init__(
        self,
        config_values: dict[str, Any],
        base_results_dir: str,
    ) -> None:
        def _check_field_exist(values: dict[str, Any], field_path: list[str]) -> bool:
            field = values
            for breadcrumb in field_path:
                if breadcrumb not in field:
                    return False
                field = field[breadcrumb]
            return True

        for field_path, desc in CONF_REQUIRED_FIELDS:
            assert _check_field_exist(config_values, field_path), (
                "Invalid experiment config. Missing required field:"
                f" {'.'.join(field_path)} - {desc}."
            )

        tde_version = pkg_resources.get_distribution(
            "transformer_document_embedding"
        ).version
        if config_values["tde_version"] != tde_version:
            logging.warning(
                "Experiment designed for different version of"
                " transformer_document_embedding, results might differ."
                " Experiment's version: %s, current version: %s",
                config_values["tde_version"],
                tde_version,
            )

        self.values = config_values
        self.base_results_dir = base_results_dir
        self._model_path = None
        self._exp_path = None

    @property
    def experiment_path(self) -> str:
        if self._exp_path is None:
            self._exp_path = os.path.join(
                self.base_results_dir,
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
        return self._import_type(
            MODEL_MODULE_PREFIX + "." + self.values["model"]["module"]
        )

    def get_task_type(self) -> type:
        return self._import_type(
            TASK_MODULE_PREFIX + "." + self.values["task"]["module"]
        )

    def save(self) -> None:
        save_path = os.path.join(self.experiment_path, "config.yaml")
        logging.info("Saving experiment config to %s.", save_path)
        with open(save_path, mode="w", encoding="utf8") as file:
            yaml.dump(self.values, file)

    def log_hparams(self) -> None:
        with tf.summary.create_file_writer(self.experiment_path).as_default():
            hparams = flatten_dict(self.values)
            hp.hparams(
                hparams,
                os.path.relpath(self.experiment_path, start=self.base_results_dir),
            )

            tf.summary.flush()

    @classmethod
    def _import_type(cls, type_path: str) -> type:
        try:
            module_path, type_name = type_path.split(":")
            module = importlib.import_module(module_path)
            return getattr(module, type_name)
        except Exception as exc:
            raise ValueError(
                f"{ExperimentConfig._import_type.__name__}: invalid type path"
                f" '{type_path}'."
            ) from exc


def flatten_dict(structure: dict[str, Any]) -> dict[str, Any]:
    flatten = {}
    for key, value in structure.items():
        if isinstance(value, dict):
            for flatten_key, flatten_value in flatten_dict(value).items():
                flatten[f"{key}.{flatten_key}"] = flatten_value
        else:
            flatten[key] = value

    return flatten
