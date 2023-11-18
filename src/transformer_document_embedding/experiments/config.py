from __future__ import annotations

from coolname import generate
import re
import importlib
import sys
import logging
import os

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Optional

import pkg_resources
import yaml

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


logger = logging.getLogger(__name__)


class ExperimentConfig:
    @staticmethod
    def from_yaml(
        config_file: str, output_base_path: str, **kwargs
    ) -> ExperimentConfig:
        return ExperimentConfig(
            load_config_values(config_file),
            output_base_path,
            **kwargs,
        )

    def __init__(
        self,
        config_values: dict[str, Any],
        output_base_path: str,
        name: Optional[str] = None,
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

        self._name = name if name is not None else "_".join(generate(2))
        self.values = config_values
        self.output_base_path = output_base_path
        self._model_path = None
        self._exp_path = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def experiment_path(self) -> str:
        if self._exp_path is None:
            self._exp_path = self._construct_experiment_path()
            if os.path.exists(self._exp_path):
                logger.error(
                    "Experiment with the given name already exists at %s",
                    self._exp_path,
                )
                sys.exit(1)
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
            self.values["model"]["module"], module_prefix=MODEL_MODULE_PREFIX
        )

    def get_task_type(self) -> type:
        return self._import_type(
            self.values["task"]["module"], module_prefix=TASK_MODULE_PREFIX
        )

    def save(self) -> None:
        save_path = os.path.join(self.experiment_path, "config.yaml")
        logging.info("Saving experiment config to %s.", save_path)
        with open(save_path, mode="w", encoding="utf8") as file:
            yaml.dump(self.values, file)

    def _construct_experiment_path(self) -> str:
        return os.path.join(
            self.output_base_path,
            self.values["task"]["module"],
            self.values["model"]["module"],
            self._name.replace("/", "-"),
        )

    @classmethod
    def _import_type(
        cls, type_spec: str, *, module_prefix: Optional[str] = None
    ) -> type:
        module_path = module_prefix if module_prefix is not None else ""
        type_name = type_spec
        if ":" in type_spec:
            module_suffix, type_name = type_spec.split(":")
            module_path += "." + module_suffix

        try:
            module = importlib.import_module(module_path)
            return getattr(module, type_name)
        except Exception as exc:
            raise ValueError(
                f"{ExperimentConfig._import_type.__name__}: invalid type path"
                f" '{type_spec}'."
            ) from exc


class HPSearchExperimentConfig(ExperimentConfig):
    """For experiments that are part of hp search."""

    @classmethod
    def _generate_name_from_params(cls, flatten_params: dict[str, Any]) -> str:
        name_parts = []
        for key, value in flatten_params.items():
            short_key = re.sub("([a-zA-Z])[a-zA-Z]+", r"\1", key)
            name_parts.append(f"{short_key}={value}")

        return "-".join(name_parts)

    def __init__(
        self,
        config_values: dict[str, Any],
        output_base_path: str,
        flatten_hparams: dict[str, Any],
    ) -> None:
        super().__init__(
            config_values,
            output_base_path,
            self._generate_name_from_params(flatten_hparams),
        )
        self._flatten_hparams = flatten_hparams

    def _construct_experiment_path(self) -> str:
        return os.path.join(
            self.output_base_path,
            self._name.replace("/", "-"),
        )

    def log_hparams(self) -> None:
        import tensorflow as tf
        from tensorboard.plugins.hparams import api as hp

        with tf.summary.create_file_writer(self.experiment_path).as_default():
            # tf is unable to log NoneTypes
            for key, value in self._flatten_hparams.items():
                if value is None:
                    self._flatten_hparams[key] = "None"
                if isinstance(value, list) or isinstance(value, dict):
                    self._flatten_hparams[key] = str(value)

            hp.hparams(
                self._flatten_hparams,
                os.path.join(os.path.basename(self.output_base_path), self.name),
            )

            tf.summary.flush()


def flatten_dict(structure: dict[str, Any]) -> dict[str, Any]:
    flatten = {}
    for key, value in structure.items():
        if isinstance(value, dict):
            for flatten_key, flatten_value in flatten_dict(value).items():
                flatten[f"{key}.{flatten_key}"] = flatten_value
        else:
            flatten[key] = value

    return flatten


def load_config_values(yaml_path: str) -> dict[str, Any]:
    with open(yaml_path, mode="r", encoding="utf8") as file:
        return yaml.safe_load(file)
