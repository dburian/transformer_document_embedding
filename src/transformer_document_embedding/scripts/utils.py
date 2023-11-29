from __future__ import annotations
import importlib
import os
from dataclasses import asdict

import logging

from typing import Optional

import yaml
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from transformer_document_embedding.scripts.config_specs import (
        BaseValuesSpec,
        ModuleSpec,
    )


def save_config(spec: BaseValuesSpec, path: str) -> None:
    save_path = os.path.join(path, "config.yaml")
    logging.info("Saving config to %s", save_path)

    with open(save_path, mode="w", encoding="utf8") as file:
        yaml.dump(asdict(spec), file)


def load_yaml(yaml_path: str) -> dict[str, Any]:
    with open(yaml_path, mode="r", encoding="utf8") as file:
        return yaml.safe_load(file)


def import_type(type_spec: str, *, module_prefix: Optional[str] = None) -> type:
    module_path = module_prefix if module_prefix is not None else ""
    type_name = type_spec
    if ":" in type_spec:
        module_suffix, type_name = type_spec.split(":")
        module_path += "." + module_suffix

    try:
        module = importlib.import_module(module_path)
        return getattr(module, type_name)
    except Exception as exc:
        raise ValueError(f"invalid type path '{module_path}:{type_name}'.") from exc


def init_type(spec: ModuleSpec) -> Any:
    cls = import_type(spec.module, module_prefix=spec.module_prefix)
    return cls(**spec.kwargs)


def log_results(log_path: str, results: dict[str, float]) -> None:
    import tensorflow as tf

    with tf.summary.create_file_writer(log_path).as_default():
        for name, res in results.items():
            tf.summary.scalar(name, res, step=1)

        tf.summary.flush()


def save_results(results: dict[str, float], out_dir: str) -> None:
    results_filepath = os.path.join(out_dir, "results.yaml")
    logging.info("Saving results to %s.", results_filepath)
    with open(results_filepath, mode="w", newline="", encoding="utf8") as yaml_file:
        yaml.dump(results, yaml_file)
