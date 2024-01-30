import logging
import os

import torch

_activations_by_name = {
    "relu": torch.nn.ReLU,
    "linear": torch.nn.Identity,
    "sigmoid": torch.nn.Sigmoid,
    "softmax": torch.nn.Softmax,
}


def get_activation(activation_name: str) -> torch.nn.Module:
    return _get_from_dict(activation_name, _activations_by_name, "Activation")


_normalizations_by_name = {
    "batch": torch.nn.BatchNorm1d,
    "layer": torch.nn.LayerNorm,
}


def get_normalization(normalization_name: str) -> torch.nn.Module:
    return _get_from_dict(normalization_name, _normalizations_by_name, "Normalization")


def _get_from_dict(
    key: str, dict_: dict[str, torch.nn.Module], log_name: str
) -> torch.nn.Module:
    key = key.lower()
    if key not in dict_:
        logging.error(
            "%s %s not found. Possible names are: %s",
            log_name,
            key,
            ", ".join(dict_.keys()),
        )
        raise ValueError("Invalid name.")

    return dict_[key]


def save_model_weights(model: torch.nn.Module, filepath: str) -> None:
    dirpath = os.path.dirname(filepath)
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    torch.save(model.state_dict(), filepath)


def load_model_weights(model: torch.nn.Module, filepath: str, *, strict: bool) -> None:
    state_dict = torch.load(
        filepath,
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(state_dict, strict=strict)
