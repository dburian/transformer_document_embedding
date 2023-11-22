import logging

import torch

_activations_by_name = {
    "relu": torch.nn.ReLU,
    "linear": torch.nn.Identity,
    "sigmoid": torch.nn.Sigmoid,
    "softmax": torch.nn.Softmax,
}


def get_activation(activation_name: str) -> torch.nn.Module:
    activation_name = activation_name.lower()
    if activation_name not in _activations_by_name:
        logging.error(
            "Activation %s not found. Possible activation names are: %s",
            activation_name,
            ", ".join(_activations_by_name.keys()),
        )
        raise ValueError("Invalid activation name.")

    return _activations_by_name[activation_name]
