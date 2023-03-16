import torch

import transformer_document_embedding.utils.torch as torch_utils


class ClsHead(torch.nn.Sequential):
    def __init__(
        self,
        *,
        in_features: int,
        hidden_features: int,
        hidden_activation: str,
        hidden_dropout: float,
        num_classes: int,
    ) -> None:
        layers = []
        if hidden_features > 0:
            layers.append(torch.nn.Linear(in_features, hidden_features))
            layers.append(torch_utils.get_activation(hidden_activation)())
            if hidden_dropout > 0:
                layers.append(torch.nn.Dropout(hidden_dropout))

        last_in_features = in_features if hidden_features == 0 else hidden_features
        if num_classes == 2:
            layers.append(torch.nn.Linear(last_in_features, 1))
            layers.append(torch.nn.Sigmoid())
        else:
            layers.append(torch.nn.Linear(last_in_features, num_classes))

        super().__init__(*layers)
