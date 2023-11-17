import torch

from transformer_document_embedding.utils.torch.net_helpers import get_activation


class ClsHead(torch.nn.Sequential):
    def __init__(
        self,
        *,
        in_features: int,
        hidden_features: int,
        hidden_activation: str,
        hidden_dropout: float,
        out_features: int,
    ) -> None:
        layers = []
        if hidden_features > 0:
            layers.append(torch.nn.Linear(in_features, hidden_features))
            layers.append(get_activation(hidden_activation)())
            if hidden_dropout > 0:
                layers.append(torch.nn.Dropout(hidden_dropout))

        last_in_features = in_features if hidden_features == 0 else hidden_features
        layers.append(torch.nn.Linear(last_in_features, out_features))

        super().__init__(*layers)
