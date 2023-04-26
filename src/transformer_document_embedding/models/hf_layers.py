from typing import Optional, cast

import torch
from transformers import PretrainedConfig
from transformers.activations import get_activation


class ClassificationConfigMixin(PretrainedConfig):
    def __init__(
        self,
        classifier_activation: Optional[str] = None,
        classifier_hidden_size: Optional[int] = None,
        classifier_dropout_prob: Optional[float] = None,
        **kwargs,
    ) -> None:
        # Due to lacking interface in HF's PreTrainedConfig, we take the
        # approach: "If bad config value is set, something will crash later on."
        if classifier_hidden_size is None:
            classifier_hidden_size = kwargs.get("hidden_size", 768)

        if classifier_activation is None:
            classifier_activation = kwargs.get("hidden_act", "gelu")

        if classifier_dropout_prob is None:
            classifier_dropout_prob = kwargs.get("hidden_dropout_prob", 0.1)

        super().__init__(
            classifier_hidden_size=classifier_hidden_size,
            classifier_activation=classifier_activation,
            classifier_dropout_prob=classifier_dropout_prob,
            **kwargs,
        )


class PooledConfigMixin(PretrainedConfig):
    def __init__(self, pooler_type: Optional[str] = None, **kwargs) -> None:
        assert (
            pooler_type is None or pooler_type in AVAILABLE_POOLERS
        ), f"pooler_type must be one of {AVAILABLE_POOLERS}"
        self.pooler_type = pooler_type

        super().__init__(
            pooler_type=pooler_type,
            **kwargs,
        )


class MeanPooler(torch.nn.Module):
    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        summed = torch.sum(last_hidden_state * attention_mask[:, :, None], 1)
        row_lengths = torch.sum(attention_mask, 1)
        return summed / row_lengths[:, None]


class SumPooler(torch.nn.Module):
    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sum(last_hidden_state * attention_mask[:, :, None], 1)


AVAILABLE_POOLERS = {
    "sum": SumPooler,
    "mean": MeanPooler,
}


class ClassificationHead(torch.nn.Module):
    """My configurable head for sentence-level classification tasks."""

    def __init__(self, config: ClassificationConfigMixin):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.classifier_hidden_size)
        self.dropout = torch.nn.Dropout(config.classifier_dropout_prob)
        self.out_proj = torch.nn.Linear(
            config.classifier_hidden_size, config.num_labels
        )
        self.activation_fn = get_activation(config.classifier_activation)

    def forward(self, hidden_state: torch.Tensor):
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.dense(hidden_state)
        hidden_state = self.activation_fn(hidden_state)
        hidden_state = self.dropout(hidden_state)
        output = self.out_proj(hidden_state)
        return output
