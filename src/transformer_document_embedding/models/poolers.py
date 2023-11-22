from __future__ import annotations

import torch


class MeanPooler(torch.nn.Module):
    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor, **_
    ) -> torch.Tensor:
        summed = torch.sum(last_hidden_state * attention_mask[:, :, None], 1)
        row_lengths = torch.sum(attention_mask, 1)
        return summed / row_lengths[:, None]


class ClsPooler(torch.nn.Module):
    def forward(self, last_hidden_state: torch.Tensor, **_) -> torch.Tensor:
        return last_hidden_state[:, 0]


class SumPooler(torch.nn.Module):
    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        **_,
    ) -> torch.Tensor:
        return torch.sum(last_hidden_state * attention_mask[:, :, None], 1)


AVAILABLE_POOLERS = {
    "sum": SumPooler,
    "mean": MeanPooler,
    "cls": ClsPooler,
}
