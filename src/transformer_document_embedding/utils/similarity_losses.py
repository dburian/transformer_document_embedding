"""All similarity-based losses.

Semi-required interface:

- Should support masking of inputs to use them in `StructuralContextualHead`.
- All return dicts (for consistency)
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from typing import Optional


class MaxMarginalsLoss(torch.nn.Module):
    def __init__(
        self, dissimilarity_fn: torch.nn.Module, lam: float, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self._dissimilarity_module = dissimilarity_fn
        self.lam = lam

    def dissimilarity_fn(self, *args, **kwargs) -> torch.Tensor:
        return self._dissimilarity_module(*args, **kwargs)["loss"]

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if mask is not None:
            # Assume shape (batch_size, 1) or same mask for all dimensions of
            # given input
            batch_mask = mask[:, 0]
            batch_idxs = batch_mask.nonzero().squeeze()
            outputs = outputs.index_select(dim=0, index=batch_idxs)
            targets = targets.index_select(dim=0, index=batch_idxs)

        positive = self.dissimilarity_fn(outputs, targets)

        negative = torch.tensor(0, device=outputs.device, dtype=outputs.dtype)
        batch_size = outputs.size(0)
        for shifts in range(1, batch_size):
            negative -= self.dissimilarity_fn(outputs, targets.roll(shifts, dims=0))

        # compute mean, to be independent of batch size
        negative /= batch_size - 1
        negative *= self.lam

        return {
            "loss": positive + negative,
            "marginals_positive": positive,
            "marginals_negative": negative,
        }


class MaskedCosineDistance(torch.nn.CosineSimilarity):
    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        cos_dist = 1 - super().forward(x1, x2)

        if mask is None:
            return {"loss": cos_dist.mean()}

        # Cos dist has dimension of 1
        cos_dist *= mask.squeeze()
        total_weight = mask.sum()

        loss = (
            cos_dist.sum() / total_weight
            if total_weight > 0
            else torch.zeros_like(cos_dist)
        )
        return {"loss": loss}


class MaskedMSE(torch.nn.MSELoss):
    def __init__(self) -> None:
        super().__init__(reduction="none")

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        mse = super().forward(x1, x2)

        if mask is None:
            return {"loss": mse.mean()}

        mse *= mask
        loss = mse.sum() / mask.sum()

        return {"loss": loss}


def create_sim_based_loss(loss_type: str, **kwargs) -> torch.nn.Module:
    if loss_type == "mse":
        return MaskedMSE()
    elif loss_type == "cos_dist":
        return MaskedCosineDistance()
    elif loss_type.startswith("max_marginals"):
        max_marginals_lam = kwargs.get("max_marginals_lam", None)
        assert (
            max_marginals_lam is not None
        ), "To use max-marginals loss `max_marginals_lam` needs to be set."

        dissimilarity = None
        if loss_type == "max_marginals_mse":
            dissimilarity = MaskedMSE()
        elif loss_type == "max_marginals_cos_dist":
            dissimilarity = MaskedCosineDistance()

        assert dissimilarity is not None, "Unknown `loss_type`."

        return MaxMarginalsLoss(
            dissimilarity_fn=dissimilarity,
            lam=max_marginals_lam,
        )

    raise ValueError("Unknown loss type: {}".format(loss_type))
