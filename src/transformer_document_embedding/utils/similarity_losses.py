"""All similarity-based losses.

Semi-required interface:

- Should support masking of inputs to use them in `StaticContextualLoss`.
- Should return a dict, not a Tensor.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from typing import Optional


class ContrastiveLoss(torch.nn.Module):
    def __init__(
        self, dissimilarity_fn: torch.nn.Module, lam: float, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.dissimilarity_fn = dissimilarity_fn
        self.lam = lam

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

        positive = -self.dissimilarity_fn(outputs, targets)

        negative = torch.tensor(0, device=outputs.device, dtype=outputs.dtype)
        batch_size = outputs.size(0)
        for shifts in range(1, batch_size):
            negative += self.lam * self.dissimilarity_fn(
                outputs, targets.roll(shifts, dims=0)
            )

        # compute mean, to be independent of batch size
        negative /= batch_size - 1

        return {
            "loss": positive + negative,
            "contrastive_positive": positive,
            "contrastive_negative": negative,
        }


class MaskedCosineDistance(torch.nn.CosineSimilarity):
    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        cos_dist = 1 - super().forward(x1, x2)

        if mask is not None:
            cos_dist *= mask

        return cos_dist.sum()


class MaskedMSE(torch.nn.MSELoss):
    def __init__(self) -> None:
        super().__init__(reduction="none")

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        mse = super().forward(x1, x2)
        if mask is not None:
            mse *= mask
        return mse.sum()


def create_sim_based_loss(loss_type: str, **kwargs) -> torch.nn.Module:
    if loss_type == "mse":
        return MaskedMSE()
    elif loss_type == "cos_dist":
        return MaskedCosineDistance()
    elif loss_type.startswith("contrastive"):
        contrastive_lam = kwargs.get("contrastive_lam", None)
        assert (
            contrastive_lam is not None
        ), "To use contrastive loss `contrastive_lam` needs to be set."

        dissimilarity = None
        if loss_type == "contrastive_mse":
            dissimilarity = MaskedMSE()
        elif loss_type == "contrastive_cos_dist":
            dissimilarity = MaskedCosineDistance()

        assert dissimilarity is not None, "Unknown `loss_type`."

        return ContrastiveLoss(
            dissimilarity_fn=dissimilarity,
            lam=contrastive_lam,
        )

    raise ValueError("Unknown loss type: {}".format(loss_type))
