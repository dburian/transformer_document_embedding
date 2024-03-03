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
        # TODO: Rename to negatives_lam
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
            # negative -= torch.clip(
            #     self.dissimilarity_fn(outputs, targets.roll(shifts, dims=0)),
            #     max=1e-1,
            # )

        # compute mean, to be independent of batch size
        negative /= batch_size - 1
        negative *= self.lam

        return {
            "loss": positive + negative,
            "marginals_positive": positive,
            "marginals_negative": negative,
        }


class ContrastiveLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

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

        logits = outputs @ targets.T

        logits /= torch.linalg.vector_norm(outputs, dim=1).unsqueeze(-1)
        logits /= torch.linalg.vector_norm(targets, dim=1).unsqueeze(0)

        labels = torch.arange(outputs.size(0), device=logits.device)
        xentropy = torch.nn.functional.cross_entropy(logits, labels)

        return {"loss": xentropy}


class MaskedCosineDistance(torch.nn.CosineSimilarity):
    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        cos_dist = 1 - super().forward(x1, x2)

        return {
            "loss": cos_dist.mean()
            if mask is None
            else masked_mean(cos_dist, mask.squeeze())
        }


class MaskedMSE(torch.nn.MSELoss):
    def __init__(self) -> None:
        super().__init__(reduction="none")

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        mse = super().forward(x1, x2)

        return {"loss": mse.mean() if mask is None else masked_mean(mse, mask)}


class MaskedHuber(torch.nn.HuberLoss):
    def __init__(self, delta: float = 1) -> None:
        super().__init__(reduction="none", delta=delta)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        huber = super().forward(x1, x2)

        return {"loss": huber.mean() if mask is None else masked_mean(huber, mask)}


class MaskedL1(torch.nn.L1Loss):
    def __init__(self) -> None:
        super().__init__(size_average=None, reduce=None, reduction="none")

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        l1 = super().forward(x1, x2)

        return {"loss": l1.mean() if mask is None else masked_mean(l1, mask)}


def masked_mean(input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Computes mean of input * mask."""
    input *= mask

    total_weight = mask.sum()
    return (input.sum() / total_weight) if total_weight > 0 else torch.zeros_like(input)


_simple_losses = {
    "mse": MaskedMSE,
    "cos_dist": MaskedCosineDistance,
    "huber": MaskedHuber,
    "l1": MaskedL1,
}


def create_sim_based_loss(loss_type: str, **kwargs) -> torch.nn.Module:
    if loss_type in _simple_losses:
        return _simple_losses[loss_type]()
    elif loss_type.startswith("max_marginals"):
        max_marginals_lam = kwargs.get("max_marginals_lam", None)
        assert (
            max_marginals_lam is not None
        ), "To use max-marginals loss `max_marginals_lam` needs to be set."

        dissimilarity_name = loss_type[len("max_marginals") + 1 :]
        dissimilarity = _simple_losses.get(dissimilarity_name, None)

        assert dissimilarity is not None, "Unknown `loss_type`."
        dissimilarity_fn = dissimilarity()

        return MaxMarginalsLoss(
            dissimilarity_fn=dissimilarity_fn,
            lam=max_marginals_lam,
        )
    elif loss_type == "contrastive":
        return ContrastiveLoss()

    raise ValueError("Unknown loss type: {}".format(loss_type))
