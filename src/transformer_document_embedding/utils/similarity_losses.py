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


def index_nonzero(
    mask: torch.Tensor, outputs: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Assume shape (batch_size, 1) or same mask for all dimensions of
    # given input
    batch_idxs = mask.nonzero().squeeze()
    outputs = outputs.index_select(dim=0, index=batch_idxs)
    targets = targets.index_select(dim=0, index=batch_idxs)

    return outputs, targets


def add_zeros(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    res = torch.zeros_like(mask, dtype=x.dtype)
    # Looks weird but the gradient propagates just fine (tested it)
    res[mask.nonzero().squeeze()] = x

    return res


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
        batch_size = outputs.size(0)
        if mask is not None:
            outputs, targets = index_nonzero(mask, outputs, targets)

        masked_batch_size = outputs.size(0)

        if masked_batch_size == 0:
            return {
                key: torch.zeros(
                    batch_size,
                    dtype=torch.float32,
                    device=outputs.device,
                )
                for key in ["loss", "marginals_positive", "marginals_negative"]
            }

        positive = self.dissimilarity_fn(outputs, targets)

        negative = torch.zeros_like(positive)
        if masked_batch_size > 1:
            for shifts in range(1, masked_batch_size):
                negative -= self.dissimilarity_fn(outputs, targets.roll(shifts, dims=0))

            # compute mean, to be independent of batch size
            negative /= masked_batch_size - 1
            negative *= self.lam

        loss = positive + negative
        if mask is not None:
            loss = add_zeros(mask, loss)

        return {
            "loss": loss,
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
            outputs, targets = index_nonzero(mask, outputs, targets)

        logits = outputs @ targets.T

        logits /= torch.linalg.vector_norm(outputs, dim=1).unsqueeze(-1)
        logits /= torch.linalg.vector_norm(targets, dim=1).unsqueeze(0)

        labels = torch.arange(outputs.size(0), device=logits.device)
        xentropy = torch.nn.functional.cross_entropy(logits, labels)

        if mask is not None:
            xentropy = add_zeros(mask, xentropy)

        return {"loss": xentropy}


class MaskedCosineDistance(torch.nn.CosineSimilarity):
    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        cos_dist = 1 - super().forward(x1, x2)

        return {"loss": cos_dist if mask is None else cos_dist * mask}


class MaskedMSE(torch.nn.MSELoss):
    def __init__(self) -> None:
        super().__init__(reduction="none")

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        mse = super().forward(x1, x2).mean(1)

        return {"loss": mse if mask is None else mse * mask}


class MaskedHuber(torch.nn.HuberLoss):
    def __init__(self, delta: float = 1) -> None:
        super().__init__(reduction="none", delta=delta)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        huber = super().forward(x1, x2).mean(1)

        return {"loss": huber if mask is None else huber * mask}


class MaskedL1(torch.nn.L1Loss):
    def __init__(self) -> None:
        super().__init__(size_average=None, reduce=None, reduction="none")

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        l1 = super().forward(x1, x2).mean(1)

        return {"loss": l1 if mask is None else l1 * mask}


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
