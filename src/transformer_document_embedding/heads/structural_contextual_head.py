from __future__ import annotations
from typing import TYPE_CHECKING, Any

import torch
from transformer_document_embedding.datasets import col
from transformer_document_embedding.utils import cca_losses

from transformer_document_embedding.utils.similarity_losses import create_sim_based_loss

if TYPE_CHECKING:
    from transformer_document_embedding.models.embedding_model import EmbeddingModel
    from typing import Optional


class StructuralContextualHead(torch.nn.Module):
    def __init__(
        self,
        lam: float,
        max_structural_length: Optional[int],
        contextual_head_kwargs: Optional[dict[str, Any]],
        structural_head_kwargs: Optional[dict[str, Any]],
        embedding_model: EmbeddingModel,
    ) -> None:
        super().__init__()

        self.structural_head = (
            None
            if structural_head_kwargs is None
            else self.create_structural_head(**structural_head_kwargs)
        )
        self.contextual_head = (
            None
            if contextual_head_kwargs is None
            else self.create_contextual_head(
                student_dim=embedding_model.embedding_dim, **contextual_head_kwargs
            )
        )

        self.max_structural_length = max_structural_length
        self._lam = lam

    def create_structural_head(
        self, loss_type: str, max_marginals_lam: Optional[float] = None
    ) -> torch.nn.Module:
        return create_sim_based_loss(
            loss_type=loss_type,
            max_marginals_lam=max_marginals_lam,
        )

    def create_contextual_head(
        self,
        student_dim: int,
        contextual_dim: int,
        student_projection: list[dict[str, Any]],
        contextual_projection: list[dict[str, Any]],
        loss_type: str,
        cca_output_dim: Optional[int] = None,
        soft_cca_lam: Optional[float] = None,
        soft_cca_sdl_alpha: Optional[float] = None,
        max_marginals_lam: Optional[float] = None,
    ) -> cca_losses.ProjectionLoss:
        student_net = cca_losses.DeepNet(
            blocks_config=student_projection, input_features=student_dim
        )
        contextual_net = cca_losses.DeepNet(
            blocks_config=contextual_projection, input_features=contextual_dim
        )

        loss_fn = None
        if loss_type == "cca":
            loss_fn = cca_losses.CCALoss(output_dimension=cca_output_dim)
        elif loss_type == "running_cca":
            loss_fn = cca_losses.RunningCCALoss(
                view1_dimension=student_net.features[-1],
                view2_dimension=contextual_net.features[-1],
                output_dimension=cca_output_dim,
            )
        elif loss_type == "soft_cca":
            assert (
                soft_cca_lam is not None and soft_cca_sdl_alpha is not None
            ), "To use soft_cca, `soft_cca_lam` and `soft_cca_sdl_alpha` must be set"
            loss_fn = cca_losses.SoftCCALoss(
                sdl1=cca_losses.StochasticDecorrelationLoss(
                    student_net.features[-1],
                    alpha=soft_cca_sdl_alpha,
                ),
                sdl2=cca_losses.StochasticDecorrelationLoss(
                    contextual_net.features[-1],
                    alpha=soft_cca_sdl_alpha,
                ),
                lam=soft_cca_lam,
            )
        else:
            loss_fn = create_sim_based_loss(
                loss_type, max_marginals_lam=max_marginals_lam
            )

        return cca_losses.ProjectionLoss(
            net1=student_net,
            net2=contextual_net,
            loss_fn=loss_fn,
        )

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        # So to rely on column names defined by constants
        embeddings = kwargs[col.EMBEDDING]
        lengths = kwargs[col.LENGTH]
        structural_targets = kwargs.pop(col.STRUCTURAL_EMBED, None)
        contextual_targets = kwargs.pop(col.CONTEXTUAL_EMBED, None)

        outputs = {
            "loss": torch.zeros(
                embeddings.shape[0],
                device=embeddings.device,
                dtype=embeddings.dtype,
            )
        }

        weighting = self._lam != 0.5
        if weighting:
            lams = torch.zeros_like(lengths, dtype=torch.float32).fill_(self._lam)

        if self.structural_head is not None:
            assert (
                structural_targets is not None
            ), "No targets for structural loss were given."

            mask = (
                lengths <= self.max_structural_length
                if self.max_structural_length is not None
                else torch.ones_like(lengths)
            )
            if weighting:
                lams *= mask

            structural_outputs = self.structural_head(
                embeddings, structural_targets, mask=mask
            )

            if weighting:
                structural_outputs["loss"] *= lams

            outputs["loss"] += structural_outputs["loss"]

            outputs.update(
                {
                    f"structural_{key}": value
                    for key, value in structural_outputs.items()
                }
            )
            outputs["structural_mask"] = mask

        if self.contextual_head is not None:
            assert (
                contextual_targets is not None
            ), "No targets for contextual loss were given."

            contextual_outputs = self.contextual_head(embeddings, contextual_targets)
            if weighting:
                contextual_outputs["loss"] *= 1 - lams

            outputs.update(
                {
                    f"contextual_{key}": value
                    for key, value in contextual_outputs.items()
                }
            )
            outputs["loss"] += contextual_outputs["loss"]

        outputs["loss"] = outputs["loss"].mean()

        return outputs
