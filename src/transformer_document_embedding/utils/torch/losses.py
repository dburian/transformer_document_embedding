from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Callable


from transformer_document_embedding.utils.torch.net_helpers import get_activation

if TYPE_CHECKING:
    from typing import Optional


class AlwaysStaticShortContextual(torch.nn.Module):
    def __init__(
        self,
        contextual_key: str,
        static_key: str,
        static_loss: torch.nn.Module,
        lambda_: float = 0.5,
        len_key: str = "length",
        len_threshold: int = 512,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._contextual_key = contextual_key
        self._static_key = static_key
        self._len_threshold = len_threshold
        self._len_key = len_key
        self._static_loss = static_loss
        self._lambda = lambda_

    def forward(
        self, inputs: torch.Tensor, targets: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        contextual_mask = targets[self._len_key] < self._len_threshold
        contextual_loss = torch.mean(
            torch.nn.functional.mse_loss(inputs, targets[self._contextual_key])
            * contextual_mask
        )
        static_loss_outputs = self._static_loss(inputs, targets[self._static_key])
        static_loss = torch.mean(static_loss_outputs.pop("loss"))

        return {
            "loss": static_loss + self._lambda * contextual_loss,
            "static_loss": static_loss.detach(),
            "contextual_loss": contextual_loss.detach(),
            **static_loss_outputs,
        }


class CCALoss(torch.nn.Module):
    def __init__(
        self,
        output_dimension: Optional[int] = None,
        regularization_constant: float = 1e-3,
        epsilon: float = 1e-9,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reg_constant = regularization_constant
        self._eps = epsilon
        self._output_dim = output_dimension

    def _covariance(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        add_regularization: bool = False,
    ) -> torch.Tensor:
        n = x.size(1)  # observation count
        cov = (1 / (n - 1)) * torch.matmul(x, y.T)

        m = x.size(0)  # features count
        return (
            cov + self._reg_constant * torch.eye(m, device=x.device)
            if add_regularization
            else cov
        )

    def _return_computation(
        self, neg_correlation: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {"loss": neg_correlation}

    def _compute_covariance_matrices(
        self, view1: torch.Tensor, view2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # So that observations are columns, features are rows
        view1, view2 = view1.T, view2.T

        view1_bar = view1 - view1.mean(dim=1).unsqueeze(dim=1)
        view2_bar = view2 - view2.mean(dim=1).unsqueeze(dim=1)

        sigma = self._covariance(view1_bar, view2_bar)
        sigma_1 = self._covariance(view1_bar, view1_bar, add_regularization=True)
        sigma_2 = self._covariance(view2_bar, view2_bar, add_regularization=True)

        return sigma, sigma_1, sigma_2

    def forward(
        self, view1: torch.Tensor, view2: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        device = view1.device

        sigma, sigma_1, sigma_2 = self._compute_covariance_matrices(view1, view2)

        D1, V1 = torch.linalg.eigh(sigma_1)
        D2, V2 = torch.linalg.eigh(sigma_2)

        # To increase stability, give up eigenvectors with small eigenvalues
        # Indices of rows with elements of `D1` larger than `eps`
        large_eigh_idxs = torch.gt(D1, self._eps).nonzero()[:, 0]
        D1 = D1[large_eigh_idxs]
        # Forget eigenvectors with small eigenvalues
        V1 = V1[:, large_eigh_idxs]

        large_eigh_idxs = torch.gt(D2, self._eps).nonzero()[:, 0]
        D2 = D2[large_eigh_idxs]
        V2 = V2[:, large_eigh_idxs]

        sigma_1_root_inv = V1 @ torch.diag(D1**-0.5) @ V1.T
        sigma_2_root_inv = V2 @ torch.diag(D2**-0.5) @ V2.T

        # The matrix whose singular values are canonical correlations
        A = sigma_1_root_inv @ sigma @ sigma_2_root_inv

        if self._output_dim is None:
            # We are using all singular values
            corr = torch.trace(torch.sqrt(A.T @ A))
            return -corr

        A_times_A = A.T @ A
        A_times_A = torch.add(
            A_times_A,
            (self._reg_constant * torch.eye(A_times_A.shape[0]).to(device)),
        )
        eigenvalues = torch.linalg.eigvalsh(A_times_A)
        eigenvalues = torch.where(
            eigenvalues > self._eps,
            eigenvalues,
            (torch.ones(eigenvalues.shape).double() * self._eps).to(device),
        )

        eigenvalues = eigenvalues.topk(self._output_dim)[0]
        corr = torch.sum(torch.sqrt(eigenvalues))

        return self._return_computation(-corr)


class RunningCCALoss(CCALoss):
    """CCA loss with running means for covariances and means"""

    def __init__(
        self,
        view1_dimension: int,
        view2_dimension: int,
        output_dimension: Optional[int] = None,
        regularization_constant: float = 0.001,
        epsilon: float = 1e-9,
        beta_mu: float = 0.9,
        beta_sigma: float = 0.9,
        device: Optional[torch.device] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            output_dimension,
            regularization_constant,
            epsilon,
            *args,
            **kwargs,
        )

        self._sigma1 = torch.nn.Parameter(
            torch.eye(view1_dimension, device=device),
            requires_grad=False,
        )
        self._sigma2 = torch.nn.Parameter(
            torch.eye(view2_dimension, device=device),
            requires_grad=False,
        )
        self._sigma = torch.nn.Parameter(
            torch.zeros(view1_dimension, view2_dimension, device=device),
            requires_grad=False,
        )
        self._mu1 = torch.nn.Parameter(
            torch.zeros(view1_dimension, device=device),
            requires_grad=False,
        )
        self._mu2 = torch.nn.Parameter(
            torch.zeros(view2_dimension, device=device),
            requires_grad=False,
        )

        self._beta_mu = beta_mu
        self._beta_sigma = beta_sigma
        self._beta_mu_power = 1
        self._beta_sigma_power = 1

    @staticmethod
    def _running_update(
        beta: torch.Tensor, old: torch.Tensor, new: torch.Tensor
    ) -> torch.Tensor:
        return old.detach() * beta + (1 - beta) * new

    def _compute_covariance_matrices(
        self, view1: torch.Tensor, view2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # So that observations are columns, features are rows
        view1, view2 = view1.T, view2.T

        view1_mean, view2_mean = self._compute_means(view1, view2)

        view1_bar = view1 - view1_mean.unsqueeze(dim=1)
        view2_bar = view2 - view2_mean.unsqueeze(dim=1)

        last_sigma = self._covariance(view1_bar, view2_bar)
        last_sigma1 = self._covariance(view1_bar, view1_bar)
        last_sigma2 = self._covariance(view2_bar, view2_bar)

        self._sigma.detach().mul_(self._beta_sigma).add_(
            last_sigma, alpha=1 - self._beta_sigma
        )
        self._sigma1.detach().mul_(self._beta_sigma).add_(
            last_sigma1, alpha=1 - self._beta_sigma
        )
        self._sigma2.detach().mul_(self._beta_sigma).add_(
            last_sigma2, alpha=1 - self._beta_sigma
        )

        self._beta_sigma_power *= self._beta_sigma
        sigma = self._sigma / (1 - self._beta_sigma_power)
        sigma1 = self._sigma1 / (1 - self._beta_sigma_power)
        sigma2 = self._sigma2 / (1 - self._beta_sigma_power)

        return sigma, sigma1, sigma2

    def _compute_means(
        self, view1: torch.Tensor, view2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
        -----------
            view1: torch.Tensor
                Embeddings of first view as columns.
            view2: torch.Tensor
                Embeddings of second view as columns.
        """
        self._mu1.detach().mul_(self._beta_mu).add_(
            view1.mean(dim=1), alpha=1 - self._beta_mu
        )
        self._mu2.detach().mul_(self._beta_mu).add_(
            view2.mean(dim=1), alpha=1 - self._beta_mu
        )

        self._beta_mu_power *= self._beta_mu
        mu1 = self._mu1 / (1 - self._beta_mu_power)
        mu2 = self._mu2 / (1 - self._beta_mu_power)
        return mu1, mu2

    def _return_computation(
        self, neg_correlation: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {
            **super()._return_computation(neg_correlation),
            "covariance_mat": self._sigma.detach(),
            "sigma2": (self._sigma2 / (1 - self._beta_sigma_power)).detach(),
        }


class DeepNet(torch.nn.Module):
    """Feed-forward net for DeepCCA loss"""

    def __init__(
        self,
        layer_features: list[int],
        input_features: int,
        activation: str = "relu",
        norm: Optional[str] = "layer",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        norm_class = None
        if norm is not None and norm == "layer":
            norm_class = torch.nn.LayerNorm
        elif norm is not None and norm == "batch":
            norm_class = torch.nn.BatchNorm1d

        layers = []
        features = [input_features] + layer_features
        for input_dim, output_dim in zip(features[:-1], features[1:], strict=True):
            if norm_class is not None:
                layers.append(norm_class(input_dim))

            layers.append(get_activation(activation)())
            layers.append(torch.nn.Linear(input_dim, output_dim))

        # Use ModuleList instead of Sequential to allow empty layers
        self.layers = torch.nn.ModuleList(modules=layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs


class DCCALoss(torch.nn.Module):
    def __init__(
        self,
        net1: DeepNet,
        net2: DeepNet,
        cca_loss: CCALoss,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._net1 = net1
        self._net2 = net2
        self._cca = cca_loss

    def forward(
        self, view1: torch.Tensor, view2: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        projected_view1 = self._net1(view1)
        projected_view2 = self._net2(view2)

        return {
            **self._cca(projected_view1, projected_view2),
            "projected_view1": projected_view1,
            "projected_view2": projected_view2,
        }


# Here just as a reminder how to do it in case I need to
def get_cross_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # All variables as rows, observations as columns
    var_obs = torch.cat([x.T, y.T]).detach()
    corr = torch.corrcoef(var_obs)
    x_vars = x.size(1)
    cross_corr = corr[x_vars:, :x_vars]
    return torch.sum(cross_corr)


class ProjectedMSE(torch.nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        activation: str = "linear",
        loss: torch.nn.Module
        | Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = torch.nn.functional.mse_loss,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._linear = torch.nn.Linear(input_features, output_features)
        self._activation = get_activation(activation)()
        self._loss = loss() if isinstance(loss, torch.nn.Module) else loss

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        transformed = self._activation(self._linear(inputs))

        return {"loss": self._loss(transformed, targets)}
