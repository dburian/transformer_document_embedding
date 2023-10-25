import torch

from transformer_document_embedding.utils.torch.net_helpers import get_activation


class AlwaysStaticShortContextual(torch.nn.Module):
    def __init__(
        self,
        contextual_key: str,
        static_key: str,
        static_loss: torch.nn.Module,
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

    def forward(
        self, inputs: torch.Tensor, targets: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        contextual_mask = targets[self._len_key] < self._len_threshold
        contextual_loss = torch.nn.functional.mse_loss(
            inputs, targets[self._contextual_key]
        )
        static_loss = self._static_loss(inputs, targets[self._static_key])

        return torch.mean(static_loss + contextual_mask * contextual_loss)


class CCALoss(torch.nn.Module):
    def __init__(
        self,
        output_dimension: int,
        use_all_singular_values: bool = True,
        regularization_constant: float = 1e-3,
        epsilon: float = 1e-9,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._use_all_singular_values = use_all_singular_values
        self._reg_constant = regularization_constant
        self._eps = epsilon
        self._output_dim = output_dimension

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        device = view1.device

        # So that observations are columns, features are rows
        view1, view2 = view1.T, view2.T

        view1_bar = view1 - view1.mean(dim=1).unsqueeze(dim=1)
        view2_bar = view2 - view2.mean(dim=1).unsqueeze(dim=1)

        def covariance(
            x: torch.Tensor,
            y: torch.Tensor,
            *,
            add_regularization: bool = False,
        ) -> torch.Tensor:
            n = x.size(1)  # observation count
            cov = (1 / (n - 1)) * torch.matmul(x, y.T)

            m = x.size(0)  # features count
            return (
                cov + self._reg_constant * torch.eye(m, device=device)
                if add_regularization
                else cov
            )

        sigma = covariance(view1_bar, view2_bar)
        sigma_1 = covariance(view1_bar, view1_bar, add_regularization=True)
        sigma_2 = covariance(view2_bar, view2_bar, add_regularization=True)

        D1, V1 = torch.linalg.eigh(sigma_1)
        D2, V2 = torch.linalg.eigh(sigma_2)

        # To increase stability, give up eigenvectors with small eigenvalues
        eps = 1e-9
        # Indices of rows with elements of `D1` larger than `eps`
        large_eigh_idxs = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[large_eigh_idxs]
        # Forget eigenvectors with small eigenvalues
        V1 = V1[:, large_eigh_idxs]

        large_eigh_idxs = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[large_eigh_idxs]
        V2 = V2[:, large_eigh_idxs]

        sigma_1_root_inv = V1 @ torch.diag(D1**-0.5) @ V1.T
        sigma_2_root_inv = V2 @ torch.diag(D2**-0.5) @ V2.T

        # The matrix whose singular values are canonical correlations
        A = sigma_1_root_inv @ sigma @ sigma_2_root_inv

        if self._use_all_singular_values:
            corr = torch.trace(torch.sqrt(A.T @ A))
            return -corr

        A_times_A = A.T @ A
        A_times_A = torch.add(
            A_times_A,
            (self._reg_constan * torch.eye(A_times_A.shape[0]).to(device)),
        )
        eigenvalues = torch.linalg.eigvalsh(A_times_A)
        eigenvalues = torch.where(
            eigenvalues > self._eps,
            eigenvalues,
            (torch.ones(eigenvalues.shape).double() * self._eps).to(device),
        )

        eigenvalues = eigenvalues.topk(self._output_dim)[0]
        corr = torch.sum(torch.sqrt(eigenvalues))
        return -corr


class DeepNet(torch.nn.Module):
    """Feed-forward net for DeepCCA loss"""

    def __init__(
        self,
        layer_features: list[int],
        input_features: int,
        activation: str = "relu",
        batch_norm: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        layers = []
        features = [input_features] + layer_features
        for input_dim, output_dim in zip(features[:-1], features[1:], strict=True):
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(input_dim))

            layers.append(get_activation(activation)())
            layers.append(torch.nn.Linear(input_dim, output_dim))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class DCCALoss(torch.nn.Module):
    def __init__(
        self, net1: DeepNet, net2: DeepNet, cca_loss: CCALoss, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self._net1 = net1
        self._net2 = net2
        self._cca = cca_loss

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        view1 = self._net1(view1)
        view2 = self._net2(view2)

        return self._cca(view1, view2)
