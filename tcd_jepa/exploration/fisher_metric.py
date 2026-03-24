"""Fisher information metric for Riemannian geometry on the latent space.

The Fisher metric defines the natural geometry of the predictor's output
distribution, enabling geometrically-aware exploration in System 2.

F_ij(z) = E[d/dz_i log p(y|z) * d/dz_j log p(y|z)]

For a deterministic predictor with Gaussian noise model:
F_ij(z) = (1/sigma^2) * sum_k (dp_k/dz_i)(dp_k/dz_j)
"""


import torch


class FisherMetric:
    """Computes the Fisher information metric on JEPA's latent space.

    Measures how sensitive the predictor output is to changes in
    latent position, defining a Riemannian metric for exploration.
    """

    def __init__(
        self,
        noise_variance: float = 1.0,
        num_jacobian_samples: int = 16,
        finite_diff_eps: float = 1e-3,
    ) -> None:
        self.noise_variance = noise_variance
        self.num_jacobian_samples = num_jacobian_samples
        self.finite_diff_eps = finite_diff_eps

    def compute_jacobian_fd(
        self,
        z: torch.Tensor,
        predictor_fn: callable,
    ) -> torch.Tensor:
        """Estimate the Jacobian dp/dz via finite differences.

        Args:
            z: Latent points [B, D_in].
            predictor_fn: Maps [B, D_in] -> [B, D_out].

        Returns:
            Jacobian estimate [B, D_out, D_in] (sampled directions).
        """
        B, D_in = z.shape
        eps = self.finite_diff_eps

        # Sample random directions to estimate Jacobian columns
        num_dirs = min(D_in, self.num_jacobian_samples)
        directions = torch.randn(num_dirs, D_in, device=z.device)
        directions = directions / directions.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            p_z = predictor_fn(z)  # [B, D_out]
            _ = p_z.shape[-1]

        jacobian_cols = []
        with torch.no_grad():
            for i in range(num_dirs):
                d_i = directions[i].unsqueeze(0) * eps  # [1, D_in]
                p_plus = predictor_fn(z + d_i)
                dp = (p_plus - p_z) / eps
                jacobian_cols.append(dp)

        # [B, D_out, num_dirs]
        return torch.stack(jacobian_cols, dim=-1)

    def compute_fisher_matrix(
        self,
        z: torch.Tensor,
        predictor_fn: callable,
    ) -> torch.Tensor:
        """Compute the Fisher information matrix at latent points.

        F(z) = (1/sigma^2) * J^T J

        Args:
            z: Latent points [B, D_in].
            predictor_fn: Maps [B, D_in] -> [B, D_out].

        Returns:
            Fisher matrix [B, K, K] where K = num_jacobian_samples.
        """
        J = self.compute_jacobian_fd(z, predictor_fn)  # [B, D_out, K]
        fisher = torch.bmm(J.transpose(1, 2), J) / self.noise_variance
        return fisher

    def compute_geodesic_distance(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        predictor_fn: callable,
    ) -> torch.Tensor:
        """Approximate geodesic distance between points using the Fisher metric.

        Uses the midpoint Fisher matrix: d(z1, z2) approx sqrt((z2-z1)^T F(z_mid) (z2-z1))

        Args:
            z1, z2: Latent points [B, D].
            predictor_fn: Predictor function.

        Returns:
            Geodesic distances [B].
        """
        z_mid = (z1 + z2) / 2.0
        J = self.compute_jacobian_fd(z_mid, predictor_fn)  # [B, D_out, K]

        # Project difference onto Jacobian directions
        # Since J is in sampled directions, project diff onto those directions
        fisher = torch.bmm(J.transpose(1, 2), J) / self.noise_variance  # [B, K, K]

        # For the actual distance, we need diff in the sampled basis
        # Approximate: use the trace of Fisher as a scalar metric
        fisher_trace = torch.diagonal(fisher, dim1=-2, dim2=-1).sum(dim=-1)  # [B]
        euclidean_dist = (z2 - z1).pow(2).sum(dim=-1)  # [B]

        return torch.sqrt(fisher_trace * euclidean_dist / z1.shape[-1])

    def compute_metric_tensor_trace(
        self,
        z: torch.Tensor,
        predictor_fn: callable,
    ) -> torch.Tensor:
        """Compute trace of Fisher metric (scalar curvature proxy).

        Higher trace = predictor is more sensitive at this point = well-mapped region.
        Lower trace = predictor is insensitive = potential blank space.

        Args:
            z: Latent points [B, D].
            predictor_fn: Predictor function.

        Returns:
            Fisher trace per sample [B].
        """
        fisher = self.compute_fisher_matrix(z, predictor_fn)
        return torch.diagonal(fisher, dim1=-2, dim2=-1).sum(dim=-1)
