"""Blank space detection in JEPA's energy landscape.

Identifies regions where the predictor is uncertain: flat energy landscape
(small Hessian eigenvalues) or high output variance under perturbation.

Mathematical basis:
    H(z) = nabla^2_z E(z)
    blank_space(z) = True if lambda_min(H(z)) < tau
"""

from typing import Optional

import torch


class BlankSpaceDetector:
    """Detects uncertain regions in JEPA's latent energy landscape.

    Two complementary strategies:
    1. Hessian-based: flat regions where energy curvature is low
    2. Perturbation-based: small input changes cause large output variance
    """

    def __init__(
        self,
        flatness_threshold: float = 0.1,
        variance_threshold: float = 1.0,
        perturbation_std: float = 0.01,
        num_perturbations: int = 10,
        num_hessian_directions: int = 32,
    ) -> None:
        self.flatness_threshold = flatness_threshold
        self.variance_threshold = variance_threshold
        self.perturbation_std = perturbation_std
        self.num_perturbations = num_perturbations
        self.num_hessian_directions = num_hessian_directions

    @torch.no_grad()
    def compute_energy_hessian_spectrum(
        self,
        z: torch.Tensor,
        energy_fn: callable,
    ) -> dict[str, torch.Tensor]:
        """Approximate Hessian eigenvalues via finite differences along random directions.

        H_dd approx (E(z+eps*d) - 2*E(z) + E(z-eps*d)) / eps^2

        Args:
            z: Latent points [B, D].
            energy_fn: Maps [B, D] -> [B] energy per sample.

        Returns:
            Dict with eigenvalue estimates and curvatures.
        """
        B, D = z.shape
        eps = self.perturbation_std
        e_z = energy_fn(z)

        num_dirs = min(D, self.num_hessian_directions)
        directions = torch.randn(num_dirs, D, device=z.device)
        directions = directions / directions.norm(dim=-1, keepdim=True)

        curvatures = []
        for i in range(num_dirs):
            d_i = directions[i].unsqueeze(0)
            e_plus = energy_fn(z + eps * d_i)
            e_minus = energy_fn(z - eps * d_i)
            curvature = (e_plus - 2.0 * e_z + e_minus) / (eps ** 2)
            curvatures.append(curvature)

        curvatures = torch.stack(curvatures, dim=-1)  # [B, num_dirs]

        return {
            "eigenvalues_min": curvatures.min(dim=-1).values,
            "eigenvalues_max": curvatures.max(dim=-1).values,
            "trace": curvatures.sum(dim=-1),
            "curvatures": curvatures,
        }

    @torch.no_grad()
    def compute_perturbation_variance(
        self,
        z: torch.Tensor,
        predictor_fn: callable,
    ) -> torch.Tensor:
        """Measure predictor output variance under small input perturbations.

        Args:
            z: Latent points [B, D].
            predictor_fn: Maps [B, D] -> [B, D_out].

        Returns:
            Per-sample variance [B].
        """
        outputs = []
        for _ in range(self.num_perturbations):
            noise = torch.randn_like(z) * self.perturbation_std
            outputs.append(predictor_fn(z + noise))

        stacked = torch.stack(outputs, dim=0)  # [K, B, D_out]
        return stacked.var(dim=0).mean(dim=-1)  # [B]

    @torch.no_grad()
    def detect(
        self,
        z: torch.Tensor,
        energy_fn: callable,
        predictor_fn: Optional[callable] = None,
    ) -> dict[str, torch.Tensor]:
        """Detect blank spaces using Hessian flatness and perturbation variance.

        Args:
            z: Latent points [B, D].
            energy_fn: Maps [B, D] -> [B] energy.
            predictor_fn: Optional, maps [B, D] -> [B, D_out].

        Returns:
            Dict with 'is_blank', 'flatness_score', 'variance_score', 'combined_score'.
        """
        B = z.shape[0]
        device = z.device

        hessian_info = self.compute_energy_hessian_spectrum(z, energy_fn)
        flatness_score = hessian_info["eigenvalues_min"].abs()
        is_flat = flatness_score < self.flatness_threshold

        if predictor_fn is not None:
            variance_score = self.compute_perturbation_variance(z, predictor_fn)
            is_high_var = variance_score > self.variance_threshold
        else:
            variance_score = torch.zeros(B, device=device)
            is_high_var = torch.zeros(B, dtype=torch.bool, device=device)

        is_blank = is_flat | is_high_var
        flatness_normalized = 1.0 / (flatness_score + 1e-8)
        combined_score = flatness_normalized + variance_score

        return {
            "is_blank": is_blank,
            "flatness_score": flatness_score,
            "variance_score": variance_score,
            "combined_score": combined_score,
            "hessian_info": hessian_info,
        }
