"""Factory for creating predictor heads from topological features.

Converts persistent homology features into lightweight predictor modules:
- H_0 (attractors) -> AttractorModule: local predictor centered on cluster centroid
- H_1 (cycles) -> CycleModule: periodic/oscillatory predictor
- H_2 (boundaries) -> BoundaryModule: surface/interface predictor
"""

from typing import Optional

import torch
import torch.nn as nn

from tcd_jepa.topology.feature_extraction import TopologicalFeature


class AttractorModule(nn.Module):
    """Local predictor centered on a cluster centroid in latent space.

    Specializes in predicting representations near a specific attractor
    basin discovered by System 2's exploration.
    """

    def __init__(self, embed_dim: int, centroid: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.centroid = nn.Parameter(
            centroid if centroid is not None else torch.randn(embed_dim) * 0.01,
            requires_grad=False,
        )
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        # Learnable radius of influence
        self.log_radius = nn.Parameter(torch.tensor(0.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict with attention weight based on distance to centroid.

        Args:
            z: Input representations [B, D].

        Returns:
            Weighted prediction [B, D].
        """
        dist = (z - self.centroid).pow(2).sum(dim=-1, keepdim=True)
        radius = self.log_radius.exp()
        weight = torch.exp(-dist / (2.0 * radius ** 2 + 1e-8))
        return weight * self.predictor(z)


class CycleModule(nn.Module):
    """Periodic predictor for cyclic patterns in latent space.

    Captures oscillatory structures discovered as H_1 features.
    """

    def __init__(self, embed_dim: int, num_frequencies: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.frequencies = nn.Parameter(torch.randn(num_frequencies) * 0.1)
        self.phases = nn.Parameter(torch.randn(num_frequencies) * 0.1)
        self.projection = nn.Linear(embed_dim, num_frequencies)
        self.output = nn.Linear(num_frequencies * 2, embed_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply periodic transformation.

        Args:
            z: Input [B, D].

        Returns:
            Periodic prediction [B, D].
        """
        projected = self.projection(z)  # [B, num_freq]
        sin_features = torch.sin(projected * self.frequencies + self.phases)
        cos_features = torch.cos(projected * self.frequencies + self.phases)
        periodic = torch.cat([sin_features, cos_features], dim=-1)
        return self.output(periodic)


class BoundaryModule(nn.Module):
    """Interface predictor for boundary structures in latent space.

    Captures surface/boundary structures discovered as H_2 features.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.boundary_detector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )
        self.predictor_a = nn.Linear(embed_dim, embed_dim)
        self.predictor_b = nn.Linear(embed_dim, embed_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict using boundary-aware interpolation.

        Args:
            z: Input [B, D].

        Returns:
            Boundary-interpolated prediction [B, D].
        """
        alpha = self.boundary_detector(z)  # [B, 1]
        return alpha * self.predictor_a(z) + (1 - alpha) * self.predictor_b(z)


class ModuleFactory:
    """Creates predictor modules from topological features."""

    def __init__(self, embed_dim: int) -> None:
        self.embed_dim = embed_dim

    def create_module(
        self,
        feature: TopologicalFeature,
        device: torch.device = torch.device("cpu"),
    ) -> nn.Module:
        """Instantiate a predictor module from a topological feature.

        Args:
            feature: TopologicalFeature with dim, centroid, etc.
            device: Target device.

        Returns:
            Initialized nn.Module.
        """
        if feature.module_type == "attractor" or feature.dim == 0:
            centroid = None
            if feature.centroid is not None:
                centroid = torch.from_numpy(feature.centroid).float()
            module = AttractorModule(self.embed_dim, centroid=centroid)

        elif feature.module_type == "cycle" or feature.dim == 1:
            module = CycleModule(self.embed_dim)

        elif feature.module_type == "boundary" or feature.dim == 2:
            module = BoundaryModule(self.embed_dim)

        else:
            # Default to attractor
            module = AttractorModule(self.embed_dim)

        return module.to(device)
