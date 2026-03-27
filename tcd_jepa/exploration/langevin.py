"""Langevin dynamics sampler for exploring JEPA's energy surface.

z_{t+1} = z_t - eta * grad_z E(z_t) + sqrt(2*eta/beta) * epsilon_t

Temperature beta is biased toward blank space regions (lower beta = more exploration).
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger("tcd_jepa")


class LangevinSampler:
    """Explores the energy landscape via Langevin dynamics.

    Biases temperature toward blank-space regions to preferentially
    explore uncertain areas of the latent space.
    """

    def __init__(
        self,
        step_size: float = 0.01,
        temperature: float = 1.0,
        min_temperature: float = 0.1,
        max_steps: int = 100,
        grad_clip: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.step_size = step_size
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.max_steps = max_steps
        self.grad_clip = grad_clip
        self.generator = generator

    def _randn_like(self, z: torch.Tensor) -> torch.Tensor:
        """Generate random noise with optional seeded generator for reproducibility."""
        if self.generator is not None:
            return torch.randn(
                z.shape, generator=self.generator, device=z.device, dtype=z.dtype
            )
        return torch.randn_like(z)

    def _compute_energy_gradient(
        self,
        z: torch.Tensor,
        energy_fn: callable,
    ) -> torch.Tensor:
        """Compute gradient of energy w.r.t. latent positions.

        Args:
            z: Latent points [B, D], requires_grad will be set.
            energy_fn: Maps [B, D] -> scalar energy.

        Returns:
            Gradient [B, D].
        """
        # Enable grad even if called from a no_grad context
        with torch.enable_grad():
            z_var = z.detach().requires_grad_(True)
            energy = energy_fn(z_var)
            if energy.dim() > 0:
                energy = energy.sum()

            # Guard: check energy is finite before computing gradients
            if not torch.isfinite(energy):
                logger.warning("Non-finite energy detected in Langevin gradient — returning zero gradient")
                return torch.zeros_like(z)

            try:
                grad_tuple = torch.autograd.grad(energy, z_var, allow_unused=True)
            except RuntimeError:
                logger.warning("Energy function is not differentiable — returning zero gradient")
                return torch.zeros_like(z)

            if grad_tuple[0] is None:
                logger.warning("Energy function returned no gradient — returning zero gradient")
                return torch.zeros_like(z)
            grad = grad_tuple[0]

        # Guard: check for NaN gradients
        if torch.isnan(grad).any():
            logger.warning("NaN gradient in Langevin dynamics — returning zero gradient")
            return torch.zeros_like(z)

        # Clip gradients for stability
        grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        grad = grad * torch.clamp(self.grad_clip / grad_norm, max=1.0)
        return grad.detach()

    def step(
        self,
        z: torch.Tensor,
        energy_fn: callable,
        temperature_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Take one Langevin dynamics step.

        z_{t+1} = z_t - eta * grad_z E(z_t) + sqrt(2*eta/beta) * eps

        Args:
            z: Current latent positions [B, D].
            energy_fn: Maps [B, D] -> energy.
            temperature_map: Per-sample temperature [B]. Lower = more exploration.

        Returns:
            Updated positions [B, D].
        """
        grad = self._compute_energy_gradient(z, energy_fn)

        if temperature_map is not None:
            beta = temperature_map.unsqueeze(-1)  # [B, 1]
        else:
            beta = torch.full((z.shape[0], 1), self.temperature, device=z.device)

        beta = beta.clamp(min=self.min_temperature)

        noise_scale = torch.sqrt(2.0 * self.step_size / beta).clamp(max=10.0)
        noise = self._randn_like(z) * noise_scale

        z_new = z - self.step_size * grad + noise
        return z_new.detach()

    @torch.no_grad()
    def sample_trajectory(
        self,
        z_init: torch.Tensor,
        energy_fn: callable,
        num_steps: Optional[int] = None,
        blank_score: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run Langevin dynamics for multiple steps, recording the trajectory.

        Args:
            z_init: Starting positions [B, D].
            energy_fn: Energy function.
            num_steps: Steps to take (defaults to self.max_steps).
            blank_score: Per-sample blank space score [B] — higher = lower temperature.

        Returns:
            Trajectory tensor [T, B, D] where T = num_steps + 1.
        """
        num_steps = num_steps or self.max_steps

        # Convert blank scores to temperature map: high blank_score -> low temperature
        if blank_score is not None:
            # Inverse relationship: more blank -> more exploration (lower beta)
            temperature_map = self.temperature / (1.0 + blank_score)
        else:
            temperature_map = None

        trajectory = [z_init.detach().clone()]
        z = z_init.detach().clone()

        for _ in range(num_steps):
            z = self.step(z, energy_fn, temperature_map)
            trajectory.append(z.clone())

        return torch.stack(trajectory, dim=0)  # [T, B, D]

    def differentiable_step(
        self,
        z: torch.Tensor,
        energy_fn: callable,
        temperature_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Take one differentiable Langevin step using create_graph=True.

        Unlike `step()`, this preserves the computation graph so gradients
        can flow back through the Langevin dynamics for end-to-end training.
        Uses the reparameterization trick for the noise term.

        Args:
            z: Current latent positions [B, D], must have requires_grad=True.
            energy_fn: Maps [B, D] -> energy (must be differentiable).
            temperature_map: Per-sample temperature [B].

        Returns:
            Updated positions [B, D] with gradient graph intact.
        """
        with torch.enable_grad():
            z_var = z if z.requires_grad else z.requires_grad_(True)
            energy = energy_fn(z_var)
            if energy.dim() > 0:
                energy = energy.sum()
            grad = torch.autograd.grad(energy, z_var, create_graph=True)[0]

        # Clip gradients for stability
        grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        grad = grad * torch.clamp(self.grad_clip / grad_norm, max=1.0)

        if temperature_map is not None:
            beta = temperature_map.unsqueeze(-1)
        else:
            beta = torch.full((z.shape[0], 1), self.temperature, device=z.device)
        beta = beta.clamp(min=self.min_temperature)

        # Reparameterization trick: noise is detached but scale is differentiable
        noise_scale = torch.sqrt(2.0 * self.step_size / beta)
        noise = self._randn_like(z).detach() * noise_scale

        z_new = z - self.step_size * grad + noise
        return z_new

    def differentiable_sample(
        self,
        z_init: torch.Tensor,
        energy_fn: callable,
        num_warmup_steps: int = 10,
        num_diff_steps: Optional[int] = None,
        blank_score: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run detached warmup followed by K differentiable final steps.

        This enables end-to-end gradient flow through the final exploration
        steps while keeping the warmup phase detached for efficiency.

        Args:
            z_init: Starting positions [B, D].
            energy_fn: Energy function (must support create_graph=True).
            num_warmup_steps: Number of initial detached warmup steps.
            num_diff_steps: Number of final differentiable steps (default 5).
            blank_score: Per-sample blank space score [B].

        Returns:
            Final positions [B, D] with gradient graph from diff steps.
        """
        num_diff_steps = num_diff_steps or 5

        # Temperature map from blank scores
        if blank_score is not None:
            temperature_map = self.temperature / (1.0 + blank_score)
        else:
            temperature_map = None

        # Phase 1: Detached warmup (no gradient tracking)
        z = z_init.detach().clone()
        for _ in range(num_warmup_steps):
            z = self.step(z, energy_fn, temperature_map)

        # Phase 2: Differentiable final steps
        z = z.requires_grad_(True)
        for _ in range(num_diff_steps):
            z = self.differentiable_step(z, energy_fn, temperature_map)

        return z
