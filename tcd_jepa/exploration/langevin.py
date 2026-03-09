"""Langevin dynamics sampler for exploring JEPA's energy surface.

z_{t+1} = z_t - eta * grad_z E(z_t) + sqrt(2*eta/beta) * epsilon_t

Temperature beta is biased toward blank space regions (lower beta = more exploration).
"""

from typing import Optional

import torch
import torch.nn as nn


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
    ) -> None:
        self.step_size = step_size
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.max_steps = max_steps
        self.grad_clip = grad_clip

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
            grad = torch.autograd.grad(energy, z_var)[0]

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

        noise_scale = torch.sqrt(2.0 * self.step_size / beta)
        noise = torch.randn_like(z) * noise_scale

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
