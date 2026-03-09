"""Langevin dynamics over the JEPA energy surface.

z_{t+1} = z_t - eta * grad_z E(z_t) + sqrt(2*eta/beta) * epsilon_t

Temperature beta is biased to explore blank space regions.

Implementation: Phase 2
"""
