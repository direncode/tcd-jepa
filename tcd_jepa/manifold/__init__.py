"""Manifold-native extensions for TCD-JEPA.

Enables TCD-JEPA to operate on spherical databases storing fingerprints on
manifolds with causal links, rather than images. The core TCD topology pipeline
(persistent homology, crystallization, module factory) is already manifold-agnostic;
this package provides the input/encoder/masking layers that bridge manifold data
to the existing transformer + TCD infrastructure.
"""
