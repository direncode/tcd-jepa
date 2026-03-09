# Mathematical Foundations of TCD-JEPA

## Energy Function

The base JEPA energy measures prediction quality in latent space:

```
E(x, y) = ||s_θ(x) - s_ξ(y)||²
```

The predictor-augmented energy:

```
E_pred(x, y) = ||p_φ(s_θ(x)) - sg(s_ξ(y))||²
```

where:
- `s_θ`: context encoder (parameterized by θ)
- `s_ξ`: target encoder (EMA of θ, parameterized by ξ)
- `p_φ`: predictor (parameterized by φ)
- `sg`: stop-gradient operator

## Blank Space Detection

Regions where the energy Hessian has small eigenvalues (flat landscape):

```
H(z) = ∇²_z E(z)
blank_space(z) = True if λ_min(H(z)) < τ
```

Also identified via predictor output variance under small perturbations:

```
V(z) = E_ε[||p_φ(z + ε) - p_φ(z)||²],  ε ~ N(0, σ²I)
blank_space(z) = True if V(z) > τ_var
```

## Langevin Exploration

System 2 explores the energy surface via overdamped Langevin dynamics:

```
z_{t+1} = z_t - η∇_z E(z_t) + √(2η/β) · ε_t,  ε_t ~ N(0, I)
```

Temperature β is spatially varying, biased toward blank space:

```
β(z) = β_base / (1 + α · blank_score(z))
```

Lower β in blank space regions → more exploration.

## Fisher Information Metric

The Riemannian metric on latent space defined by the Fisher information of the predictor:

```
g_ij(z) = E[∂_i log p(y|z) · ∂_j log p(y|z)]
```

This gives geodesic distances that respect the predictor's uncertainty structure.

## Persistent Homology on Trajectories

Given trajectory T = {z_0, z_1, ..., z_N}:

1. Compute the Vietoris-Rips complex VR(T, ε) at multiple scales ε
2. Track homology groups:
   - H_0: connected components (clusters)
   - H_1: loops (cycles)
   - H_2: voids (cavities)
3. Record birth-death pairs (b_i, d_i) in persistence diagrams
4. Features with persistence p_i = d_i - b_i > τ_module are stable

## Module Instantiation

Each stable topological feature f with persistence p_f > τ_module is converted to a predictor head:

- **H_0 features** (connected components) → point attractor modules: local predictors centered on cluster centroids, implemented as small MLPs
- **H_1 features** (loops/cycles) → cycle modules: periodic/oscillatory predictors that capture cyclic structure
- **H_2 features** (voids) → boundary modules: surface/interface predictors that model transitions

## Convergence

The recursive loop convergence metric:

```
C(t) = |M(t) - M(t-1)| / M(t) + KL(R(t) || R(t-1)) + |S(t) - S(t-1)|
```

where:
- M(t): number of active modules at iteration t
- R(t): representation distribution at iteration t
- S(t): energy landscape smoothness at iteration t

The system converges when C(t) < ε for k consecutive iterations.
