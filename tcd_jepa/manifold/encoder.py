"""Manifold-native encoder for TCD-JEPA / Latent Ocean Layer 3.

Replaces the vision-specific PatchEmbed + 2D sincos positional encoding with:
1. ManifoldEmbed: projects 384D sentence-transformer fingerprints into embedding space
2. SphericalPositionalEncoding: encodes S² positions (radius 4.5) via spherical harmonics
3. VelocityEncoding: encodes tangent-space velocities as temporal dynamics signal
4. CausalAdjacencyEncoding: encodes directed causal graph structure (FK/semantic/URL/cosine links)
5. ManifoldTransformer: full transformer encoder operating on manifold entity tokens

Each "token" is an entity at a point on S², analogous to a patch at a grid position.
Attention captures inter-entity relationships; TCD crystallizer discovers the topology.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from tcd_jepa.utils.tensors import apply_masks, trunc_normal_


class ManifoldEmbed(nn.Module):
    """Projects sentence-transformer fingerprints into transformer embedding space.

    Latent Ocean uses all-MiniLM-L6-v2 producing 384D embeddings. This module
    learns a task-specific projection that captures manifold structure beyond
    raw semantic similarity.
    """

    def __init__(self, fingerprint_dim: int = 384, embed_dim: int = 192, num_tokens: int = 64):
        super().__init__()
        self.fingerprint_dim = fingerprint_dim
        self.embed_dim = embed_dim
        self.num_patches = num_tokens  # compatibility with existing predictor code
        self.proj = nn.Sequential(
            nn.Linear(fingerprint_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project fingerprints [B, N, 384] -> [B, N, embed_dim]."""
        return self.proj(x)


def spherical_harmonics_encoding(
    coords: torch.Tensor,
    embed_dim: int,
    sphere_radius: float = 4.5,
    max_degree: int = 8,
) -> torch.Tensor:
    """Encode S² positions using spherical harmonics-inspired basis functions.

    Latent Ocean entities live on S² at radius 4.5. Positions are (x,y,z) Cartesian.
    We convert to (theta, phi) and encode with multi-frequency sin/cos basis that
    respects spherical geometry: nearby points get similar encodings, antipodal
    points get maximally different ones.

    Args:
        coords: Cartesian coordinates [B, N, 3] on S² (radius ~4.5).
        embed_dim: Output embedding dimension.
        sphere_radius: Radius of the sphere (4.5 in Latent Ocean).
        max_degree: Maximum frequency degree.

    Returns:
        Positional encoding [B, N, embed_dim].
    """
    B, N, _ = coords.shape
    device = coords.device

    # Normalize to unit sphere for angle extraction
    coords_norm = coords / (coords.norm(dim=-1, keepdim=True) + 1e-8)
    x, y, z = coords_norm[..., 0], coords_norm[..., 1], coords_norm[..., 2]

    # Spherical coordinates
    theta = torch.acos(z.clamp(-1 + 1e-6, 1 - 1e-6))  # [B, N] polar angle [0, pi]
    phi = torch.atan2(y, x)  # [B, N] azimuthal angle [-pi, pi]

    # Radial distance (captures deviations from exact sphere)
    r = coords.norm(dim=-1) / sphere_radius  # [B, N] normalized radius

    # Multi-frequency encoding for theta, phi, and r
    freqs_per_coord = embed_dim // 6  # 3 coords * 2 (sin+cos)
    if freqs_per_coord < 1:
        freqs_per_coord = 1

    freqs = torch.logspace(0, math.log10(max_degree), freqs_per_coord, device=device)
    freqs = freqs.view(1, 1, -1)  # [1, 1, F]

    encodings = []
    for signal in [theta, phi, r]:
        s = signal.unsqueeze(-1)  # [B, N, 1]
        encodings.append(torch.sin(s * freqs * math.pi))
        encodings.append(torch.cos(s * freqs * math.pi))

    encoding = torch.cat(encodings, dim=-1)  # [B, N, 6*F]

    # Pad or trim to exact embed_dim
    if encoding.shape[-1] < embed_dim:
        pad = torch.zeros(B, N, embed_dim - encoding.shape[-1], device=device)
        encoding = torch.cat([encoding, pad], dim=-1)
    elif encoding.shape[-1] > embed_dim:
        encoding = encoding[:, :, :embed_dim]

    return encoding


class VelocityEncoding(nn.Module):
    """Encodes tangent-space velocities as temporal dynamics signal.

    Latent Ocean entities have (N,3) velocities tangent to S². These encode
    the current dynamics: direction of drift, speed of evolution, convergence
    vs. exploration. The JEPA predictor uses this to forecast next-state.
    """

    def __init__(self, embed_dim: int, velocity_dim: int = 3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(velocity_dim + 2, embed_dim),  # +2 for speed and angular_speed
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, velocity: torch.Tensor, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode velocities into embedding space.

        Args:
            velocity: Tangent-space velocities [B, N, 3].
            coords: Optional positions [B, N, 3] for computing angular velocity.

        Returns:
            Velocity encoding [B, N, embed_dim].
        """
        speed = velocity.norm(dim=-1, keepdim=True)  # [B, N, 1]

        # Angular speed: how fast the entity moves on the sphere surface
        if coords is not None:
            r = coords.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            angular_speed = speed / r
        else:
            angular_speed = speed

        features = torch.cat([velocity, speed, angular_speed], dim=-1)  # [B, N, 5]
        return self.proj(features)


class CausalAdjacencyEncoding(nn.Module):
    """Encodes directed causal graph structure around each entity.

    Latent Ocean discovers 4 types of causal links:
    - FK links (strength 0.9) — foreign key relationships
    - Semantic links (strength 0.7) — related_to, depends_on, tags
    - URL hierarchy (strength 0.6) — parent-path extraction
    - Embedding similarity (strength 0.65+) — cosine sim above threshold

    This module encodes the local causal neighborhood structure as additional
    positional information for the transformer.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Features: [in_degree, out_degree, weighted_in, weighted_out,
        #            causal_depth, local_density, pagerank, link_diversity]
        self.struct_proj = nn.Sequential(
            nn.Linear(8, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

    def forward(
        self,
        adjacency: torch.Tensor,
        causal_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode causal adjacency structure.

        Args:
            adjacency: Weighted directed adjacency [B, N, N], strength in [0, 1].
            causal_features: Optional pre-computed features [B, N, 8].

        Returns:
            Causal encoding [B, N, embed_dim].
        """
        if causal_features is not None:
            return self.struct_proj(causal_features)

        B, N, _ = adjacency.shape

        adj_binary = (adjacency.abs() > 1e-6).float()

        # Degree features (binary)
        in_degree = adj_binary.sum(dim=1) / max(N, 1)
        out_degree = adj_binary.sum(dim=2) / max(N, 1)

        # Weighted degree (preserves link strength information)
        weighted_in = adjacency.sum(dim=1) / max(N, 1)
        weighted_out = adjacency.sum(dim=2) / max(N, 1)

        # Causal depth (reachability via 3-step power iteration)
        adj_power = adj_binary.clone()
        depth_sum = adj_binary.clone()
        for _ in range(min(3, N)):
            adj_power = torch.bmm(adj_power, adj_binary).clamp(max=1.0)
            depth_sum = depth_sum + adj_power
        causal_depth = depth_sum.sum(dim=2) / max(N, 1)

        # Local clustering density
        local_density = torch.bmm(adj_binary, adj_binary.transpose(1, 2))
        local_density = local_density.diagonal(dim1=1, dim2=2) / max(N, 1)

        # PageRank approximation
        pr = torch.ones(B, N, device=adjacency.device) / N
        for _ in range(5):
            pr = 0.15 / N + 0.85 * torch.bmm(
                adj_binary.transpose(1, 2), pr.unsqueeze(-1)
            ).squeeze(-1)
            pr = pr / (pr.sum(dim=-1, keepdim=True) + 1e-8)

        # Link diversity: entropy of link strength distribution per node
        link_strengths = adjacency.clone()
        link_strengths[link_strengths == 0] = 1e-8
        row_sums = link_strengths.sum(dim=2, keepdim=True).clamp(min=1e-8)
        link_probs = link_strengths / row_sums
        link_diversity = -(link_probs * (link_probs + 1e-8).log()).sum(dim=2) / math.log(max(N, 2))

        features = torch.stack([
            in_degree, out_degree, weighted_in, weighted_out,
            causal_depth, local_density, pr, link_diversity,
        ], dim=-1)

        return self.struct_proj(features)


class ManifoldTransformer(nn.Module):
    """Transformer encoder for Latent Ocean manifold data.

    Replaces VisionTransformer with manifold-native components:
    - ManifoldEmbed: 384D sentence-transformer → embed_dim
    - Spherical positional encoding: S² positions at radius 4.5
    - Velocity encoding: tangent-space dynamics as temporal signal
    - Causal adjacency encoding: directed graph structure
    - Same transformer blocks (attention is geometry-agnostic)

    Compatible with existing ContextEncoder/TargetEncoder wrappers.
    """

    def __init__(
        self,
        fingerprint_dim: int = 384,
        num_tokens: int = 64,
        coord_dim: int = 3,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type = nn.LayerNorm,
        init_std: float = 0.02,
        use_causal_encoding: bool = True,
        use_velocity_encoding: bool = True,
        sphere_radius: float = 4.5,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.fingerprint_dim = fingerprint_dim
        self.coord_dim = coord_dim
        self.use_causal_encoding = use_causal_encoding
        self.use_velocity_encoding = use_velocity_encoding
        self.sphere_radius = sphere_radius

        # Manifold-native embedding (384D → embed_dim)
        self.patch_embed = ManifoldEmbed(fingerprint_dim, embed_dim, num_tokens)

        # Spherical positional encoding
        self.pos_scale = nn.Parameter(torch.ones(1))
        self.pos_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Velocity encoding (temporal dynamics)
        if use_velocity_encoding:
            self.velocity_encoding = VelocityEncoding(embed_dim)
            self.velocity_gate = nn.Parameter(torch.tensor(0.3))

        # Causal adjacency encoding
        if use_causal_encoding:
            self.causal_encoding = CausalAdjacencyEncoding(embed_dim)
            self.causal_gate = nn.Parameter(torch.tensor(0.5))

        # Transformer blocks (reused from vision ViT — attention is geometry-agnostic)
        from tcd_jepa.models.vision_transformer import Block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        masks: Optional[list[torch.Tensor]] = None,
        coords: Optional[torch.Tensor] = None,
        adjacency: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
        causal_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode manifold entity tokens.

        Args:
            x: Fingerprint vectors [B, N, 384].
            masks: Optional index tensors for token selection.
            coords: S² positions [B, N, 3] (Cartesian, radius ~4.5).
            adjacency: Weighted directed adjacency [B, N, N].
            velocity: Tangent-space velocities [B, N, 3].
            causal_features: Pre-computed causal features [B, N, 8].

        Returns:
            Encoded representations [B', N', D].
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        # Project fingerprints
        x = self.patch_embed(x)
        B, N, D = x.shape

        # Add spherical positional encoding
        if coords is not None:
            pos = spherical_harmonics_encoding(coords, D, sphere_radius=self.sphere_radius)
            x = x + self.pos_scale * self.pos_proj(pos)

        # Add velocity encoding (temporal dynamics)
        if self.use_velocity_encoding and velocity is not None:
            vel_enc = self.velocity_encoding(velocity, coords)
            gate = torch.sigmoid(self.velocity_gate)
            x = x + gate * vel_enc

        # Add causal adjacency encoding
        if self.use_causal_encoding and (adjacency is not None or causal_features is not None):
            if adjacency is not None:
                causal_enc = self.causal_encoding(adjacency, causal_features)
            else:
                causal_enc = self.causal_encoding(
                    torch.zeros(B, N, N, device=x.device), causal_features
                )
            gate = torch.sigmoid(self.causal_gate)
            x = x + gate * causal_enc

        # Apply masks (same mechanism as ViT — select subset of tokens)
        if masks is not None:
            x = apply_masks(x, masks)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x
