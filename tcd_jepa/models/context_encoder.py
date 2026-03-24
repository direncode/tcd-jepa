"""Context encoder with System 1 instrumentation hooks.

Wraps the VisionTransformer to track embedding flow statistics,
representation diversity, and information flow rates.
"""

from typing import Optional

import torch
import torch.nn as nn

from tcd_jepa.models.vision_transformer import VisionTransformer


class ContextEncoder(nn.Module):
    """System 1 — Knowledge Corpus Releaser.

    Wraps a VisionTransformer encoder and instruments it with hooks to monitor
    representation statistics at each layer. This data feeds into System 2's
    exploration decisions in later phases.

    Hooks are disabled by default and must be explicitly enabled via
    ``enable_hooks()`` to avoid graph breaks in ``torch.compile`` and
    unnecessary CPU–GPU synchronisation during training.
    """

    def __init__(self, encoder: VisionTransformer):
        super().__init__()
        self.encoder = encoder
        # Statistics collected during forward pass
        self._layer_stats: list[dict[str, torch.Tensor]] = []
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._hooks_enabled = False

    @property
    def embed_dim(self) -> int:
        return self.encoder.embed_dim

    @property
    def num_heads(self) -> int:
        return self.encoder.num_heads

    @property
    def patch_embed(self) -> nn.Module:
        return self.encoder.patch_embed

    def enable_hooks(self) -> None:
        """Enable layer-statistics hooks (adds CPU–GPU sync overhead)."""
        if not self._hooks_enabled:
            self._remove_hooks()
            for i, block in enumerate(self.encoder.blocks):
                hook = block.register_forward_hook(self._make_hook(i))
                self._hooks.append(hook)
            self._hooks_enabled = True

    def disable_hooks(self) -> None:
        """Disable hooks to allow clean torch.compile graphs."""
        self._remove_hooks()
        self._hooks_enabled = False

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _make_hook(self, layer_idx: int):
        """Create a hook that records statistics for a specific layer."""
        def hook_fn(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
            with torch.no_grad():
                if isinstance(output, torch.Tensor):
                    stats = {
                        "mean": output.mean().item(),
                        "std": output.std().item(),
                        "norm": output.norm(dim=-1).mean().item(),
                    }
                    # Pad stats list if needed
                    while len(self._layer_stats) <= layer_idx:
                        self._layer_stats.append({})
                    self._layer_stats[layer_idx] = stats
        return hook_fn

    def get_layer_stats(self) -> list[dict[str, float]]:
        """Return statistics from the most recent forward pass."""
        return list(self._layer_stats)

    def clear_stats(self) -> None:
        """Clear collected statistics."""
        self._layer_stats.clear()

    def forward(
        self, x: torch.Tensor, masks: Optional[list[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Encode images with optional statistics collection.

        Args:
            x: Input images [B, C, H, W].
            masks: Optional list of index tensors for patch selection.

        Returns:
            Encoded patch representations.
        """
        if self._hooks_enabled:
            self._layer_stats.clear()
        return self.encoder(x, masks=masks)

    def parameters(self, recurse: bool = True):
        return self.encoder.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self.encoder.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.encoder.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.encoder.load_state_dict(state_dict, strict=strict)

    def modules(self):
        return self.encoder.modules()
