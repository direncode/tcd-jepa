"""Registry for dynamically created predictor modules.

Manages lifecycle: creation, performance tracking, pruning.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from tcd_jepa.topology.feature_extraction import TopologicalFeature


@dataclass
class ModuleRecord:
    """Tracks a registered module's metadata and performance."""
    module_id: str
    module: nn.Module
    feature: TopologicalFeature
    created_epoch: int
    total_loss: float = 0.0
    num_uses: int = 0
    last_used_epoch: int = 0

    @property
    def avg_loss(self) -> float:
        return self.total_loss / max(self.num_uses, 1)


class ModuleRegistry:
    """Manages dynamically created predictor modules.

    Tracks performance of each module and prunes underperformers.
    """

    def __init__(
        self,
        max_modules: int = 20,
        pruning_patience: int = 10,
        min_improvement: float = 0.01,
    ) -> None:
        self.max_modules = max_modules
        self.pruning_patience = pruning_patience
        self.min_improvement = min_improvement
        self._modules: dict[str, ModuleRecord] = {}
        self._next_id = 0

    def register(
        self,
        module: nn.Module,
        feature: TopologicalFeature,
        epoch: int,
    ) -> str:
        """Register a new module.

        Args:
            module: The nn.Module to register.
            feature: Topological feature that spawned it.
            epoch: Current epoch.

        Returns:
            Module ID string.
        """
        module_id = f"mod_{feature.module_type}_{self._next_id}"
        self._next_id += 1

        self._modules[module_id] = ModuleRecord(
            module_id=module_id,
            module=module,
            feature=feature,
            created_epoch=epoch,
            last_used_epoch=epoch,
        )

        # Prune if over capacity
        if len(self._modules) > self.max_modules:
            self._prune(epoch)

        return module_id

    def update_performance(
        self,
        module_id: str,
        loss: float,
        epoch: int,
    ) -> None:
        """Record a module's performance on a batch."""
        if module_id in self._modules:
            record = self._modules[module_id]
            record.total_loss += loss
            record.num_uses += 1
            record.last_used_epoch = epoch

    def get_module(self, module_id: str) -> Optional[nn.Module]:
        """Retrieve a module by ID."""
        record = self._modules.get(module_id)
        return record.module if record else None

    def get_all_modules(self) -> list[tuple[str, nn.Module]]:
        """Get all registered (id, module) pairs."""
        return [(r.module_id, r.module) for r in self._modules.values()]

    def get_active_modules(self) -> nn.ModuleList:
        """Get all active modules as a ModuleList (for parameter access)."""
        return nn.ModuleList([r.module for r in self._modules.values()])

    def _prune(self, current_epoch: int) -> list[str]:
        """Remove underperforming or stale modules.

        Returns list of pruned module IDs.
        """
        pruned = []
        if not self._modules:
            return pruned

        # Sort by average loss (worst first)
        sorted_records = sorted(
            self._modules.values(),
            key=lambda r: r.avg_loss,
            reverse=True,
        )

        # Remove worst performers until under capacity
        while len(self._modules) > self.max_modules and sorted_records:
            worst = sorted_records.pop(0)
            # Don't prune very new modules
            if current_epoch - worst.created_epoch < self.pruning_patience:
                continue
            del self._modules[worst.module_id]
            pruned.append(worst.module_id)

        # Also prune stale modules (unused for too long)
        stale = [
            mid for mid, r in self._modules.items()
            if current_epoch - r.last_used_epoch > self.pruning_patience * 2
            and r.num_uses > 0
        ]
        for mid in stale:
            del self._modules[mid]
            pruned.append(mid)

        return pruned

    def prune(self, current_epoch: int) -> list[str]:
        """Public pruning interface."""
        return self._prune(current_epoch)

    @property
    def num_modules(self) -> int:
        return len(self._modules)

    def get_statistics(self) -> dict:
        """Summary statistics for all registered modules."""
        if not self._modules:
            return {"num_modules": 0, "avg_loss": 0.0, "module_types": {}}

        type_counts: dict[str, int] = {}
        total_loss = 0.0
        total_uses = 0
        for r in self._modules.values():
            t = r.feature.module_type
            type_counts[t] = type_counts.get(t, 0) + 1
            total_loss += r.total_loss
            total_uses += r.num_uses

        return {
            "num_modules": len(self._modules),
            "avg_loss": total_loss / max(total_uses, 1),
            "module_types": type_counts,
        }
