"""System 3 — Module Crystallizer.

Observes exploration trajectories from System 2, applies persistent homology
to identify stable topological features, and crystallizes them into
reusable predictor modules.
"""

from typing import Optional

import torch
import torch.nn as nn

from tcd_jepa.topology.persistent_homology import PersistentHomologyComputer
from tcd_jepa.topology.persistence_diagrams import PersistenceDiagramAnalyzer
from tcd_jepa.topology.feature_extraction import TopologicalFeatureExtractor
from tcd_jepa.modules.module_factory import ModuleFactory
from tcd_jepa.modules.module_registry import ModuleRegistry


class ModuleCrystallizer:
    """System 3: crystallizes predictor modules from trajectory topology.

    Pipeline:
    1. Receive point cloud from System 2 trajectories
    2. Compute persistent homology (Vietoris-Rips)
    3. Analyze persistence diagrams for stable features
    4. Extract topological features as module candidates
    5. Instantiate new predictor modules via factory
    6. Register modules in the registry
    """

    def __init__(
        self,
        embed_dim: int,
        persistence_threshold: float = 0.1,
        max_homology_dim: int = 2,
        max_modules: int = 20,
        max_points_for_ph: int = 500,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.embed_dim = embed_dim
        self.max_points_for_ph = max_points_for_ph
        self.device = device

        self.ph_computer = PersistentHomologyComputer(
            max_homology_dim=max_homology_dim,
        )

        self.diagram_analyzer = PersistenceDiagramAnalyzer(
            persistence_threshold=persistence_threshold,
        )

        self.feature_extractor = TopologicalFeatureExtractor(
            persistence_threshold=persistence_threshold,
        )

        self.module_factory = ModuleFactory(embed_dim)
        self.registry = ModuleRegistry(max_modules=max_modules)

        # Track crystallization history
        self._crystallization_log: list[dict] = []

    def crystallize(
        self,
        point_cloud: torch.Tensor,
        epoch: int,
    ) -> dict:
        """Run full crystallization pipeline on a point cloud.

        Args:
            point_cloud: [N, D] tensor of latent points from trajectories.
            epoch: Current training epoch.

        Returns:
            Dict with crystallization results:
                'new_modules': list of (module_id, module_type) tuples
                'ph_analysis': persistence diagram analysis
                'num_features': number of topological features found
                'total_persistence': sum of all persistence values
        """
        # Step 1: Compute persistent homology
        ph_result = self.ph_computer.compute(
            point_cloud, max_points=self.max_points_for_ph
        )

        # Step 2: Analyze persistence diagrams
        ph_analysis = self.diagram_analyzer.analyze(ph_result)

        # Step 3: Extract topological features
        features = self.feature_extractor.extract_features(
            ph_analysis, point_cloud=point_cloud
        )

        # Step 4: Create and register modules for significant features
        new_modules = []
        for feature in features:
            module = self.module_factory.create_module(feature, self.device)
            module_id = self.registry.register(module, feature, epoch)
            new_modules.append((module_id, feature.module_type))

        # Step 5: Prune underperformers
        pruned = self.registry.prune(epoch)

        # Log
        result = {
            "new_modules": new_modules,
            "pruned_modules": pruned,
            "ph_analysis": ph_analysis,
            "num_features": len(features),
            "total_persistence": self.diagram_analyzer.total_persistence(ph_result),
            "registry_stats": self.registry.get_statistics(),
        }
        self._crystallization_log.append(result)

        return result

    def get_active_modules(self) -> nn.ModuleList:
        """Get all active modules for parameter optimization."""
        return self.registry.get_active_modules()

    def update_module_performance(
        self,
        module_id: str,
        loss: float,
        epoch: int,
    ) -> None:
        """Update a module's performance tracking."""
        self.registry.update_performance(module_id, loss, epoch)

    @property
    def num_modules(self) -> int:
        return self.registry.num_modules

    @property
    def crystallization_history(self) -> list[dict]:
        return self._crystallization_log
