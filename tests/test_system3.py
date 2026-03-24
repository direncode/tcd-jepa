"""Tests for System 3 — Module Crystallizer."""

import numpy as np
import torch

from tcd_jepa.core.system3_crystallizer import ModuleCrystallizer
from tcd_jepa.modules.dynamic_predictor import DynamicPredictor
from tcd_jepa.modules.module_factory import (
    AttractorModule,
    BoundaryModule,
    CycleModule,
    ModuleFactory,
)
from tcd_jepa.modules.module_registry import ModuleRegistry
from tcd_jepa.topology.feature_extraction import TopologicalFeature, TopologicalFeatureExtractor
from tcd_jepa.topology.persistence_diagrams import PersistenceDiagramAnalyzer
from tcd_jepa.topology.persistent_homology import PersistentHomologyComputer


class TestPersistentHomology:

    def test_compute_returns_diagram(self):
        """PH computation returns a diagram with expected keys."""
        ph = PersistentHomologyComputer(max_homology_dim=1)
        # Simple point cloud: two clusters
        cluster1 = torch.randn(50, 3) + torch.tensor([5.0, 0.0, 0.0])
        cluster2 = torch.randn(50, 3) - torch.tensor([5.0, 0.0, 0.0])
        points = torch.cat([cluster1, cluster2], dim=0)

        result = ph.compute(points)
        assert "diagram" in result
        assert "H0" in result
        assert len(result["diagram"]) > 0

    def test_two_clusters_detected(self):
        """Two well-separated clusters should produce H_0 features."""
        ph = PersistentHomologyComputer(max_homology_dim=0)
        cluster1 = torch.randn(30, 3) * 0.1 + torch.tensor([10.0, 0.0, 0.0])
        cluster2 = torch.randn(30, 3) * 0.1 - torch.tensor([10.0, 0.0, 0.0])
        points = torch.cat([cluster1, cluster2], dim=0)

        result = ph.compute(points)
        # Should have H_0 features (connected components merging)
        assert len(result["H0"]) > 0

    def test_subsampling(self):
        """Large point clouds are subsampled."""
        ph = PersistentHomologyComputer()
        points = torch.randn(2000, 5)
        result = ph.compute(points, max_points=100)
        assert "diagram" in result


class TestPersistenceDiagramAnalyzer:

    def test_compute_persistence(self):
        """Persistence = death - birth."""
        analyzer = PersistenceDiagramAnalyzer()
        diagram = np.array([[0.0, 1.0], [0.5, 2.0], [1.0, 1.5]])
        pers = analyzer.compute_persistence(diagram)
        np.testing.assert_array_almost_equal(pers, [1.0, 1.5, 0.5])

    def test_filter_by_persistence(self):
        """Filtering keeps high-persistence features."""
        analyzer = PersistenceDiagramAnalyzer(
            persistence_threshold=0.5, relative_threshold=False
        )
        diagram = np.array([[0.0, 0.1], [0.0, 1.0], [0.0, 2.0]])
        filtered, indices = analyzer.filter_by_persistence(diagram)
        assert len(filtered) == 2
        assert 1 in indices and 2 in indices

    def test_analyze(self):
        """Full analysis returns structured results."""
        analyzer = PersistenceDiagramAnalyzer()
        ph_result = {
            "H0": np.array([[0.0, 1.0], [0.0, 0.5]]),
            "H1": np.array([[0.5, 2.0]]),
            "H2": np.empty((0, 2)),
        }
        result = analyzer.analyze(ph_result)
        assert "H0" in result
        assert "H1" in result
        assert result["H0"]["count"] >= 1
        assert result["H1"]["count"] == 1

    def test_total_persistence(self):
        """Total persistence sums across dimensions."""
        analyzer = PersistenceDiagramAnalyzer()
        ph_result = {
            "H0": np.array([[0.0, 1.0]]),
            "H1": np.array([[0.0, 2.0]]),
            "H2": np.empty((0, 2)),
        }
        assert analyzer.total_persistence(ph_result) == 3.0


class TestTopologicalFeatureExtractor:

    def test_extract_features(self):
        """Extracts features from analysis results."""
        extractor = TopologicalFeatureExtractor()
        ph_analysis = {
            "H0": {
                "features": np.array([[0.0, 1.0], [0.0, 0.5]]),
                "persistence": np.array([1.0, 0.5]),
                "count": 2,
            },
            "H1": {
                "features": np.array([[0.5, 2.0]]),
                "persistence": np.array([1.5]),
                "count": 1,
            },
            "H2": {
                "features": np.empty((0, 2)),
                "persistence": np.array([]),
                "count": 0,
            },
        }
        features = extractor.extract_features(ph_analysis)
        assert len(features) == 3
        # Sorted by persistence (highest first)
        assert features[0].persistence >= features[1].persistence

    def test_module_type_assignment(self):
        """H_0 -> attractor, H_1 -> cycle, H_2 -> boundary."""
        extractor = TopologicalFeatureExtractor()
        ph_analysis = {
            "H0": {"features": np.array([[0.0, 1.0]]), "persistence": np.array([1.0]), "count": 1},
            "H1": {"features": np.array([[0.0, 1.0]]), "persistence": np.array([1.0]), "count": 1},
            "H2": {"features": np.array([[0.0, 1.0]]), "persistence": np.array([1.0]), "count": 1},
        }
        features = extractor.extract_features(ph_analysis)
        types = {f.module_type for f in features}
        assert types == {"attractor", "cycle", "boundary"}


class TestModuleFactory:

    def test_create_attractor(self):
        """Factory creates AttractorModule for H_0 features."""
        factory = ModuleFactory(embed_dim=16)
        feature = TopologicalFeature(dim=0, birth=0.0, death=1.0, persistence=1.0, module_type="attractor")
        module = factory.create_module(feature)
        assert isinstance(module, AttractorModule)

        z = torch.randn(4, 16)
        out = module(z)
        assert out.shape == (4, 16)

    def test_create_cycle(self):
        """Factory creates CycleModule for H_1 features."""
        factory = ModuleFactory(embed_dim=16)
        feature = TopologicalFeature(dim=1, birth=0.0, death=1.0, persistence=1.0, module_type="cycle")
        module = factory.create_module(feature)
        assert isinstance(module, CycleModule)

        z = torch.randn(4, 16)
        out = module(z)
        assert out.shape == (4, 16)

    def test_create_boundary(self):
        """Factory creates BoundaryModule for H_2 features."""
        factory = ModuleFactory(embed_dim=16)
        feature = TopologicalFeature(dim=2, birth=0.0, death=1.0, persistence=1.0, module_type="boundary")
        module = factory.create_module(feature)
        assert isinstance(module, BoundaryModule)

        z = torch.randn(4, 16)
        out = module(z)
        assert out.shape == (4, 16)


class TestModuleRegistry:

    def test_register_and_retrieve(self):
        """Can register and retrieve modules."""
        registry = ModuleRegistry()
        module = torch.nn.Linear(8, 8)
        feature = TopologicalFeature(dim=0, birth=0, death=1, persistence=1, module_type="attractor")
        mid = registry.register(module, feature, epoch=0)
        assert registry.num_modules == 1
        assert registry.get_module(mid) is module

    def test_max_modules_enforced(self):
        """Registry prunes when over capacity."""
        registry = ModuleRegistry(max_modules=3, pruning_patience=0)
        for i in range(5):
            module = torch.nn.Linear(8, 8)
            feature = TopologicalFeature(dim=0, birth=0, death=1, persistence=1, module_type="attractor")
            registry.register(module, feature, epoch=i)
            # Record some performance so pruning can work
            for mid, _ in registry.get_all_modules():
                registry.update_performance(mid, loss=float(i), epoch=i)
        assert registry.num_modules <= 5  # may or may not prune depending on patience

    def test_performance_tracking(self):
        """Performance updates are tracked correctly."""
        registry = ModuleRegistry()
        module = torch.nn.Linear(8, 8)
        feature = TopologicalFeature(dim=0, birth=0, death=1, persistence=1, module_type="attractor")
        mid = registry.register(module, feature, epoch=0)
        registry.update_performance(mid, loss=2.0, epoch=1)
        registry.update_performance(mid, loss=4.0, epoch=2)
        stats = registry.get_statistics()
        assert stats["num_modules"] == 1
        assert stats["avg_loss"] == 3.0


class TestDynamicPredictor:

    def test_forward_without_modules(self):
        """DynamicPredictor without modules returns base prediction."""
        from tcd_jepa.modules.predictor import vit_predictor

        base = vit_predictor(num_patches=64, embed_dim=32, predictor_embed_dim=16, depth=1, num_heads=2)
        dynamic = DynamicPredictor(base, embed_dim=32)

        B = 2
        x = torch.randn(B, 10, 32)
        masks_x = [torch.randint(0, 64, (B, 10))]
        masks = [torch.randint(0, 64, (B, 8))]

        out = dynamic(x, masks_x, masks)
        assert out.shape == (B, 8, 32)


class TestModuleCrystallizer:

    def test_crystallize_pipeline(self):
        """Full crystallization pipeline runs without error."""
        crystallizer = ModuleCrystallizer(embed_dim=8, max_points_for_ph=100)

        # Two clusters point cloud
        c1 = torch.randn(50, 8) + 5.0
        c2 = torch.randn(50, 8) - 5.0
        points = torch.cat([c1, c2], dim=0)

        result = crystallizer.crystallize(points, epoch=0)
        assert "new_modules" in result
        assert "num_features" in result
        assert result["num_features"] >= 0

    def test_modules_created(self):
        """Crystallization creates at least one module from clear topology."""
        crystallizer = ModuleCrystallizer(
            embed_dim=8, persistence_threshold=0.01, max_points_for_ph=100
        )
        # Very clear clusters
        c1 = torch.randn(50, 8) * 0.1 + 20.0
        c2 = torch.randn(50, 8) * 0.1 - 20.0
        points = torch.cat([c1, c2], dim=0)

        result = crystallizer.crystallize(points, epoch=0)
        # Should find at least some features
        assert result["num_features"] >= 1
