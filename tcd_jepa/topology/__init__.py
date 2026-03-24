"""System 3 — Module Crystallizer (topology layer).

Applies persistent homology to exploration trajectories to identify
stable topological features that become reusable predictor modules.
"""

from tcd_jepa.topology.feature_extraction import TopologicalFeature as TopologicalFeature
from tcd_jepa.topology.feature_extraction import (
    TopologicalFeatureExtractor as TopologicalFeatureExtractor,
)
from tcd_jepa.topology.persistence_diagrams import (
    PersistenceDiagramAnalyzer as PersistenceDiagramAnalyzer,
)
from tcd_jepa.topology.persistent_homology import (
    PersistentHomologyComputer as PersistentHomologyComputer,
)
