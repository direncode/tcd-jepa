"""System 3 — Module Crystallizer (topology layer).

Applies persistent homology to exploration trajectories to identify
stable topological features that become reusable predictor modules.
"""

from tcd_jepa.topology.persistent_homology import PersistentHomologyComputer
from tcd_jepa.topology.persistence_diagrams import PersistenceDiagramAnalyzer
from tcd_jepa.topology.feature_extraction import TopologicalFeatureExtractor, TopologicalFeature
