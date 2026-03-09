"""Predictor modules for TCD-JEPA.

Includes the vanilla JEPA predictor, dynamic predictor with crystallized
modules, and the module lifecycle management system.
"""

from tcd_jepa.modules.predictor import VisionTransformerPredictor
from tcd_jepa.modules.module_factory import ModuleFactory, AttractorModule, CycleModule, BoundaryModule
from tcd_jepa.modules.module_registry import ModuleRegistry
from tcd_jepa.modules.dynamic_predictor import DynamicPredictor
