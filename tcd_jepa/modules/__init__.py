"""Predictor modules for TCD-JEPA.

Includes the vanilla JEPA predictor, dynamic predictor with crystallized
modules, and the module lifecycle management system.
"""

# Lazy imports to avoid circular dependency with tcd_jepa.models
def __getattr__(name):
    if name == "VisionTransformerPredictor":
        from tcd_jepa.modules.predictor import VisionTransformerPredictor
        return VisionTransformerPredictor
    elif name == "ModuleFactory":
        from tcd_jepa.modules.module_factory import ModuleFactory
        return ModuleFactory
    elif name == "AttractorModule":
        from tcd_jepa.modules.module_factory import AttractorModule
        return AttractorModule
    elif name == "CycleModule":
        from tcd_jepa.modules.module_factory import CycleModule
        return CycleModule
    elif name == "BoundaryModule":
        from tcd_jepa.modules.module_factory import BoundaryModule
        return BoundaryModule
    elif name == "ModuleRegistry":
        from tcd_jepa.modules.module_registry import ModuleRegistry
        return ModuleRegistry
    elif name == "DynamicPredictor":
        from tcd_jepa.modules.dynamic_predictor import DynamicPredictor
        return DynamicPredictor
    raise AttributeError(f"module 'tcd_jepa.modules' has no attribute {name}")
