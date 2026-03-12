"""Model components for TCD-JEPA."""

# Lazy imports to avoid circular dependency with tcd_jepa.modules
def __getattr__(name):
    if name in ("TCDJEPAModel", "build_tcd_jepa"):
        from tcd_jepa.models.tcd_jepa_model import TCDJEPAModel, build_tcd_jepa
        return {"TCDJEPAModel": TCDJEPAModel, "build_tcd_jepa": build_tcd_jepa}[name]
    raise AttributeError(f"module 'tcd_jepa.models' has no attribute {name}")

__all__ = ["TCDJEPAModel", "build_tcd_jepa"]
