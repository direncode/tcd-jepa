"""RunPod serverless handler for TCD-JEPA inference."""

import runpod
import torch

from ocean_core import OceanConfig, set_config
from tcd_jepa.backends.ocean import OceanBackend


_backend = None


def _init():
    global _backend
    if _backend is not None:
        return
    set_config(OceanConfig(btut_l_max=64, irdb_max_entities=100_000))
    _backend = OceanBackend(embed_dim=384)


def handler(event):
    """Handle RunPod serverless requests."""
    _init()
    action = event["input"].get("action", "query")

    if action == "ingest":
        records = event["input"]["records"]
        result = _backend.ingest(records)
        return {"status": "ok", **result}

    elif action == "evolve":
        n_steps = event["input"].get("steps", 100)
        if _backend.btut is not None:
            _backend.btut.step(n_steps=n_steps)
        return {"status": "ok", "metrics": _backend.get_landscape_metrics()}

    elif action == "state":
        return {"status": "ok", "metrics": _backend.get_landscape_metrics()}

    else:
        return {"status": "error", "message": f"Unknown action: {action}"}


runpod.serverless.start({"handler": handler})
