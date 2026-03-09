"""Configuration loading and management using PyYAML.

Uses a simple dict-based config with dot-notation access via DotDict.
OmegaConf can be used as an optional upgrade when available.
"""

import copy
from pathlib import Path
from typing import Any, Optional, Union

import yaml


class DotDict(dict):
    """Dictionary with dot-notation access for nested keys."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict) and not isinstance(val, DotDict):
                val = DotDict(val)
                self[key] = val
            return val
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")


def load_config(path: Union[str, Path]) -> DotDict:
    """Load a YAML config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return DotDict(data or {})


def merge_configs(*configs: dict) -> DotDict:
    """Deep merge multiple configs, later configs override earlier ones."""
    result = {}
    for cfg in configs:
        _deep_update(result, cfg)
    return DotDict(result)


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively update base dict with override dict."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def config_from_cli(dotlist: Optional[list[str]] = None) -> DotDict:
    """Create a config from CLI-style dotlist overrides (e.g., ['model.depth=6'])."""
    if dotlist is None:
        return DotDict()
    result = {}
    for item in dotlist:
        key, value = item.split("=", 1)
        # Try to parse value as Python literal
        try:
            value = yaml.safe_load(value)
        except yaml.YAMLError:
            pass
        # Set nested key
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return DotDict(result)


def load_config_with_overrides(
    config_path: Union[str, Path],
    overrides: Optional[list[str]] = None,
) -> DotDict:
    """Load a config file and apply CLI overrides."""
    cfg = load_config(config_path)
    if overrides:
        cli_cfg = config_from_cli(overrides)
        cfg = merge_configs(cfg, cli_cfg)
    return cfg


def get_nested(cfg: dict, key: str, default: Any = None) -> Any:
    """Safely get a nested config value using dot notation."""
    parts = key.split(".")
    d = cfg
    for part in parts:
        if isinstance(d, dict) and part in d:
            d = d[part]
        else:
            return default
    return d
