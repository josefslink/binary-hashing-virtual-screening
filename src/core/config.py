from __future__ import annotations

from pathlib import Path

# yaml is optional: the scoring paths run without it, only the config-driven entry points
# need it, so import failure is deferred to the call rather than the import.
try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_yaml_config(config_path: str | Path) -> dict:
    if yaml is None:
        raise SystemExit("PyYAML is required to load config files. Install it with `pip install pyyaml`.")
    with open(config_path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    # an empty file parses to None; callers merge this into a dict of CLI overrides
    return data or {}
