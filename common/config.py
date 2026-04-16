import yaml
from copy import deepcopy


def load_config(path: str) -> dict:
    """Load YAML file and return a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _coerce(value: str):
    """Coerce a CLI override string to bool/int/float/str."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def apply_overrides(cfg: dict, overrides: list) -> dict:
    """Apply 'dotted.path=value' overrides to cfg. Raises KeyError if the path
    does not already exist — overrides must patch known keys."""
    out = deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        parts = key.split(".")
        cursor = out
        for p in parts[:-1]:
            if not isinstance(cursor, dict) or p not in cursor:
                raise KeyError(f"override path not found: {key}")
            cursor = cursor[p]
        last = parts[-1]
        if not isinstance(cursor, dict) or last not in cursor:
            raise KeyError(f"override path not found: {key}")
        cursor[last] = _coerce(raw)
    return out
