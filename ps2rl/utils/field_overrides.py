"""``KEY=VALUE`` command-line overrides for dataclass configs (e.g. the landing BCBF settings)."""

from __future__ import annotations

from dataclasses import fields
from typing import Any


def _parse_bool(key: str, val: str) -> bool:
    low = val.lower()
    if low not in ("1", "0", "true", "false", "yes", "no", "on", "off"):
        raise ValueError(f"{key} expects a boolean, got '{val}'")
    return low in ("1", "true", "yes", "on")


def parse_field_overrides(items: list[str] | None, cls, extra: dict[str, type] | None = None) -> dict[str, Any]:
    """Parse ``KEY=VALUE`` strings into typed overrides for the dataclass ``cls``.

    Types follow the field defaults, or the annotation for fields defaulting to None
    (bool accepts 1/0/true/false/yes/no/on/off). ``extra`` adds non-field names with a type.
    Unknown keys raise.
    """
    out: dict[str, Any] = {}
    defaults = {f.name: f.default for f in fields(cls)}
    annotations = {f.name: str(f.type) for f in fields(cls)}
    extra = dict(extra or {})
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' is not KEY=VALUE")
        key, val = (t.strip() for t in item.split("=", 1))
        if key in extra:
            out[key] = _parse_bool(key, val) if extra[key] is bool else extra[key](val)
            continue
        if key not in defaults:
            raise KeyError(f"'{key}' is not a field of {cls.__name__}")
        d = defaults[key]
        if d is None:
            ann = annotations[key]
            if "bool" in ann:
                out[key] = _parse_bool(key, val)
            elif "int" in ann and "float" not in ann:
                out[key] = int(val)
            elif "str" in ann and "float" not in ann:
                out[key] = val
            else:
                out[key] = float(val)
            continue
        if isinstance(d, bool):
            out[key] = _parse_bool(key, val)
        elif isinstance(d, int) and not isinstance(d, bool):
            out[key] = int(val)
        elif isinstance(d, float):
            out[key] = float(val)
        else:
            out[key] = type(d)(val)
    return out


def overrides_tag(overrides: dict[str, Any]) -> str:
    """Short, filesystem-safe tag for a set of overrides (empty if none)."""
    def fmt(v):
        if isinstance(v, bool):
            return "on" if v else "off"
        return f"{v:g}" if isinstance(v, float) else str(v)
    return "_".join(f"{k}-{fmt(v)}" for k, v in sorted(overrides.items()))
