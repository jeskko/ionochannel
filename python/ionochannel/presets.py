"""Named channels from presets.json (shared with the Rust crate)."""
from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

from .channel import Component, Path


def _component(v: dict) -> Component:
    if v.get("specular", False):
        return Component(power=v["power"], spread_hz=0.0, shift_hz=v.get("shift_hz", 0.0), specular=True)
    return Component(power=v["power"], spread_hz=v["spread_hz"], shift_hz=v.get("shift_hz", 0.0))


@lru_cache(maxsize=1)
def presets() -> dict[str, dict]:
    """{name: {"source": str, "paths": [Path, ...]}}"""
    raw = json.loads(resources.files(__package__).joinpath("presets.json").read_text())
    out = {}
    for k, p in raw.items():
        if k.startswith("_"):
            continue
        paths = []
        for q in p["paths"]:
            comps = [_component(c) for c in q["components"]] if "components" in q else [_component(q)]
            paths.append(Path(delay_s=q["delay_ms"] * 1e-3, components=tuple(comps)))
        out[k] = {"source": p.get("source", ""), "paths": tuple(paths)}
    return out


def names() -> list[str]:
    return sorted(presets())


def preset(name: str) -> tuple[Path, ...]:
    try:
        return presets()[name]["paths"]
    except KeyError:
        raise KeyError(f"no preset {name!r}; have {', '.join(names())}") from None
