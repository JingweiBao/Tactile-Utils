from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


VALID_SIDES = {"left", "right"}


@dataclass(frozen=True)
class SensorMount:
    name: str
    source_key: str
    side: str
    parent_link: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    size_mm: tuple[float, float]
    grid_shape: tuple[int, int]
    semantic_region: str

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "SensorMount":
        missing = [
            key
            for key in [
                "source_key",
                "side",
                "parent_link",
                "xyz",
                "rpy",
                "size_mm",
                "grid_shape",
                "semantic_region",
            ]
            if key not in data
        ]
        if missing:
            raise ValueError(f"sensor {name!r} missing required keys: {missing}")
        side = str(data["side"])
        if side not in VALID_SIDES:
            raise ValueError(f"sensor {name!r} has invalid side {side!r}")
        return cls(
            name=name,
            source_key=str(data["source_key"]),
            side=side,
            parent_link=str(data["parent_link"]),
            xyz=_tuple_float(data["xyz"], 3, name, "xyz"),
            rpy=_tuple_float(data["rpy"], 3, name, "rpy"),
            size_mm=_tuple_float(data["size_mm"], 2, name, "size_mm"),
            grid_shape=_tuple_int(data["grid_shape"], 2, name, "grid_shape"),
            semantic_region=str(data["semantic_region"]),
        )


@dataclass(frozen=True)
class SensorMountConfig:
    sensors: tuple[SensorMount, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SensorMountConfig":
        raw_sensors = data.get("sensors")
        if not isinstance(raw_sensors, dict) or not raw_sensors:
            raise ValueError("mount config must contain a non-empty 'sensors' mapping")
        sensors = tuple(SensorMount.from_dict(name, value) for name, value in raw_sensors.items())
        names = [sensor.name for sensor in sensors]
        source_keys = [sensor.source_key for sensor in sensors]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate sensor names: {duplicates}")
        duplicate_sources = sorted({key for key in source_keys if source_keys.count(key) > 1})
        if duplicate_sources:
            raise ValueError(f"duplicate source keys: {duplicate_sources}")
        return cls(sensors=sensors)

    def by_source_key(self) -> dict[str, SensorMount]:
        return {sensor.source_key: sensor for sensor in self.sensors}

    def source_keys(self) -> set[str]:
        return {sensor.source_key for sensor in self.sensors}


def load_sensor_mount_config(path: str | Path) -> SensorMountConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return SensorMountConfig.from_dict(data)


def _tuple_float(value: Any, length: int, name: str, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name}.{field} must be a list of length {length}")
    return tuple(float(v) for v in value)


def _tuple_int(value: Any, length: int, name: str, field: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name}.{field} must be a list of length {length}")
    out = tuple(int(v) for v in value)
    if any(v <= 0 for v in out):
        raise ValueError(f"{name}.{field} must be positive")
    return out
