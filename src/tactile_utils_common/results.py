from __future__ import annotations

from datetime import datetime
from pathlib import Path


DEFAULT_RESULTS_ROOT = Path("results")
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def make_result_timestamp(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime(TIMESTAMP_FORMAT)


def module_result_path(
    module_name: str,
    filename: str | Path,
    *,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    timestamp: str | None = None,
    default_suffix: str | None = None,
) -> Path:
    name = _result_filename(filename, default_suffix=default_suffix)
    prefix = timestamp or make_result_timestamp()
    if not name.startswith(f"{prefix}_"):
        name = f"{prefix}_{name}"
    out_path = Path(results_root) / module_name / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def _result_filename(filename: str | Path, *, default_suffix: str | None = None) -> str:
    path = Path(filename)
    name = path.name
    if not name:
        raise ValueError("result filename must not be empty")
    if default_suffix and not Path(name).suffix:
        name = f"{name}{default_suffix}"
    return name
