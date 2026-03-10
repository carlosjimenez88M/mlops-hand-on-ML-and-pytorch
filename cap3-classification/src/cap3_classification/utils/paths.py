"""Filesystem helpers."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Locate the project root by walking upwards until `pyproject.toml` is found."""
    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if candidate.is_dir() and (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root")


def resolve_path(path_value: str | Path, root_path: Path) -> Path:
    """Resolve a project-relative path against the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else (root_path / path).resolve()


def ensure_parent(path: Path) -> None:
    """Create the parent directory for a file path."""
    path.parent.mkdir(parents=True, exist_ok=True)
