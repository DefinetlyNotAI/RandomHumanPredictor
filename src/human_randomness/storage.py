"""Safe atomic JSONL persistence, with separate records for real answers."""
from __future__ import annotations
import json
import os
from pathlib import Path
from .model import validate_number

DEFAULT_HOME = Path(os.environ.get("HRP_HOME", str(Path.home() / ".human-randomness")))


def profile_path(profile: str, home: Path | None = None) -> Path:
    if not profile or len(profile) > 64 or not all(c.isascii() and (c.isalnum() or c in "-_") for c in profile):
        raise ValueError("profile name must contain only ASCII letters, digits, - or _ (max 64)")
    return (home or DEFAULT_HOME) / f"{profile}.json"


def load(profile: str, home: Path | None = None) -> dict:
    path = profile_path(profile, home)
    if not path.exists():
        return {"version": 1, "rounds": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("rounds"), list):
        raise ValueError(f"Invalid profile format: {path}")
    for round_ in data["rounds"]:
        if not isinstance(round_, dict) or "actual" not in round_:
            raise ValueError("Corrupt round record")
        validate_number(round_["actual"])
    return data


def save(profile: str, data: dict, home: Path | None = None) -> Path:
    path = profile_path(profile, home)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    try:
        with temp.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    return path
