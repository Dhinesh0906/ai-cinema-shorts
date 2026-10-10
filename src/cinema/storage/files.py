"""Atomic file helpers and the per-job directory layout."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_json(path: Path, data: Any) -> None:
    if isinstance(data, BaseModel):
        text = data.model_dump_json(indent=2)
    else:
        text = json.dumps(data, indent=2, default=_default)
    atomic_write_bytes(path, text.encode())


def _default(o: Any) -> Any:
    if isinstance(o, BaseModel):
        return o.model_dump()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class JobDirs:
    root: Path

    @property
    def project(self) -> Path:
        return self.root

    def sub(self, name: str) -> Path:
        p = self.root / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def script(self) -> Path: return self.sub("script")
    @property
    def characters(self) -> Path: return self.sub("references/characters")
    @property
    def locations(self) -> Path: return self.sub("references/locations")
    @property
    def keyframes(self) -> Path: return self.sub("references/keyframes")
    @property
    def clips(self) -> Path: return self.sub("clips")
    @property
    def voice(self) -> Path: return self.sub("audio/voice")
    @property
    def sfx(self) -> Path: return self.sub("audio/sfx")
    @property
    def music(self) -> Path: return self.sub("audio/music")
    @property
    def edit(self) -> Path: return self.sub("edit")
    @property
    def reports(self) -> Path: return self.sub("validation_reports")
    @property
    def rejected(self) -> Path: return self.sub("rejected")
    @property
    def logs(self) -> Path: return self.sub("logs")
    @property
    def temp(self) -> Path: return self.sub("temp")
    @property
    def work(self) -> Path: return self.sub("work")
