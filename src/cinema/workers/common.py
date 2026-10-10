"""Shared worker I/O. A worker reads a JSON request file, processes a batch, writes a JSON response
file incrementally (so a crash keeps finished items), then exits; exiting is what frees Metal memory."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


def run_batch(handler_factory: Callable[[dict], Callable[[dict], dict]]) -> None:
    req_path, resp_path = Path(sys.argv[1]), Path(sys.argv[2])
    request = json.loads(req_path.read_text())
    results: dict[str, Any] = {}
    if resp_path.exists():
        try:
            results = json.loads(resp_path.read_text()).get("results", {})
        except ValueError:
            results = {}
    handle = handler_factory(request.get("options", {}))
    for item in request["items"]:
        if item["id"] in results and "error" not in results[item["id"]]:
            continue
        try:
            results[item["id"]] = handle(item)
        except Exception as exc:  # recorded per item; caller decides
            results[item["id"]] = {"error": f"{type(exc).__name__}: {exc}"}
        _write(resp_path, {"results": results})
        print(f"[worker] done {item['id']}", flush=True)
    _write(resp_path, {"results": results, "complete": True})


def _write(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)
