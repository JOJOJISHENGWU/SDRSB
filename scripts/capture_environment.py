from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    files = [path for path in Path("data").rglob("*") if path.is_file() and path.name != "MANIFEST.json"]
    record = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "files": {str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in files}}
    Path("environment.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("wrote environment.json")
