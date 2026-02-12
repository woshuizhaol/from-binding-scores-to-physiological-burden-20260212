from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def sanitize_id(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return value.strip("_") or "unnamed"


def ligand_center_from_pdb(pdb_path: Path, ligand_resname: str) -> Tuple[float, float, float]:
    coords: List[Tuple[float, float, float]] = []
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("HETATM", "ATOM")):
                continue
            resname = line[17:20].strip()
            if resname != ligand_resname:
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append((x, y, z))
    if not coords:
        raise ValueError(f"Ligand {ligand_resname} not found in {pdb_path}")
    arr = np.asarray(coords, dtype=float)
    center = arr.mean(axis=0)
    return float(center[0]), float(center[1]), float(center[2])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def geometric_mean(values: Sequence[float], eps: float = 1e-18) -> float:
    arr = np.asarray(values, dtype=float)
    arr = np.clip(arr, eps, None)
    return float(math.exp(np.log(arr).mean()))

