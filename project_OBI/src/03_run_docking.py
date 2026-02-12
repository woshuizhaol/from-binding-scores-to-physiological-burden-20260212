#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from config import DATA_PROCESSED
from utils import ensure_dirs


def parse_best_affinity(text: str) -> Optional[float]:
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2 and re.fullmatch(r"\d+", parts[0]):
            try:
                return float(parts[1])
            except ValueError:
                continue
    return None


def run_one(job: Tuple[Dict[str, object], str, Path, Path, int, int, int]) -> Dict[str, object]:
    box, ligand_id, ligand_path, out_dir, exhaustiveness, num_modes, seed = job
    ensure_dirs([out_dir])
    out_pdbqt = out_dir / f"{ligand_id}.pdbqt"
    log_file = out_dir / f"{ligand_id}.log"

    cmd = [
        "vina",
        "--receptor",
        str(box["receptor_pdbqt"]),
        "--ligand",
        str(ligand_path),
        "--center_x",
        str(box["center_x"]),
        "--center_y",
        str(box["center_y"]),
        "--center_z",
        str(box["center_z"]),
        "--size_x",
        str(box["size_x"]),
        "--size_y",
        str(box["size_y"]),
        "--size_z",
        str(box["size_z"]),
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        str(num_modes),
        "--cpu",
        "1",
        "--seed",
        str(seed),
        "--out",
        str(out_pdbqt),
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    log_text = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    log_file.write_text(log_text, encoding="utf-8")
    affinity = parse_best_affinity(log_text)
    ok = proc.returncode == 0 and affinity is not None and not math.isnan(affinity)

    return {
        "site_id": box["site_id"],
        "target": box["target"],
        "axis": box.get("axis", ""),
        "ligand_id": ligand_id,
        "ligand_pdbqt": str(ligand_path),
        "out_pdbqt": str(out_pdbqt),
        "log_file": str(log_file),
        "affinity_kcal_mol": affinity if affinity is not None else "",
        "elapsed_s": round(elapsed, 3),
        "ok": int(ok),
        "stderr": proc.stderr.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run large-scale docking across all PFAS and target sites.")
    parser.add_argument("--state", choices=["anionic", "neutral"], default="anionic")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--num-modes", type=int, default=9)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    lig_manifest = DATA_PROCESSED / "ligands_3d" / "ligand_manifest.csv"
    lig_dir = DATA_PROCESSED / "ligands_3d" / "pdbqt" / args.state
    boxes_json = DATA_PROCESSED / "receptors_prepared" / "boxes" / "docking_boxes.json"
    out_root = DATA_PROCESSED / "docking_scores" / args.state
    ensure_dirs([out_root])

    lig_df = pd.read_csv(lig_manifest)
    with boxes_json.open("r", encoding="utf-8") as handle:
        boxes = json.load(handle)["boxes"]

    jobs: List[Tuple[Dict[str, object], str, Path, Path, int, int, int]] = []
    for _, row in lig_df.iterrows():
        ligand_id = row["ligand_id"]
        lig_path = lig_dir / f"{ligand_id}.pdbqt"
        if not lig_path.exists():
            continue
        for box in boxes:
            out_dir = out_root / box["site_id"]
            log_path = out_dir / f"{ligand_id}.log"
            if log_path.exists() and not args.overwrite:
                continue
            jobs.append((box, ligand_id, lig_path, out_dir, args.exhaustiveness, args.num_modes, args.seed))

    print(f"Queued docking jobs: {len(jobs)}")

    results: List[Dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Docking[{args.state}]"):
            results.append(fut.result())

    # Read existing results from previous runs and merge by site+ligand.
    summary_path = out_root / "docking_summary.csv"
    old_rows: List[Dict[str, object]] = []
    if summary_path.exists():
        old_rows = pd.read_csv(summary_path).to_dict(orient="records")
    merged = {(r["site_id"], r["ligand_id"]): r for r in old_rows}
    for r in results:
        merged[(r["site_id"], r["ligand_id"])] = r

    final_rows = list(merged.values())
    if not final_rows:
        print("No docking rows generated.")
        return
    final_rows.sort(key=lambda x: (x["site_id"], x["ligand_id"]))
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)

    fail_path = out_root / "docking_failures.tsv"
    with fail_path.open("w", encoding="utf-8") as handle:
        for r in final_rows:
            if int(r["ok"]) == 0:
                handle.write(f"{r['site_id']}\t{r['ligand_id']}\t{r.get('stderr', '')}\n")

    success = sum(int(r["ok"]) for r in final_rows)
    print(f"Docking summary rows: {len(final_rows)}")
    print(f"Successful rows: {success}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
