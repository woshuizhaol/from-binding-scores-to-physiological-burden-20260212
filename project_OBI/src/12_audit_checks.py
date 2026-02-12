#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Dict

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, MODELS, RESULTS


def relative_error(a: float, b: float) -> float:
    if b == 0:
        return float("inf") if a != 0 else 0.0
    return abs(a - b) / abs(b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run consistency audits for OBI pipeline outputs.")
    parser.add_argument("--state", choices=["anionic", "neutral"], default="anionic")
    parser.add_argument("--out-tag", type=str, default="", help="Optional tag appended to indices output folder name.")
    args = parser.parse_args()

    tag = f"_{args.out_tag}" if args.out_tag else ""
    idx_dir = DATA_PROCESSED / "indices_OBI" / f"{args.state}{tag}"
    audit_dir = RESULTS / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    checks: Dict[str, Dict[str, object]] = {}

    pop_path = idx_dir / "population_index_summary.csv"
    contrib_path = idx_dir / "ligand_contributions.csv"
    if pop_path.exists() and contrib_path.exists():
        pop = pd.read_csv(pop_path)
        contrib = pd.read_csv(contrib_path)
        pop_map = {r["metric"]: float(r["weighted_mean"]) for _, r in pop.iterrows()}
        t_sum = float(contrib["TTII_contribution"].sum())
        r_sum = float(contrib["RTII_contribution"].sum())
        obi_sum = 0.5 * (t_sum + r_sum)

        checks["ttii_contrib_sum"] = {
            "value": t_sum,
            "target": pop_map.get("TTII", float("nan")),
            "rel_error": relative_error(t_sum, pop_map.get("TTII", 0.0)),
            "pass": relative_error(t_sum, pop_map.get("TTII", 0.0)) < 1e-3,
        }
        checks["rtii_contrib_sum"] = {
            "value": r_sum,
            "target": pop_map.get("RTII", float("nan")),
            "rel_error": relative_error(r_sum, pop_map.get("RTII", 0.0)),
            "pass": relative_error(r_sum, pop_map.get("RTII", 0.0)) < 1e-3,
        }
        checks["obi_contrib_sum"] = {
            "value": obi_sum,
            "target": pop_map.get("OBI_total", float("nan")),
            "rel_error": relative_error(obi_sum, pop_map.get("OBI_total", 0.0)),
            "pass": relative_error(obi_sum, pop_map.get("OBI_total", 0.0)) < 1e-3,
        }
    else:
        checks["contrib_files"] = {
            "pass": False,
            "note": "population_index_summary.csv or ligand_contributions.csv missing",
        }

    # Pocket sequence audit.
    pocket_path = MODELS / "kronrls" / args.state / "pocket_sequences.csv"
    if pocket_path.exists():
        seq = pd.read_csv(pocket_path)
        min_len = int(seq["seq_len"].min()) if not seq.empty else 0
        n_x = int((seq["sequence"] == "X").sum()) if not seq.empty else 0
        checks["pocket_sequence"] = {
            "min_len": min_len,
            "n_x": n_x,
            "pass": min_len > 0 and n_x == 0,
        }
    else:
        checks["pocket_sequence"] = {"pass": False, "note": "pocket_sequences.csv missing"}

    # Redocking audit.
    redock_path = RESULTS / "redocking" / "redocking_summary.csv"
    if redock_path.exists():
        redock = pd.read_csv(redock_path)
        cols = set(redock.columns)
        has_min = "rmsd_min_A" in cols and "mode_of_best_rmsd" in cols
        num_modes_ok = True
        if "num_modes" in cols:
            num_modes_ok = bool((redock["num_modes"] > 1).all())
        checks["redocking_modes"] = {
            "has_rmsd_min": has_min,
            "num_modes_gt1": num_modes_ok,
            "pass": has_min and num_modes_ok,
        }
    else:
        checks["redocking_modes"] = {"pass": False, "note": "redocking_summary.csv missing"}

    # MC summary audit.
    mc_path = idx_dir / "population_index_summary_mc.csv"
    checks["mc_summary"] = {"pass": mc_path.exists(), "path": str(mc_path)}

    summary = {
        "state": args.state,
        "out_tag": args.out_tag,
        "checks": checks,
    }

    out_json = audit_dir / "audit_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    out_md = audit_dir / "audit_summary.md"
    rows = []
    for key, val in checks.items():
        row = {"check": key, **{k: v for k, v in val.items() if k != "note"}}
        if "note" in val:
            row["note"] = val["note"]
        rows.append(row)
    out_md.write_text(pd.DataFrame(rows).to_markdown(index=False), encoding="utf-8")

    print(f"Audit summary: {out_json}")


if __name__ == "__main__":
    main()
