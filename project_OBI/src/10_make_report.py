#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from config import DATA_PROCESSED, MODELS, NHANES_WEIGHT_VAR, RESULTS


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def try_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def metric_table(scope_metrics: Dict[str, object]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for k, v in scope_metrics.items():
        if k == "n_matched":
            continue
        rows.append({"metric": k, "value": v})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact markdown report for the full pipeline.")
    parser.add_argument("--state", choices=["anionic", "neutral"], default="anionic")
    args = parser.parse_args()

    idx_dir = DATA_PROCESSED / "indices_OBI" / args.state
    val_dir = DATA_PROCESSED / "validation" / args.state
    qsar_dir = MODELS / "qsar" / args.state
    kron_dir = MODELS / "kronrls" / args.state
    redock_dir = RESULTS / "redocking"

    out_md = RESULTS / "reports" / f"pipeline_summary_{args.state}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)

    pop = try_read_csv(idx_dir / "population_index_summary.csv")
    pop_mc = try_read_csv(idx_dir / "population_index_summary_mc.csv")
    weights_used = read_json(idx_dir / "thyroid_weights_used.json")
    top = try_read_csv(idx_dir / "top1pct_drivers.csv")
    val_metrics = read_json(val_dir / "validation_metrics.json")
    qsar_metrics = try_read_csv(qsar_dir / "metrics_OBI_ref.csv")
    qsar_metrics_summary = try_read_csv(qsar_dir / "metrics_OBI_ref_summary.csv")
    qsar_shap = try_read_csv(qsar_dir / "rf_shap_importance_OBI_ref.csv")
    qsar_summary = read_json(qsar_dir / "summary_OBI_ref.json")
    kron_metrics = try_read_csv(kron_dir / "loto_metrics.csv")
    kron_reg = try_read_csv(kron_dir / "reg_selection.csv")
    redock_metrics = read_json(redock_dir / "redocking_metrics.json")
    redock_summary = try_read_csv(redock_dir / "redocking_summary.csv")
    audit_summary = read_json(RESULTS / "audit" / "audit_summary.json")

    lines: List[str] = []
    lines.append(f"# From Binding Scores to Physiological Burden ({args.state})")
    lines.append("")

    lines.append("## Population OBI Summary")
    lines.append(f"- NHANES weight variable: `{NHANES_WEIGHT_VAR}`")
    if not pop.empty:
        lines.append(pop.to_markdown(index=False))
    else:
        lines.append("No population index summary found.")
    if not pop_mc.empty:
        lines.append("")
        lines.append("### Monte Carlo Summary")
        lines.append(pop_mc.to_markdown(index=False))
    if weights_used:
        lines.append("")
        lines.append("### Thyroid Weights Used")
        lines.append(pd.DataFrame([weights_used]).to_markdown(index=False))
    lines.append("")

    lines.append("## Top 1% Driver PFAS")
    if not top.empty:
        cols = [c for c in ["ligand_id", "DTXSID", "preferred_name", "TTII_contribution", "RTII_contribution", "mean_contribution"] if c in top.columns]
        lines.append(top[cols].to_markdown(index=False))
    else:
        lines.append("No top driver table found.")
    lines.append("")

    lines.append("## External Validation Metrics")
    if val_metrics:
        if "assay_proxy_note" in val_metrics:
            lines.append(f"- potency proxy: {val_metrics['assay_proxy_note']}")
        if "concentration_units_seen" in val_metrics:
            lines.append(f"- assay concentration units: {val_metrics['concentration_units_seen']}")
        for scope, payload in val_metrics.items():
            if not isinstance(payload, dict) or "n_matched" not in payload:
                continue
            lines.append("")
            lines.append(f"### {scope}")
            lines.append(f"- n_matched: {payload['n_matched']}")
            mtab = metric_table(payload)
            if not mtab.empty:
                lines.append(mtab.to_markdown(index=False))
    else:
        lines.append("No validation metrics found.")
    lines.append("")

    lines.append("## QSAR Metrics (OBI_ref)")
    if not qsar_metrics_summary.empty:
        lines.append(qsar_metrics_summary.to_markdown(index=False))
    elif not qsar_metrics.empty:
        lines.append(qsar_metrics.to_markdown(index=False))
    else:
        lines.append("No QSAR metrics found.")

    if not qsar_metrics.empty and "repeat" in qsar_metrics.columns:
        lines.append("")
        lines.append("### QSAR Repeats (Raw)")
        lines.append(qsar_metrics.to_markdown(index=False))
    lines.append("")
    if qsar_summary:
        lines.append(f"- split strategy: {qsar_summary.get('split_strategy')}")
        lines.append(f"- repeats: {qsar_summary.get('n_repeats')}")
        lines.append(f"- shap generated: {qsar_summary.get('shap_generated')}")

    if not qsar_shap.empty:
        lines.append("")
        lines.append("### QSAR SHAP (RandomForest)")
        lines.append(qsar_shap.head(20).to_markdown(index=False))
    lines.append("")

    lines.append("## Kron-RLS LOTO Metrics")
    if not kron_metrics.empty:
        lines.append(kron_metrics.to_markdown(index=False))
    else:
        lines.append("No Kron-RLS metrics found.")
    if not kron_reg.empty:
        lines.append("")
        lines.append("### Kron-RLS Reg Selection")
        lines.append(kron_reg.to_markdown(index=False))
    lines.append("")

    lines.append("## Redocking RMSD")
    if redock_metrics:
        rows = [{"metric": k, "value": v} for k, v in redock_metrics.items()]
        lines.append(pd.DataFrame(rows).to_markdown(index=False))
    else:
        lines.append("No redocking metrics found.")

    if not redock_summary.empty:
        cols = [
            c
            for c in [
                "target",
                "selected_residue",
                "ligand_resname",
                "best_affinity_kcal_mol",
                "rmsd_mode1_A",
                "rmsd_min_A",
                "mode_of_best_rmsd",
                "pass_rmsd_mode1",
                "pass_rmsd_min",
                "status",
                "error",
            ]
            if c in redock_summary.columns
        ]
        lines.append("")
        lines.append(redock_summary[cols].to_markdown(index=False))

    lines.append("")
    lines.append("## Audit Summary")
    if audit_summary:
        rows = [{"check": k, **v} for k, v in audit_summary.get("checks", {}).items()]
        if rows:
            lines.append(pd.DataFrame(rows).to_markdown(index=False))
        else:
            lines.append(pd.DataFrame([audit_summary]).to_markdown(index=False))
    else:
        lines.append("No audit summary found.")

    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report generated: {out_md}")


if __name__ == "__main__":
    main()
