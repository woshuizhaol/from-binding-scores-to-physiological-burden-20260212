#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from config import DATA_PROCESSED, SOURCE_DIR


def unit_to_molar_factor(unit: str) -> float:
    u = str(unit).strip().lower()
    if u in {"m", "mol/l", "mol l-1"}:
        return 1.0
    if u in {"mm", "mmol/l", "mmol l-1"}:
        return 1e-3
    if u in {"um", "µm", "μm", "umol/l", "umol l-1"}:
        return 1e-6
    if u in {"nm", "nmol/l", "nmol l-1"}:
        return 1e-9
    if u in {"pm", "pmol/l", "pmol l-1"}:
        return 1e-12
    raise RuntimeError(f"Unsupported concentration unit: {unit}")


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return {"rho": float("nan"), "p": float("nan")}
    rho, pval = spearmanr(x, y)
    return {"rho": float(rho), "p": float(pval)}


def metric_pack(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    out = {}
    if len(np.unique(y_true)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def summarize_external_assay(cr_df: pd.DataFrame, enzyme_whitelist: Optional[Set[str]] = None) -> pd.DataFrame:
    cr = cr_df.copy()
    if enzyme_whitelist is not None:
        cr = cr[cr["enzyme"].isin(enzyme_whitelist)].copy()

    rows: List[Dict[str, object]] = []
    for (dtxsid, enzyme), sub in cr.groupby(["DTXSID", "enzyme"]):
        sub = sub.sort_values("Conc_M")
        activity = sub["Median % Activity"].to_numpy(dtype=float)
        conc_m = sub["Conc_M"].to_numpy(dtype=float)
        abs_act = np.abs(activity)

        max_abs = float(np.max(abs_act))
        idx_max = int(np.argmax(abs_act))
        conc_at_max_m = float(conc_m[idx_max])

        x = np.log10(np.clip(conc_m, 1e-18, None))
        auc_abs = float(np.trapz(abs_act, x)) if len(abs_act) >= 2 else float(abs_act[0])

        active20 = int(max_abs >= 20.0)
        active30 = int(max_abs >= 30.0)

        above20 = conc_m[abs_act >= 20.0]
        above30 = conc_m[abs_act >= 30.0]
        min_active20_m = float(np.min(above20)) if len(above20) > 0 else np.nan
        min_active30_m = float(np.min(above30)) if len(above30) > 0 else np.nan

        pacc20 = float(-np.log10(min_active20_m)) if np.isfinite(min_active20_m) and min_active20_m > 0 else np.nan
        pacc30 = float(-np.log10(min_active30_m)) if np.isfinite(min_active30_m) and min_active30_m > 0 else np.nan

        rows.append(
            {
                "DTXSID": dtxsid,
                "enzyme": enzyme,
                "max_abs_activity": max_abs,
                "auc_abs_activity": auc_abs,
                "conc_at_max_M": conc_at_max_m,
                "active20": active20,
                "active30": active30,
                "min_active20_M": min_active20_m,
                "min_active30_M": min_active30_m,
                "pacc20": pacc20,
                "pacc30": pacc30,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "DTXSID",
                "thyroid_activity_score",
                "thyroid_auc_score",
                "any_active20",
                "any_active30",
                "potency_pacc20",
                "potency_pacc30",
                "n_enzymes",
            ]
        )

    per_enzyme = pd.DataFrame(rows)
    per_chem = (
        per_enzyme.groupby("DTXSID")
        .agg(
            thyroid_activity_score=("max_abs_activity", "max"),
            thyroid_auc_score=("auc_abs_activity", "mean"),
            any_active20=("active20", "max"),
            any_active30=("active30", "max"),
            potency_pacc20=("pacc20", "max"),
            potency_pacc30=("pacc30", "max"),
            n_enzymes=("enzyme", "nunique"),
        )
        .reset_index()
    )
    return per_chem


def eval_scope(ibp: pd.DataFrame, assay: pd.DataFrame) -> Dict[str, object]:
    merged = ibp.merge(assay, on="DTXSID", how="inner")
    metrics: Dict[str, float] = {}

    if not merged.empty:
        prevalence20 = float(np.mean(merged["any_active20"].to_numpy(dtype=int)))
        prevalence30 = float(np.mean(merged["any_active30"].to_numpy(dtype=int)))
        metrics["prevalence_active20"] = prevalence20
        metrics["prevalence_active30"] = prevalence30
        metrics["baseline_pr_auc_active20"] = prevalence20
        metrics["baseline_pr_auc_active30"] = prevalence30

    for endpoint in ["TTII_ref", "RTII_ref", "OBI_ref"]:
        if endpoint not in merged.columns:
            continue
        x = merged[endpoint].to_numpy(dtype=float)

        y_activity = merged["thyroid_activity_score"].to_numpy(dtype=float)
        sp_act = safe_spearman(x, y_activity)
        metrics[f"{endpoint}_spearman_activity_rho"] = sp_act["rho"]
        metrics[f"{endpoint}_spearman_activity_p"] = sp_act["p"]

        y_auc = merged["thyroid_auc_score"].to_numpy(dtype=float)
        sp_auc = safe_spearman(x, y_auc)
        metrics[f"{endpoint}_spearman_auc_rho"] = sp_auc["rho"]
        metrics[f"{endpoint}_spearman_auc_p"] = sp_auc["p"]

        cls20 = metric_pack(merged["any_active20"].to_numpy(dtype=int), x)
        metrics[f"{endpoint}_roc_auc_active20"] = cls20["roc_auc"]
        metrics[f"{endpoint}_pr_auc_active20"] = cls20["pr_auc"]

        cls30 = metric_pack(merged["any_active30"].to_numpy(dtype=int), x)
        metrics[f"{endpoint}_roc_auc_active30"] = cls30["roc_auc"]
        metrics[f"{endpoint}_pr_auc_active30"] = cls30["pr_auc"]

        mask20 = merged["potency_pacc20"].notna().to_numpy()
        metrics[f"{endpoint}_n_pacc20"] = int(mask20.sum())
        if mask20.sum() >= 3:
            sp_p20 = safe_spearman(x[mask20], merged.loc[mask20, "potency_pacc20"].to_numpy(dtype=float))
            metrics[f"{endpoint}_spearman_pacc20_rho"] = sp_p20["rho"]
            metrics[f"{endpoint}_spearman_pacc20_p"] = sp_p20["p"]
        else:
            metrics[f"{endpoint}_spearman_pacc20_rho"] = float("nan")
            metrics[f"{endpoint}_spearman_pacc20_p"] = float("nan")

        mask30 = merged["potency_pacc30"].notna().to_numpy()
        metrics[f"{endpoint}_n_pacc30"] = int(mask30.sum())
        if mask30.sum() >= 3:
            sp_p30 = safe_spearman(x[mask30], merged.loc[mask30, "potency_pacc30"].to_numpy(dtype=float))
            metrics[f"{endpoint}_spearman_pacc30_rho"] = sp_p30["rho"]
            metrics[f"{endpoint}_spearman_pacc30_p"] = sp_p30["p"]
        else:
            metrics[f"{endpoint}_spearman_pacc30_rho"] = float("nan")
            metrics[f"{endpoint}_spearman_pacc30_p"] = float("nan")

    return {"n_matched": int(len(merged)), "metrics": metrics, "merged": merged}


def main() -> None:
    parser = argparse.ArgumentParser(description="External validation against EPA PFAS thyroid in vitro screening.")
    parser.add_argument("--state", choices=["anionic", "neutral"], default="anionic")
    args = parser.parse_args()

    excel = SOURCE_DIR / "EPA PFAS thyroid in vitro screening（Excel）.xlsx"
    ibp = pd.read_csv(DATA_PROCESSED / "indices_OBI" / args.state / "intrinsic_burden_potential.csv")

    out_dir = DATA_PROCESSED / "validation" / args.state
    out_dir.mkdir(parents=True, exist_ok=True)

    cr = pd.read_excel(excel, sheet_name="Concentration Response")
    if "DTXSID" not in cr.columns:
        raise RuntimeError("DTXSID column missing in external validation sheet.")

    cr["DTXSID"] = cr["DTXSID"].astype(str)
    cr["enzyme"] = cr["enzyme"].astype(str)
    cr["Median % Activity"] = pd.to_numeric(cr["Median % Activity"], errors="coerce")
    cr["Conc"] = pd.to_numeric(cr["Conc"], errors="coerce")
    cr = cr.dropna(subset=["DTXSID", "enzyme", "Median % Activity", "Conc", "Conc_unit"]).copy()
    cr["Conc_M"] = [float(v) * unit_to_molar_factor(u) for v, u in zip(cr["Conc"], cr["Conc_unit"])]

    final_metrics: Dict[str, object] = {
        "assay_proxy_note": "pACC20/pACC30 derived as -log10(min concentration in M reaching |Median % Activity| >= 20/30 in concentration-response data)",
        "concentration_units_seen": sorted(set(str(x) for x in cr["Conc_unit"].unique())),
    }

    assay_all = summarize_external_assay(cr, enzyme_whitelist=None)
    all_eval = eval_scope(ibp, assay_all)
    all_eval["merged"].to_csv(out_dir / "validation_merged.csv", index=False)
    all_eval["merged"].sort_values("OBI_ref", ascending=False).to_csv(out_dir / "validation_ranked_by_obi.csv", index=False)
    final_metrics["all_enzymes"] = {"n_matched": all_eval["n_matched"], **all_eval["metrics"]}

    human_enzymes = {e for e in cr["enzyme"].astype(str).unique() if not e.lower().startswith("x")}
    assay_human = summarize_external_assay(cr, enzyme_whitelist=human_enzymes)
    human_eval = eval_scope(ibp, assay_human)
    human_eval["merged"].to_csv(out_dir / "validation_merged_human_only.csv", index=False)
    final_metrics["human_only"] = {"n_matched": human_eval["n_matched"], **human_eval["metrics"]}

    assay_transport = summarize_external_assay(cr, enzyme_whitelist={"hTBG", "hTTR"})
    trans_eval = eval_scope(ibp, assay_transport)
    trans_eval["merged"].to_csv(out_dir / "validation_merged_transport_binding.csv", index=False)
    final_metrics["transport_binding"] = {"n_matched": trans_eval["n_matched"], **trans_eval["metrics"]}

    (out_dir / "validation_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")

    print(f"Validation matched compounds (all enzymes): {all_eval['n_matched']}")
    print(f"Metrics: {out_dir / 'validation_metrics.json'}")


if __name__ == "__main__":
    main()
