#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyreadstat

from config import DATA_PROCESSED, NHANES_ANALYTES, SOURCE_DIR


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cdf = np.cumsum(weights)
    if cdf[-1] <= 0:
        return float(np.nan)
    cdf /= cdf[-1]
    return float(np.interp(quantile, cdf, values))


def match_ligand_id(manifest: pd.DataFrame, patterns: List[str]) -> Optional[Tuple[str, float]]:
    for pat in patterns:
        mask = manifest["preferred_name"].str.lower().str.contains(pat, na=False)
        if mask.any():
            row = manifest[mask].iloc[0]
            return str(row["ligand_id"]), float(row["mw"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Process NHANES PFAS exposure and fit concentration distributions.")
    parser.add_argument("--xpt", type=str, default=str(SOURCE_DIR / "P_PFAS.xpt"))
    args = parser.parse_args()

    out_dir = DATA_PROCESSED / "exposure_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, meta = pyreadstat.read_xport(args.xpt)
    manifest = pd.read_csv(DATA_PROCESSED / "ligands_3d" / "ligand_manifest.csv")
    manifest["preferred_name"] = manifest["preferred_name"].astype(str)

    # Map NHANES analytes to ligands in docking list.
    patterns = {
        "LBXPFDE": ["perfluorodecanoic acid"],
        "LBXPFHS": ["perfluorohexane sulfonic acid"],
        "LBXMPAH": ["methyl", "pfosa"],
        "LBXPFNA": ["perfluorononanoic acid"],
        "LBXPFUA": ["perfluoroundecanoic acid"],
        "LBXNFOA": ["perfluorooctanoic acid"],
        "LBXBFOA": ["perfluorooctanoic acid"],
        "LBXNFOS": ["perfluorooctane sulfonic acid"],
        "LBXMFOS": ["perfluorooctane sulfonic acid"],
    }
    mw_fallback = {
        "LBXPFDE": 514.08,
        "LBXPFHS": 400.12,
        "LBXMPAH": 585.11,
        "LBXPFNA": 464.09,
        "LBXPFUA": 564.10,
        "LBXNFOA": 414.07,
        "LBXBFOA": 414.07,
        "LBXNFOS": 500.13,
        "LBXMFOS": 500.13,
    }

    map_rows: List[Dict[str, object]] = []
    ligand_for_var: Dict[str, str] = {}
    mw_for_var: Dict[str, float] = {}
    for var in NHANES_ANALYTES:
        m = match_ligand_id(manifest, patterns[var])
        if m:
            ligand_id, mw = m
        else:
            ligand_id, mw = "", mw_fallback[var]
        ligand_for_var[var] = ligand_id
        mw_for_var[var] = mw
        map_rows.append(
            {
                "variable": var,
                "label": meta.column_names_to_labels.get(var, ""),
                "ligand_id": ligand_id,
                "mw_g_mol": mw,
            }
        )
    pd.DataFrame(map_rows).to_csv(out_dir / "nhanes_variable_to_ligand.csv", index=False)

    weights = df["WTSBAPRP"].fillna(0.0).to_numpy(dtype=float)
    summary_rows: List[Dict[str, object]] = []
    molar_df = pd.DataFrame({"SEQN": df["SEQN"], "WTSBAPRP": df["WTSBAPRP"]})

    for var in NHANES_ANALYTES:
        comment_var = var.replace("LBX", "LBD") + "L"
        values = df[var].copy()
        comments = df[comment_var] if comment_var in df.columns else pd.Series(np.zeros(len(df)))

        # If below LOD (comment code==1), NHANES already substitutes LOD/sqrt(2).
        valid = values.notna() & (values > 0)
        vals = values[valid].to_numpy(dtype=float)
        w = weights[valid.to_numpy()]
        if len(vals) == 0:
            continue

        mw = mw_for_var[var]
        conc_m = vals * 1e-6 / mw  # ng/mL -> mol/L
        q50 = weighted_quantile(conc_m, w, 0.50)
        q95 = weighted_quantile(conc_m, w, 0.95)
        q99 = weighted_quantile(conc_m, w, 0.99)
        ln_c = np.log(np.clip(conc_m, 1e-18, None))
        mu = float(np.average(ln_c, weights=w))
        sigma = float(np.sqrt(np.average((ln_c - mu) ** 2, weights=w)))

        summary_rows.append(
            {
                "variable": var,
                "label": meta.column_names_to_labels.get(var, ""),
                "ligand_id": ligand_for_var[var],
                "mw_g_mol": mw,
                "n_valid": int(valid.sum()),
                "below_lod_n": int((comments == 1).sum()) if comment_var in df.columns else np.nan,
                "median_M": q50,
                "p95_M": q95,
                "p99_M": q99,
                "lognormal_mu_ln": mu,
                "lognormal_sigma_ln": sigma,
            }
        )

        col_name = f"{var}_M"
        molar_col = values.astype(float) * 1e-6 / mw
        molar_df[col_name] = molar_col

    # Aggregate PFOA/PFOS isomers.
    if {"LBXNFOA_M", "LBXBFOA_M"} <= set(molar_df.columns):
        molar_df["PFOA_total_M"] = molar_df["LBXNFOA_M"].fillna(0.0) + molar_df["LBXBFOA_M"].fillna(0.0)
    if {"LBXNFOS_M", "LBXMFOS_M"} <= set(molar_df.columns):
        molar_df["PFOS_total_M"] = molar_df["LBXNFOS_M"].fillna(0.0) + molar_df["LBXMFOS_M"].fillna(0.0)

    summary_df = pd.DataFrame(summary_rows).sort_values("variable")
    summary_df.to_csv(out_dir / "nhanes_analyte_summary.csv", index=False)
    molar_df.to_csv(out_dir / "exposure_samples_molar.csv", index=False)

    # Correlation matrix for multivariate sampling.
    expo_cols = [c for c in molar_df.columns if c.endswith("_M")]
    corr = molar_df[expo_cols].corr(method="spearman")
    corr.to_csv(out_dir / "exposure_spearman_corr.csv")

    print(f"NHANES samples: {len(df)}")
    print(f"Analytes processed: {len(summary_df)}")
    print(f"Exposure summary: {out_dir / 'nhanes_analyte_summary.csv'}")


if __name__ == "__main__":
    main()

