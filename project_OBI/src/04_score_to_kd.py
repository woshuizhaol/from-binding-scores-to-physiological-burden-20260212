#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, DEFAULT_HSA_CALIB
from utils import geometric_mean, write_json


R_KCAL = 1.987e-3
T_K = 310.0
RT = R_KCAL * T_K
LOG10E = np.log10(np.e)

CALIB_DTXSID: Dict[str, str] = {
    "pfoa": "DTXSID8031865",
    "pfos": "DTXSID3031864",
    "pfhxs": "DTXSID7040150",
    "pfna": "DTXSID8031863",
    "pfda": "DTXSID3031860",
    "pfbs": "DTXSID5030030",
    "pfhpa": "DTXSID1037303",
    "pfosa": "DTXSID3038939",
}


@dataclass
class CalibPoint:
    target: str
    chem_alias: str
    kd_exp_m: float
    method: str


def dg_to_kd_app(delta_g: float) -> float:
    return float(np.exp(delta_g / RT))


def dg_to_log10kd_app(delta_g: float) -> float:
    return float(delta_g / RT * LOG10E)


def build_calibration_points() -> List[CalibPoint]:
    points: List[CalibPoint] = []

    # TTR IC50 data (uM) from Experimental_data.md, treated as Kd proxy.
    ttr_um = {
        "pfoa": [0.378, 0.949],
        "pfos": [0.13, 0.94],
        "pfhxs": [0.594, 0.717],
        "pfna": [1.977, 2.737],
        "pfda": [1.623, 8.954],
        "pfbs": [13.33],
        "pfhpa": [1.128, 1.565],
        "pfosa": [6.124],
    }
    for chem, vals in ttr_um.items():
        points.append(CalibPoint("TTR", chem, geometric_mean([v * 1e-6 for v in vals]), "ttr_ic50"))

    # HSA Kd data from Experimental_data.md.
    # Strong-site (fluorescence/ITC) anchors.
    hsa_strong_m = {
        "pfoa": [3.7e-6, 4.0e-5],
        "pfos": [4.5e-5],
    }
    for chem, vals in hsa_strong_m.items():
        points.append(CalibPoint("HSA", chem, geometric_mean(vals), "hsa_strong"))

    # DSF anchors (mM-range apparent Kd).
    hsa_dsf_m = {
        "pfoa": [7.9e-4],
        "pfos": [6.9e-4],
        "pfbs": [1.68e-3],
    }
    for chem, vals in hsa_dsf_m.items():
        points.append(CalibPoint("HSA", chem, geometric_mean(vals), "hsa_dsf"))

    return points


def collect_calib_rows(
    points: List[CalibPoint],
    dock_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> tuple[list[Dict[str, object]], Dict[str, List[float]]]:
    calib_rows: List[Dict[str, object]] = []
    by_target_resid: Dict[str, List[float]] = {}
    for pt in points:
        dtxsid = CALIB_DTXSID.get(pt.chem_alias)
        if not dtxsid:
            continue
        man_hits = meta_df[meta_df["DTXSID"] == dtxsid]
        if len(man_hits) != 1:
            continue
        ligand_id = str(man_hits.iloc[0]["ligand_id"])
        target_rows = dock_df[(dock_df["ligand_id"] == ligand_id) & (dock_df["target"] == pt.target)]
        if target_rows.empty:
            continue
        dg = float(target_rows["affinity_kcal_mol"].min())
        log10_app = dg_to_log10kd_app(dg)
        log10_exp = float(np.log10(pt.kd_exp_m))
        resid = log10_exp - log10_app

        by_target_resid.setdefault(pt.target, []).append(resid)
        calib_rows.append(
            {
                "target": pt.target,
                "chem_alias": pt.chem_alias,
                "dtxsid": dtxsid,
                "ligand_id": ligand_id,
                "deltaG": dg,
                "log10Kd_app": log10_app,
                "log10Kd_exp": log10_exp,
                "offset_residual": resid,
                "Kd_exp_M": pt.kd_exp_m,
                "method": pt.method,
            }
        )
    return calib_rows, by_target_resid


def compute_offsets(by_target_resid: Dict[str, List[float]]) -> tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    offset_by_target: Dict[str, float] = {}
    offset_stats: Dict[str, Dict[str, float]] = {}
    for target, vals in by_target_resid.items():
        arr = np.asarray(vals, dtype=float)
        off = float(np.median(arr))
        mad = float(np.median(np.abs(arr - off)))
        offset_by_target[target] = off
        offset_stats[target] = {
            "offset_log10": off,
            "mad_log10": mad,
            "n_points": int(arr.size),
        }
    return offset_by_target, offset_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert docking scores to Kd and calibrate with per-target offsets.")
    parser.add_argument("--state", choices=["anionic", "neutral"], default="anionic")
    parser.add_argument(
        "--hsa-calib",
        choices=["strong", "dsf"],
        default=DEFAULT_HSA_CALIB,
        help="HSA calibration mode: strong (fluorescence/ITC) or dsf (mM DSF).",
    )
    args = parser.parse_args()

    summary = DATA_PROCESSED / "docking_scores" / args.state / "docking_summary.csv"
    lig_manifest = DATA_PROCESSED / "ligands_3d" / "ligand_manifest.csv"
    out_dir = DATA_PROCESSED / "Kd_matrices" / args.state
    out_dir.mkdir(parents=True, exist_ok=True)

    dock_df = pd.read_csv(summary)
    dock_df = dock_df[dock_df["ok"] == 1].copy()
    dock_df["affinity_kcal_mol"] = dock_df["affinity_kcal_mol"].astype(float)
    dock_df["Kd_app_M"] = dock_df["affinity_kcal_mol"].map(dg_to_kd_app)
    dock_df["log10Kd_app"] = dock_df["affinity_kcal_mol"].map(dg_to_log10kd_app)
    dock_df["pKd_app"] = -dock_df["log10Kd_app"]

    meta_df = pd.read_csv(lig_manifest)
    meta_df["DTXSID"] = meta_df["DTXSID"].astype(str)

    calib_points = build_calibration_points()

    # Main calibration points for selected HSA mode.
    main_points = [
        pt
        for pt in calib_points
        if pt.target != "HSA" or (args.hsa_calib == "strong" and pt.method == "hsa_strong") or (args.hsa_calib == "dsf" and pt.method == "hsa_dsf")
    ]
    calib_rows, by_target_resid = collect_calib_rows(main_points, dock_df, meta_df)
    if len(calib_rows) < 4:
        raise RuntimeError(f"Not enough calibration points matched ({len(calib_rows)}).")

    offset_by_target, offset_stats = compute_offsets(by_target_resid)

    # Optional alternate HSA (DSF) offset for sensitivity analysis.
    hsa_dsf_rows, hsa_dsf_resid = collect_calib_rows(
        [pt for pt in calib_points if pt.target == "HSA" and pt.method == "hsa_dsf"], dock_df, meta_df
    )
    hsa_dsf_offset = None
    hsa_dsf_stats = None
    if "HSA" in hsa_dsf_resid and len(hsa_dsf_resid["HSA"]) >= 1:
        hsa_dsf_offset = float(np.median(np.asarray(hsa_dsf_resid["HSA"], dtype=float)))
        hsa_dsf_stats = {
            "offset_log10": hsa_dsf_offset,
            "mad_log10": float(np.median(np.abs(np.asarray(hsa_dsf_resid["HSA"], dtype=float) - hsa_dsf_offset))),
            "n_points": int(len(hsa_dsf_resid["HSA"])),
        }

    dock_df["target_offset_log10"] = dock_df["target"].map(offset_by_target).fillna(0.0)
    dock_df["log10Kd_calib"] = dock_df["log10Kd_app"] + dock_df["target_offset_log10"]
    dock_df["Kd_calib_M"] = np.power(10.0, dock_df["log10Kd_calib"])
    dock_df["pKd_calib"] = -dock_df["log10Kd_calib"]

    if hsa_dsf_offset is not None:
        dock_df["log10Kd_calib_dsf"] = dock_df["log10Kd_calib"]
        mask_hsa = dock_df["target"] == "HSA"
        dock_df.loc[mask_hsa, "log10Kd_calib_dsf"] = dock_df.loc[mask_hsa, "log10Kd_app"] + hsa_dsf_offset
        dock_df["Kd_calib_M_dsf"] = np.power(10.0, dock_df["log10Kd_calib_dsf"])

    # Aggregate by target.
    agg_rows: List[Dict[str, object]] = []
    for ligand_id, sub in dock_df.groupby("ligand_id"):
        row = {"ligand_id": ligand_id}
        for target, tdf in sub.groupby("target"):
            if target == "HSA":
                ka_sum = float((1.0 / tdf["Kd_calib_M"].to_numpy(dtype=float)).sum())
                kd_eff = 1.0 / ka_sum if ka_sum > 0 else np.nan
                dg_eff = float(tdf["affinity_kcal_mol"].min())
                if "Kd_calib_M_dsf" in tdf.columns:
                    ka_sum_dsf = float((1.0 / tdf["Kd_calib_M_dsf"].to_numpy(dtype=float)).sum())
                    kd_eff_dsf = 1.0 / ka_sum_dsf if ka_sum_dsf > 0 else np.nan
                    row["HSA_Kd_M_dsf"] = kd_eff_dsf
                    row["HSA_pKd_dsf"] = -np.log10(kd_eff_dsf) if kd_eff_dsf > 0 else np.nan
            else:
                best = tdf.sort_values("affinity_kcal_mol").iloc[0]
                kd_eff = float(best["Kd_calib_M"])
                dg_eff = float(best["affinity_kcal_mol"])
            row[f"{target}_Kd_M"] = kd_eff
            row[f"{target}_pKd"] = -np.log10(kd_eff) if kd_eff > 0 else np.nan
            row[f"{target}_dG_kcal_mol"] = dg_eff
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows).sort_values("ligand_id")
    agg_df.to_csv(out_dir / "kd_target_matrix.csv", index=False)
    dock_df.to_csv(out_dir / "kd_site_level.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(out_dir / "calibration_points_used.csv", index=False)
    if hsa_dsf_rows:
        pd.DataFrame(hsa_dsf_rows).to_csv(out_dir / "calibration_points_hsa_dsf.csv", index=False)

    # Global residual summary on calibration points.
    calib_used_df = pd.DataFrame(calib_rows)
    calib_used_df["offset_applied"] = calib_used_df["target"].map(offset_by_target).fillna(0.0)
    calib_used_df["log10Kd_pred_calib"] = calib_used_df["log10Kd_app"] + calib_used_df["offset_applied"]
    resid = calib_used_df["log10Kd_exp"].to_numpy(dtype=float) - calib_used_df["log10Kd_pred_calib"].to_numpy(dtype=float)
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.3

    sigma_dsf = None
    if hsa_dsf_rows:
        df_dsf = pd.DataFrame(hsa_dsf_rows)
        if hsa_dsf_offset is not None:
            resid_dsf = df_dsf["log10Kd_exp"].to_numpy(dtype=float) - (
                df_dsf["log10Kd_app"].to_numpy(dtype=float) + hsa_dsf_offset
            )
            sigma_dsf = float(np.std(resid_dsf, ddof=1)) if len(resid_dsf) > 1 else 0.3

    write_json(
        out_dir / "calibration_model.json",
        {
            "formula": "log10(Kd_M) = deltaG/(RT*ln(10)) + offset_target",
            "R_kcal_mol_K": R_KCAL,
            "T_K": T_K,
            "hsa_calibration_mode": args.hsa_calib,
            "offset_by_target": offset_by_target,
            "offset_stats": offset_stats,
            "sigma_log10_residual": sigma,
            "n_points": int(len(calib_rows)),
            "hsa_dsf_offset_log10": hsa_dsf_offset,
            "hsa_dsf_stats": hsa_dsf_stats,
            "sigma_log10_residual_hsa_dsf": sigma_dsf,
        },
    )

    print(f"Calibration points used: {len(calib_rows)}")
    print(f"Offsets: {offset_by_target}")
    print(f"sigma_log10_residual: {sigma:.4f}")
    print(f"Kd matrix: {out_dir / 'kd_target_matrix.csv'}")


if __name__ == "__main__":
    main()
