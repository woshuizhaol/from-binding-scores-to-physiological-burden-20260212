#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from typing import Dict, List

import numpy as np
import pandas as pd

from config import (
    DATA_PROCESSED,
    NHANES_ANALYTE_SPECS,
    NHANES_WEIGHT_VAR,
    OBI_COMPONENT_WEIGHTS,
    PHYSIOLOGY,
    PHYSIOLOGY_RANGES,
    RENAL_WEIGHTS,
    THYROID_WEIGHTS,
)


def safe_get(row: pd.Series, key: str, default: float = np.nan) -> float:
    value = row.get(key, default)
    if pd.isna(value):
        return default
    return float(value)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cdf = np.cumsum(weights)
    if cdf[-1] <= 0:
        return float(np.nan)
    cdf /= cdf[-1]
    return float(np.interp(quantile, cdf, values))


def solve_t4_free(cl_thy: Dict[str, float], kd_t4: Dict[str, float], physiology: Dict[str, float] | None = None) -> float:
    phys = PHYSIOLOGY if physiology is None else physiology
    t4_total = phys["T4_total_M"]
    protein_total = {
        "TBG": phys["TBG_M"],
        "TTR": phys["TTR_M"],
        "HSA": phys["HSA_M"],
    }

    def f(t: float) -> float:
        bound = 0.0
        for target in ["TBG", "TTR", "HSA"]:
            kd = kd_t4[target]
            cl = cl_thy.get(target, 0.0)
            theta_t4 = (t / kd) / (1.0 + t / kd + cl)
            bound += protein_total[target] * theta_t4
        return t + bound - t4_total

    lo = 0.0
    t4_free_guess = phys.get("T4_free_M", PHYSIOLOGY["T4_free_M"])
    hi = max(t4_free_guess * 10.0, t4_total)
    f_hi = f(hi)
    guard = 0
    while f_hi < 0 and guard < 80:
        hi *= 2.0
        f_hi = f(hi)
        guard += 1
    if f_hi < 0:
        raise RuntimeError("Failed to bracket T4_free root.")

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < 1e-18:
            return float(mid)
        if f_mid > 0:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


def normalized_weighted_mean(values: Dict[str, float], weights: Dict[str, float], keys: List[str]) -> float:
    wsum = sum(weights[k] for k in keys)
    if wsum <= 0:
        raise RuntimeError(f"Invalid weights: sum <= 0 for keys {keys}")
    return float(sum(weights[k] * values[k] for k in keys) / wsum)


def derive_thyroid_weights(kd_t4: Dict[str, float], physiology: Dict[str, float]) -> Dict[str, float]:
    baseline_cl = {"TBG": 0.0, "TTR": 0.0, "HSA": 0.0}
    t4_free = solve_t4_free(baseline_cl, kd_t4, physiology)
    bound = {}
    total_bound = 0.0
    for target in ["TBG", "TTR", "HSA"]:
        kd = kd_t4[target]
        theta = (t4_free / kd) / (1.0 + t4_free / kd)
        bound[target] = physiology[f"{target}_M"] * theta
        total_bound += bound[target]
    if total_bound <= 0:
        return THYROID_WEIGHTS.copy()
    return {k: bound[k] / total_bound for k in bound}


def calibrate_kd_tbg_to_target(
    kd_t4: Dict[str, float],
    physiology: Dict[str, float],
    target_free: float,
) -> Dict[str, float]:
    if target_free <= 0:
        raise RuntimeError(f"Invalid target T4 free concentration: {target_free}")

    baseline_cl = {"TBG": 0.0, "TTR": 0.0, "HSA": 0.0}

    def free_for_kd(kd_tbg: float) -> float:
        kd = dict(kd_t4)
        kd["TBG"] = kd_tbg
        return solve_t4_free(baseline_cl, kd, physiology)

    grid = np.logspace(-13, -8, 200)
    vals = np.array([free_for_kd(k) - target_free for k in grid], dtype=float)
    sign = np.sign(vals)
    idx = np.where(sign[:-1] * sign[1:] <= 0)[0]
    if len(idx) == 0:
        raise RuntimeError("Failed to bracket Kd_T4_TBG for target free T4.")

    lo = grid[idx[0]]
    hi = grid[idx[0] + 1]
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        v = free_for_kd(mid) - target_free
        if abs(v) < 1e-18:
            lo = hi = mid
            break
        if (free_for_kd(lo) - target_free) * v <= 0:
            hi = mid
        else:
            lo = mid
    kd_adj = dict(kd_t4)
    kd_adj["TBG"] = float(np.sqrt(lo * hi))
    return kd_adj


def sample_renal_weights(rng: np.random.Generator, renal_targets: List[str]) -> Dict[str, float]:
    base = np.array([RENAL_WEIGHTS[t] for t in renal_targets], dtype=float)
    alpha = np.clip(base * 20.0, 0.1, None)
    draw = rng.dirichlet(alpha)
    return {t: float(draw[i]) for i, t in enumerate(renal_targets)}


def compute_samples(
    sample_ligand_total: Dict[int, Dict[str, float]],
    expo_idx: pd.DataFrame,
    kd_index: pd.DataFrame,
    kd_col_map: Dict[str, str],
    fu: Dict[str, float],
    kd_t4: Dict[str, float],
    thyroid_weights: Dict[str, float],
    renal_weights: Dict[str, float],
    physiology: Dict[str, float],
    with_contrib: bool,
) -> tuple[list[Dict[str, float]], Dict[str, float], Dict[str, float], Dict[str, float], float, float]:
    thyroid_targets = ["TBG", "TTR", "HSA"]
    renal_targets = ["OAT1", "OAT4", "URAT1"]

    baseline_cl = {"TBG": 0.0, "TTR": 0.0, "HSA": 0.0}
    t4_free_baseline = solve_t4_free(baseline_cl, kd_t4, physiology)

    sample_rows: List[Dict[str, float]] = []
    contrib_ttii = defaultdict(float)
    contrib_rtii_equal = defaultdict(float)
    contrib_rtii_weighted = defaultdict(float)
    total_w = 0.0

    for seqn, lig_conc in sample_ligand_total.items():
        if seqn not in expo_idx.index:
            continue
        w = float(expo_idx.loc[seqn, NHANES_WEIGHT_VAR]) if NHANES_WEIGHT_VAR in expo_idx.columns else 1.0
        total_w += w

        free_conc = {lig: conc * fu.get(lig, 1.0) for lig, conc in lig_conc.items()}

        cl = {}
        for target in thyroid_targets + renal_targets:
            acc = 0.0
            kd_col = kd_col_map[target]
            for lig, c_free in free_conc.items():
                if lig not in kd_index.index:
                    continue
                if kd_col not in kd_index.columns:
                    continue
                kd_val = safe_get(kd_index.loc[lig], kd_col, np.nan)
                if np.isnan(kd_val) or kd_val <= 0:
                    continue
                acc += c_free / kd_val
            cl[target] = float(acc)

        cl_thy = {t: cl[t] for t in thyroid_targets}
        t4_free = solve_t4_free(cl_thy, kd_t4, physiology)

        d = {}
        for target in thyroid_targets:
            d[target] = cl[target] / (1.0 + t4_free / kd_t4[target] + cl[target])

        i_renal = {t: cl[t] / (1.0 + cl[t]) for t in renal_targets}
        ttii = (
            thyroid_weights["TBG"] * d["TBG"]
            + thyroid_weights["TTR"] * d["TTR"]
            + thyroid_weights["HSA"] * d["HSA"]
        )
        rtii_equal = float(np.mean([i_renal[t] for t in renal_targets]))
        rtii_weighted = normalized_weighted_mean(i_renal, renal_weights, renal_targets)

        rtii = rtii_weighted
        obi_total = OBI_COMPONENT_WEIGHTS["TTII"] * ttii + OBI_COMPONENT_WEIGHTS["RTII"] * rtii

        sample_rows.append(
            {
                "SEQN": seqn,
                NHANES_WEIGHT_VAR: w,
                "T4_free_baseline_M": t4_free_baseline,
                "T4_free_with_pfas_M": t4_free,
                "T4_free_ratio_change": (t4_free - t4_free_baseline) / max(t4_free_baseline, 1e-30),
                "D_TBG": d["TBG"],
                "D_TTR": d["TTR"],
                "D_HSA": d["HSA"],
                "I_OAT1": i_renal["OAT1"],
                "I_OAT4": i_renal["OAT4"],
                "I_URAT1": i_renal["URAT1"],
                "TTII": ttii,
                "RTII_equal": rtii_equal,
                "RTII_weighted": rtii_weighted,
                "RTII": rtii,
                "OBI_total": obi_total,
            }
        )

        if with_contrib:
            for lig, c_free in free_conc.items():
                if lig not in kd_index.index:
                    continue

                t_contrib = 0.0
                for t in thyroid_targets:
                    kd_col = kd_col_map[t]
                    kd_val = safe_get(kd_index.loc[lig], kd_col, np.nan)
                    if np.isnan(kd_val) or kd_val <= 0:
                        continue
                    theta = (c_free / kd_val) / (1.0 + t4_free / kd_t4[t] + cl[t])
                    t_contrib += thyroid_weights[t] * theta

                r_terms = {}
                for t in renal_targets:
                    kd_col = kd_col_map[t]
                    kd_val = safe_get(kd_index.loc[lig], kd_col, np.nan)
                    if np.isnan(kd_val) or kd_val <= 0:
                        r_terms[t] = 0.0
                    else:
                        r_terms[t] = (c_free / kd_val) / (1.0 + cl[t])

                r_contrib_equal = float(np.mean([r_terms[t] for t in renal_targets]))
                r_contrib_weighted = normalized_weighted_mean(r_terms, renal_weights, renal_targets)

                contrib_ttii[lig] += w * t_contrib
                contrib_rtii_equal[lig] += w * r_contrib_equal
                contrib_rtii_weighted[lig] += w * r_contrib_weighted

    return sample_rows, contrib_ttii, contrib_rtii_equal, contrib_rtii_weighted, total_w, t4_free_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute TTII/RTII/OBI and strict occupancy contributions.")
    parser.add_argument("--state", choices=["anionic", "neutral"], default="anionic")
    parser.add_argument("--cref", type=float, default=1e-7, help="Reference concentration (M) for IBP.")
    parser.add_argument("--out-tag", type=str, default="", help="Optional tag appended to output folder name.")
    parser.add_argument(
        "--hsa-calib",
        choices=["strong", "dsf"],
        default="strong",
        help="Select which HSA calibration column to use (strong or dsf).",
    )
    parser.add_argument(
        "--thyroid-weights",
        choices=["auto", "fixed"],
        default="auto",
        help="Use physiology-derived thyroid binding weights or fixed config weights.",
    )
    parser.add_argument(
        "--calibrate-t4-kd",
        action="store_true",
        help="Calibrate Kd_T4_TBG to match target free T4 at baseline physiology.",
    )
    parser.add_argument(
        "--t4-free-target",
        type=float,
        default=PHYSIOLOGY["T4_free_M"],
        help="Target free T4 concentration (M) used in Kd_T4_TBG calibration.",
    )
    parser.add_argument("--mc-samples", type=int, default=0, help="Monte Carlo samples for uncertainty propagation.")
    parser.add_argument("--mc-seed", type=int, default=2026)
    args = parser.parse_args()

    kd_mat = pd.read_csv(DATA_PROCESSED / "Kd_matrices" / args.state / "kd_target_matrix.csv")
    exposure = pd.read_csv(DATA_PROCESSED / "exposure_models" / "exposure_samples_molar.csv")
    var_map = pd.read_csv(DATA_PROCESSED / "exposure_models" / "nhanes_variable_to_ligand.csv")
    manifest = pd.read_csv(DATA_PROCESSED / "ligands_3d" / "ligand_manifest.csv")

    tag = f"_{args.out_tag}" if args.out_tag else ""
    out_dir = DATA_PROCESSED / "indices_OBI" / f"{args.state}{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    kd_cols = {c for c in kd_mat.columns if c.endswith("_Kd_M")}
    hsa_kd_col = "HSA_Kd_M_dsf" if args.hsa_calib == "dsf" else "HSA_Kd_M"
    if hsa_kd_col not in kd_mat.columns:
        raise RuntimeError(f"{hsa_kd_col} missing in kd matrix.")
    if NHANES_WEIGHT_VAR not in exposure.columns:
        raise RuntimeError(f"{NHANES_WEIGHT_VAR} missing in exposure matrix.")

    # Strict analyte mapping audit: all required NHANES variables must be present and mapped.
    required_vars = set(NHANES_ANALYTE_SPECS.keys())
    map_vars = set(var_map["variable"].astype(str).tolist())
    if required_vars - map_vars:
        raise RuntimeError(f"Missing variables in mapping table: {sorted(required_vars - map_vars)}")
    empty_map = var_map[var_map["ligand_id"].isna() | (var_map["ligand_id"].astype(str).str.len() == 0)]
    if not empty_map.empty:
        raise RuntimeError(f"Empty ligand mapping rows detected: {empty_map[['variable']].to_dict(orient='records')}")

    kd_index = kd_mat.set_index("ligand_id")
    ligands = list(kd_index.index)

    thyroid_targets = ["TBG", "TTR", "HSA"]
    renal_targets = ["OAT1", "OAT4", "URAT1"]

    missing_renal_weights = [k for k in renal_targets if k not in RENAL_WEIGHTS]
    if missing_renal_weights:
        raise RuntimeError(f"Missing renal weights for targets: {missing_renal_weights}")

    # Per-ligand free fraction parameter from HSA.
    hsa_conc = PHYSIOLOGY["HSA_M"]
    hsa_median = float(np.nanmedian(kd_mat[hsa_kd_col]))
    fu: Dict[str, float] = {}
    for lig in ligands:
        kd_hsa = safe_get(kd_index.loc[lig], hsa_kd_col, np.nan)
        if np.isnan(kd_hsa) or kd_hsa <= 0:
            kd_hsa = hsa_median
        fu[lig] = 1.0 / (1.0 + hsa_conc / kd_hsa)

    # Build per-sample total concentration by ligand from NHANES variables.
    var_to_lig = {str(r["variable"]): str(r["ligand_id"]) for _, r in var_map.iterrows()}
    sample_ligand_total: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for _, row in exposure.iterrows():
        seqn = int(row["SEQN"])
        for var, lig in var_to_lig.items():
            col = f"{var}_M"
            if col not in exposure.columns:
                continue
            val = row[col]
            if pd.isna(val):
                continue
            sample_ligand_total[seqn][lig] += float(val)

    kd_t4 = {
        "HSA": PHYSIOLOGY["Kd_T4_HSA_M"],
        "TTR": PHYSIOLOGY["Kd_T4_TTR_M"],
        "TBG": PHYSIOLOGY["Kd_T4_TBG_M"],
    }
    if args.calibrate_t4_kd:
        kd_t4 = calibrate_kd_tbg_to_target(kd_t4, PHYSIOLOGY, args.t4_free_target)

    kd_col_map = {t: f"{t}_Kd_M" for t in thyroid_targets + renal_targets}
    kd_col_map["HSA"] = hsa_kd_col

    if args.thyroid_weights == "auto":
        thyroid_weights = derive_thyroid_weights(kd_t4, PHYSIOLOGY)
    else:
        thyroid_weights = THYROID_WEIGHTS.copy()

    expo_idx = exposure.set_index("SEQN")
    (
        sample_rows,
        contrib_ttii,
        contrib_rtii_equal,
        contrib_rtii_weighted,
        total_w,
        t4_free_baseline,
    ) = compute_samples(
        sample_ligand_total,
        expo_idx,
        kd_index,
        kd_col_map,
        fu,
        kd_t4,
        thyroid_weights,
        RENAL_WEIGHTS,
        PHYSIOLOGY,
        with_contrib=True,
    )

    sample_df = pd.DataFrame(sample_rows).sort_values("SEQN")
    sample_df.to_csv(out_dir / "sample_indices.csv", index=False)

    # Weighted population statistics.
    stat_rows: List[Dict[str, float]] = []
    for col in ["TTII", "RTII", "RTII_equal", "RTII_weighted", "OBI_total", "T4_free_ratio_change"]:
        vals = sample_df[col].to_numpy(dtype=float)
        ws = sample_df[NHANES_WEIGHT_VAR].to_numpy(dtype=float)
        mu = float(np.average(vals, weights=ws))
        p95 = weighted_quantile(vals, ws, 0.95)
        p99 = weighted_quantile(vals, ws, 0.99)
        stat_rows.append({"metric": col, "weighted_mean": mu, "p95": p95, "p99": p99})
    pd.DataFrame(stat_rows).to_csv(out_dir / "population_index_summary.csv", index=False)

    contrib_rows = []
    for lig in sorted(set(contrib_ttii.keys()) | set(contrib_rtii_weighted.keys()) | set(contrib_rtii_equal.keys())):
        t = contrib_ttii.get(lig, 0.0) / max(total_w, 1e-30)
        r_eq = contrib_rtii_equal.get(lig, 0.0) / max(total_w, 1e-30)
        r_wt = contrib_rtii_weighted.get(lig, 0.0) / max(total_w, 1e-30)
        r = r_wt
        contrib_rows.append(
            {
                "ligand_id": lig,
                "TTII_contribution": t,
                "RTII_contribution": r,
                "RTII_contribution_equal": r_eq,
                "RTII_contribution_weighted": r_wt,
                "mean_contribution": 0.5 * (t + r),
            }
        )
    contrib_df = pd.DataFrame(contrib_rows).sort_values("mean_contribution", ascending=False)
    contrib_df = contrib_df.merge(manifest[["ligand_id", "DTXSID", "preferred_name"]], on="ligand_id", how="left")
    contrib_df.to_csv(out_dir / "ligand_contributions.csv", index=False)

    top_n = max(1, int(np.ceil(len(contrib_df) * 0.01)))
    contrib_df.head(top_n).to_csv(out_dir / "top1pct_drivers.csv", index=False)

    weight_payload = {
        "mode": args.thyroid_weights,
        "weights": thyroid_weights,
        "baseline_t4_free_M": t4_free_baseline,
        "hsa_calib": args.hsa_calib,
        "hsa_kd_column": hsa_kd_col,
        "calibrate_t4_kd": bool(args.calibrate_t4_kd),
        "t4_free_target_M": float(args.t4_free_target),
        "Kd_T4_TBG_M": float(kd_t4["TBG"]),
    }
    (out_dir / "thyroid_weights_used.json").write_text(json.dumps(weight_payload, indent=2), encoding="utf-8")

    # Intrinsic Burden Potential (IBP) at fixed concentration.
    ibp_rows = []
    for _, row in kd_mat.iterrows():
        lig = row["ligand_id"]
        c_total = args.cref
        c_free = c_total * fu.get(lig, 1.0)

        cl_thy = {}
        d = {}
        for t in thyroid_targets:
            kd_val = safe_get(row, kd_col_map[t], np.nan)
            cl_val = c_free / kd_val if kd_val > 0 and not np.isnan(kd_val) else 0.0
            cl_thy[t] = cl_val

        t4_free = solve_t4_free(cl_thy, kd_t4)

        for t in thyroid_targets:
            d[t] = cl_thy[t] / (1.0 + t4_free / kd_t4[t] + cl_thy[t])
        ttii_ref = thyroid_weights["TBG"] * d["TBG"] + thyroid_weights["TTR"] * d["TTR"] + thyroid_weights["HSA"] * d["HSA"]

        i_renal = {}
        for t in renal_targets:
            kd_val = safe_get(row, kd_col_map[t], np.nan)
            cl_val = c_free / kd_val if kd_val > 0 and not np.isnan(kd_val) else 0.0
            i_renal[t] = cl_val / (1.0 + cl_val)

        rtii_ref_equal = float(np.mean([i_renal[t] for t in renal_targets]))
        rtii_ref_weighted = normalized_weighted_mean(i_renal, RENAL_WEIGHTS, renal_targets)
        rtii_ref = rtii_ref_weighted

        obi_ref = OBI_COMPONENT_WEIGHTS["TTII"] * ttii_ref + OBI_COMPONENT_WEIGHTS["RTII"] * rtii_ref

        ibp_rows.append(
            {
                "ligand_id": lig,
                "C_ref_M": args.cref,
                "fu": fu.get(lig, 1.0),
                "T4_free_baseline_M": t4_free_baseline,
                "T4_free_with_pfas_M": t4_free,
                "T4_free_ratio_change": (t4_free - t4_free_baseline) / max(t4_free_baseline, 1e-30),
                "TTII_ref": ttii_ref,
                "RTII_ref_equal": rtii_ref_equal,
                "RTII_ref_weighted": rtii_ref_weighted,
                "RTII_ref": rtii_ref,
                "OBI_ref": obi_ref,
            }
        )

    ibp_df = pd.DataFrame(ibp_rows)
    ibp_df = ibp_df.merge(manifest[["ligand_id", "DTXSID", "preferred_name", "smiles_anionic"]], on="ligand_id", how="left")
    ibp_df.to_csv(out_dir / "intrinsic_burden_potential.csv", index=False)

    # Monte Carlo uncertainty propagation for population indices.
    if args.mc_samples > 0:
        rng = np.random.default_rng(args.mc_seed)
        calib_json = DATA_PROCESSED / "Kd_matrices" / args.state / "calibration_model.json"
        sigma = None
        sigma_hsa = None
        if calib_json.exists():
            meta = json.loads(calib_json.read_text(encoding="utf-8"))
            sigma = float(meta.get("sigma_log10_residual", 0.3))
            sigma_hsa = float(meta.get("sigma_log10_residual_hsa_dsf", sigma)) if args.hsa_calib == "dsf" else sigma
        sigma = sigma if sigma is not None else 0.3
        sigma_hsa = sigma_hsa if sigma_hsa is not None else sigma

        targets = thyroid_targets + renal_targets
        log10_base = np.zeros((len(ligands), len(targets)), dtype=float)
        for j, t in enumerate(targets):
            col = kd_col_map[t]
            log10_base[:, j] = np.log10(np.clip(kd_index[col].to_numpy(dtype=float), 1e-30, None))

        mc_stats: Dict[str, Dict[str, List[float]]] = {
            m: {"weighted_mean": [], "p95": [], "p99": []}
            for m in ["TTII", "RTII", "RTII_equal", "RTII_weighted", "OBI_total", "T4_free_ratio_change"]
        }

        for _ in range(args.mc_samples):
            phys = dict(PHYSIOLOGY)
            for k, (lo, hi) in PHYSIOLOGY_RANGES.items():
                phys[k] = float(rng.uniform(lo, hi))

            hsa_conc_mc = phys["HSA_M"]
            renal_weights_mc = sample_renal_weights(rng, renal_targets)
            if args.thyroid_weights == "auto":
                thyroid_weights_mc = derive_thyroid_weights(kd_t4, phys)
            else:
                thyroid_weights_mc = THYROID_WEIGHTS.copy()

            noise = rng.normal(0.0, sigma, size=log10_base.shape)
            hsa_idx = targets.index("HSA")
            noise[:, hsa_idx] = rng.normal(0.0, sigma_hsa, size=log10_base.shape[0])
            log10_kd = log10_base + noise
            kd_vals = np.power(10.0, log10_kd)

            kd_cols = {kd_col_map[t]: kd_vals[:, j] for j, t in enumerate(targets)}
            kd_index_mc = pd.DataFrame(kd_cols, index=ligands)

            hsa_median_mc = float(np.nanmedian(kd_index_mc[hsa_kd_col]))
            fu_mc: Dict[str, float] = {}
            for lig in ligands:
                kd_hsa = safe_get(kd_index_mc.loc[lig], hsa_kd_col, np.nan)
                if np.isnan(kd_hsa) or kd_hsa <= 0:
                    kd_hsa = hsa_median_mc
                fu_mc[lig] = 1.0 / (1.0 + hsa_conc_mc / kd_hsa)

            sample_rows_mc, _, _, _, _, _ = compute_samples(
                sample_ligand_total,
                expo_idx,
                kd_index_mc,
                kd_col_map,
                fu_mc,
                kd_t4,
                thyroid_weights_mc,
                renal_weights_mc,
                phys,
                with_contrib=False,
            )
            df_mc = pd.DataFrame(sample_rows_mc)
            for metric in mc_stats.keys():
                vals = df_mc[metric].to_numpy(dtype=float)
                ws = df_mc[NHANES_WEIGHT_VAR].to_numpy(dtype=float)
                mc_stats[metric]["weighted_mean"].append(float(np.average(vals, weights=ws)))
                mc_stats[metric]["p95"].append(weighted_quantile(vals, ws, 0.95))
                mc_stats[metric]["p99"].append(weighted_quantile(vals, ws, 0.99))

        rows = []
        for metric, stat_map in mc_stats.items():
            for stat, arr in stat_map.items():
                vec = np.asarray(arr, dtype=float)
                rows.append(
                    {
                        "metric": metric,
                        "stat": stat,
                        "mc_mean": float(np.mean(vec)),
                        "mc_median": float(np.quantile(vec, 0.50)),
                        "mc_p05": float(np.quantile(vec, 0.05)),
                        "mc_p95": float(np.quantile(vec, 0.95)),
                        "n_mc": int(args.mc_samples),
                        "hsa_calib": args.hsa_calib,
                        "thyroid_weights": args.thyroid_weights,
                    }
                )
        pd.DataFrame(rows).to_csv(out_dir / "population_index_summary_mc.csv", index=False)

    print(f"Sample OBI rows: {len(sample_df)}")
    print(f"Top drivers file: {out_dir / 'top1pct_drivers.csv'}")
    print(f"IBP file: {out_dir / 'intrinsic_burden_potential.csv'}")


if __name__ == "__main__":
    main()
