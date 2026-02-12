#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import shap
except Exception:
    shap = None

from config import DATA_PROCESSED, MODELS


LOG_EPS = 1e-12


def featurize_smiles(smiles: str, n_bits: int = 512) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits + 8, dtype=float)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=float)
    from rdkit.DataStructs import ConvertToNumpyArray

    ConvertToNumpyArray(fp, arr)
    phys = np.array(
        [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.RingCount(mol),
            Descriptors.TPSA(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.HeavyAtomCount(mol),
        ],
        dtype=float,
    )
    return np.concatenate([arr, phys], axis=0)


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def to_log10(y: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(y + LOG_EPS, 1e-30, None))


def from_log10(y_log: np.ndarray) -> np.ndarray:
    return np.power(10.0, y_log) - LOG_EPS


def murcko(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def scaffold_split_indices(smiles: List[str], test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaffolds = [murcko(s) for s in smiles]
    groups: Dict[str, List[int]] = {}
    for i, scf in enumerate(scaffolds):
        key = scf if scf else f"NO_SCF_{i}"
        groups.setdefault(key, []).append(i)

    rng = np.random.default_rng(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: len(kv[1]), reverse=True)

    n_total = len(smiles)
    target_test_n = max(1, int(round(n_total * test_size)))
    test_idx: List[int] = []
    for _, idxs in items:
        if len(test_idx) < target_test_n:
            test_idx.extend(idxs)
    test_idx = sorted(set(test_idx))
    train_idx = sorted(set(range(n_total)) - set(test_idx))

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError("Scaffold split failed to create non-empty train/test sets.")

    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int), np.asarray(scaffolds, dtype=object)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QSAR models for intrinsic burden potential.")
    parser.add_argument("--state", choices=["anionic", "neutral"], default="anionic")
    parser.add_argument("--endpoint", choices=["OBI_ref", "TTII_ref", "RTII_ref"], default="OBI_ref")
    parser.add_argument("--split", choices=["scaffold", "random"], default="scaffold")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeated splits with different seeds.")
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed.")
    args = parser.parse_args()

    ibp = pd.read_csv(DATA_PROCESSED / "indices_OBI" / args.state / "intrinsic_burden_potential.csv")
    ibp = ibp.dropna(subset=["smiles_anionic", args.endpoint]).copy().reset_index(drop=True)

    X = np.vstack([featurize_smiles(smi) for smi in ibp["smiles_anionic"]])
    y_raw = ibp[args.endpoint].to_numpy(dtype=float)
    y = to_log10(y_raw)

    out_dir = MODELS / "qsar" / args.state
    out_dir.mkdir(parents=True, exist_ok=True)

    repeats = max(1, int(args.repeats))
    metrics_all: List[Dict[str, object]] = []
    shap_generated = False
    feat_names = [f"ECFP4_{i}" for i in range(512)] + [
        "MolWt",
        "LogP",
        "HBD",
        "HBA",
        "RingCount",
        "TPSA",
        "FractionCSP3",
        "HeavyAtomCount",
    ]

    for rep in range(repeats):
        seed = args.seed + rep
        if args.split == "scaffold":
            idx_train, idx_test, scaffolds = scaffold_split_indices(
                ibp["smiles_anionic"].tolist(), test_size=0.2, seed=seed
            )
        else:
            idx_train, idx_test = train_test_split(np.arange(len(ibp)), test_size=0.2, random_state=seed)
            idx_train = np.asarray(sorted(idx_train), dtype=int)
            idx_test = np.asarray(sorted(idx_test), dtype=int)
            scaffolds = np.asarray([murcko(s) for s in ibp["smiles_anionic"].tolist()], dtype=object)

        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        groups_train = scaffolds[idx_train]
        unique_groups = np.unique(groups_train)
        if len(unique_groups) >= 5:
            cv = GroupKFold(n_splits=5)
            cv_kwargs = {"groups": groups_train}
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=seed)
            cv_kwargs = {}

        elastic = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.8, 1.0], cv=5, random_state=seed, max_iter=20000)),
            ]
        )
        elastic.fit(X_train, y_train)
        pred_test_elastic = elastic.predict(X_test)
        cv_pred_elastic = cross_val_predict(elastic, X_train, y_train, cv=cv, n_jobs=-1, **cv_kwargs)

        rf = RandomForestRegressor(
            n_estimators=600,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=2,
        )
        rf.fit(X_train, y_train)
        pred_test_rf = rf.predict(X_test)
        cv_pred_rf = cross_val_predict(rf, X_train, y_train, cv=cv, n_jobs=-1, **cv_kwargs)

        for model_name, split_name, yt, yp in [
            ("ElasticNetCV", "cv_train", y_train, cv_pred_elastic),
            ("ElasticNetCV", "test", y_test, pred_test_elastic),
            ("RandomForest", "cv_train", y_train, cv_pred_rf),
            ("RandomForest", "test", y_test, pred_test_rf),
        ]:
            m_log = eval_metrics(yt, yp)
            yt_lin = from_log10(yt)
            yp_lin = from_log10(yp)
            m_lin = eval_metrics(yt_lin, yp_lin)
            metrics_all.append(
                {
                    "repeat": rep,
                    "seed": seed,
                    "model": model_name,
                    "split": split_name,
                    "r2_log": m_log["r2"],
                    "rmse_log": m_log["rmse"],
                    "mae_log": m_log["mae"],
                    "r2_linear": m_lin["r2"],
                    "rmse_linear": m_lin["rmse"],
                    "mae_linear": m_lin["mae"],
                }
            )

        if rep == 0:
            imp = pd.DataFrame({"feature": feat_names, "importance": rf.feature_importances_}).sort_values(
                "importance", ascending=False
            )
            imp.head(120).to_csv(out_dir / f"rf_feature_importance_{args.endpoint}.csv", index=False)

            if shap is not None:
                try:
                    explainer = shap.TreeExplainer(rf)
                    shap_values = explainer.shap_values(X_test)
                    arr = np.asarray(shap_values)
                    if arr.ndim == 3:
                        arr = arr[0]
                    mean_abs = np.mean(np.abs(arr), axis=0)
                    shap_df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs}).sort_values(
                        "mean_abs_shap", ascending=False
                    )
                    shap_df.head(120).to_csv(out_dir / f"rf_shap_importance_{args.endpoint}.csv", index=False)
                    shap_generated = True
                except Exception as exc:
                    (out_dir / f"rf_shap_error_{args.endpoint}.txt").write_text(str(exc), encoding="utf-8")

            pred_df = ibp.loc[idx_test, ["ligand_id", "DTXSID", "preferred_name", args.endpoint]].copy()
            pred_df["scaffold"] = scaffolds[idx_test]
            pred_df["repeat"] = rep
            pred_df[f"true_log10_{args.endpoint}"] = y_test
            pred_df["pred_elastic_log10"] = pred_test_elastic
            pred_df["pred_rf_log10"] = pred_test_rf
            pred_df["pred_elastic"] = from_log10(pred_test_elastic)
            pred_df["pred_rf"] = from_log10(pred_test_rf)
            pred_df.to_csv(out_dir / f"test_predictions_{args.endpoint}.csv", index=False)

    metrics_df = pd.DataFrame(metrics_all)
    metrics_df.to_csv(out_dir / f"metrics_{args.endpoint}.csv", index=False)
    metrics_df.to_csv(out_dir / f"metrics_{args.endpoint}_repeats.csv", index=False)

    summary_rows = (
        metrics_df.groupby(["model", "split"])
        .agg(
            r2_log_mean=("r2_log", "mean"),
            r2_log_std=("r2_log", "std"),
            rmse_log_mean=("rmse_log", "mean"),
            rmse_log_std=("rmse_log", "std"),
            mae_log_mean=("mae_log", "mean"),
            mae_log_std=("mae_log", "std"),
            r2_linear_mean=("r2_linear", "mean"),
            r2_linear_std=("r2_linear", "std"),
            rmse_linear_mean=("rmse_linear", "mean"),
            rmse_linear_std=("rmse_linear", "std"),
            mae_linear_mean=("mae_linear", "mean"),
            mae_linear_std=("mae_linear", "std"),
        )
        .reset_index()
    )
    summary_rows.to_csv(out_dir / f"metrics_{args.endpoint}_summary.csv", index=False)

    best_model = (
        summary_rows[summary_rows["split"] == "test"]
        .sort_values("r2_log_mean", ascending=False)
        .iloc[0]["model"]
        if not summary_rows.empty
        else "NA"
    )

    summary = {
        "endpoint": args.endpoint,
        "transform": f"log10(y + {LOG_EPS})",
        "split_strategy": args.split,
        "n_samples": int(len(ibp)),
        "n_repeats": repeats,
        "base_seed": int(args.seed),
        "best_model_by_mean_test_r2_log": best_model,
        "shap_generated": bool(shap_generated),
    }
    (out_dir / f"summary_{args.endpoint}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"QSAR samples: {len(ibp)}")
    print(f"Split strategy: {args.split}")
    print(f"Repeats: {repeats}")
    print(f"SHAP generated: {shap_generated}")
    print(f"Metrics: {out_dir / f'metrics_{args.endpoint}.csv'}")


if __name__ == "__main__":
    main()
