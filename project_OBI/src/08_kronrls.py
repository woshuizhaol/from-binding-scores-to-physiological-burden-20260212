#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from Bio import pairwise2
from Bio.Align import substitution_matrices
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rlscore.learner import KronRLS
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

from config import DATA_PROCESSED, MODELS, PDB_DIR, TARGETS


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def extract_pocket_sequence(pdb_path: Path, ligand_resname: str, cutoff: float = 5.0) -> tuple[str, dict]:
    ligand_atoms = []
    residues = {}
    chain_order: Dict[str, List[tuple[str, str]]] = {}
    chain_seen = set()
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            rec = line[:6].strip()
            if rec not in {"ATOM", "HETATM"}:
                continue
            resname = line[17:20].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            chain = line[21].strip() or "_"
            resi = line[22:26].strip()
            icode = line[26].strip()
            res_id = f"{resi}{icode}" if icode else resi
            key = (chain, res_id, resname)
            if rec == "HETATM" and resname == ligand_resname:
                ligand_atoms.append(np.array([x, y, z], dtype=float))
            elif rec == "ATOM":
                residues.setdefault(key, []).append(np.array([x, y, z], dtype=float))
                if (chain, res_id) not in chain_seen:
                    chain_order.setdefault(chain, []).append((res_id, resname))
                    chain_seen.add((chain, res_id))
    if not ligand_atoms:
        return "", {"mode": "missing_ligand"}
    ligand_atoms = np.asarray(ligand_atoms)
    pocket_keys = []
    for key, atoms in residues.items():
        arr = np.asarray(atoms)
        d2 = np.sum((arr[:, None, :] - ligand_atoms[None, :, :]) ** 2, axis=2)
        if np.sqrt(d2.min()) <= cutoff:
            pocket_keys.append(key)
    if not pocket_keys:
        return "X", {"mode": "no_pocket"}

    # Choose chain with most pocket residues.
    pocket_by_chain: Dict[str, List[tuple[str, str, str]]] = {}
    for key in pocket_keys:
        pocket_by_chain.setdefault(key[0], []).append(key)
    chain_sel = max(pocket_by_chain.keys(), key=lambda c: len(pocket_by_chain[c]))

    # Build continuous segment covering all pocket residues in that chain.
    chain_residues = chain_order.get(chain_sel, [])
    pocket_set = {(k[0], k[1]) for k in pocket_by_chain[chain_sel]}
    idxs = [i for i, (res_id, _) in enumerate(chain_residues) if (chain_sel, res_id) in pocket_set]

    if not idxs:
        # Fallback to concatenated pocket residues.
        pocket_keys = sorted(pocket_keys, key=lambda k: (k[0], k[1]))
        seq = "".join(AA3_TO_1.get(k[2], "X") for k in pocket_keys)
        return seq if seq else "X", {"mode": "concat_fallback"}

    i0, i1 = min(idxs), max(idxs)
    seg = chain_residues[i0 : i1 + 1]
    seq = "".join(AA3_TO_1.get(resname, "X") for _, resname in seg)
    meta = {
        "mode": "continuous_segment",
        "chain": chain_sel,
        "segment_start": seg[0][0] if seg else "",
        "segment_end": seg[-1][0] if seg else "",
        "pocket_residue_count": len(idxs),
        "segment_len": len(seq),
    }
    return (seq if seq else "X"), meta


def protein_kernel_from_sequences(seq_map: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
    targets = sorted(seq_map.keys())
    n = len(targets)
    k = np.zeros((n, n), dtype=float)
    blosum62 = substitution_matrices.load("BLOSUM62")
    self_scores = {}
    for t in targets:
        aln = pairwise2.align.localds(seq_map[t], seq_map[t], blosum62, -10.0, -0.5, one_alignment_only=True, score_only=True)
        self_scores[t] = max(aln, 1e-6)
    for i, ti in enumerate(targets):
        for j, tj in enumerate(targets):
            if i > j:
                continue
            score = pairwise2.align.localds(
                seq_map[ti], seq_map[tj], blosum62, -10.0, -0.5, one_alignment_only=True, score_only=True
            )
            sim = score / np.sqrt(self_scores[ti] * self_scores[tj])
            k[i, j] = k[j, i] = sim
    return k, targets


def chemical_kernel(smiles: List[str]) -> np.ndarray:
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    n = len(fps)
    k = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            if fps[i] is None or fps[j] is None:
                sim = 0.0
            else:
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            k[i, j] = k[j, i] = sim
    return k


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = 0.0
    concordant = 0.0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] == y_true[j]:
                continue
            n += 1.0
            diff_true = y_true[i] - y_true[j]
            diff_pred = y_pred[i] - y_pred[j]
            prod = diff_true * diff_pred
            if prod > 0:
                concordant += 1.0
            elif prod == 0:
                concordant += 0.5
    return float(concordant / n) if n > 0 else float("nan")


def run_loto_rlscore(y_df: pd.DataFrame, kc: np.ndarray, kp: np.ndarray, target_order: List[str], reg: float):
    eval_rows = []
    pred_tables = []

    y_all = y_df.to_numpy(dtype=float)
    n_lig = y_all.shape[0]

    for hold_idx, hold_target in enumerate(target_order):
        train_cols = [i for i in range(len(target_order)) if i != hold_idx]

        # RLScore KronRLS requires complete Y on training Kronecker grid.
        train_lig_mask = ~np.isnan(y_all[:, train_cols]).any(axis=1)
        train_lig_idx = np.where(train_lig_mask)[0]
        if len(train_lig_idx) < 10:
            raise RuntimeError(f"Too few complete training ligands for holdout {hold_target}: {len(train_lig_idx)}")

        y_train_mat = y_all[np.ix_(train_lig_idx, train_cols)]
        y_train_vec = y_train_mat.reshape(-1, order="F")

        k1_train = kc[np.ix_(train_lig_idx, train_lig_idx)]
        k2_train = kp[np.ix_(train_cols, train_cols)]

        model = KronRLS(K1=k1_train, K2=k2_train, Y=y_train_vec, regparam=reg)

        # Predict holdout target for all ligands against the trained target space.
        k1_pred = kc[:, train_lig_idx]
        k2_pred = kp[np.ix_([hold_idx], train_cols)]
        pred_all = np.asarray(model.predict(k1_pred, k2_pred)).ravel(order="F")

        truth_all = y_all[:, hold_idx]
        eval_mask = ~np.isnan(truth_all)
        n_eval = int(eval_mask.sum())

        if n_eval < 3:
            spr = np.nan
            rmse = np.nan
            r2 = np.nan
            cidx = np.nan
        else:
            spr, _ = spearmanr(truth_all[eval_mask], pred_all[eval_mask])
            rmse = float(np.sqrt(mean_squared_error(truth_all[eval_mask], pred_all[eval_mask])))
            r2 = float(r2_score(truth_all[eval_mask], pred_all[eval_mask]))
            cidx = concordance_index(truth_all[eval_mask], pred_all[eval_mask])

        eval_rows.append(
            {
                "holdout_target": hold_target,
                "n_train_ligands": int(len(train_lig_idx)),
                "n_eval": n_eval,
                "spearman": spr,
                "cindex": cidx,
                "rmse": rmse,
                "r2": r2,
            }
        )

        tab = pd.DataFrame(
            {
                "holdout_target": hold_target,
                "true_pKd": truth_all,
                "pred_pKd": pred_all,
            }
        )
        pred_tables.append(tab)

    return pd.DataFrame(eval_rows), pd.concat(pred_tables, axis=0, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official RLScore KronRLS target extrapolation.")
    parser.add_argument("--state", choices=["anionic", "neutral"], default="anionic")
    parser.add_argument("--reg", type=float, default=1e-2)
    parser.add_argument(
        "--reg-grid",
        type=str,
        default="",
        help="Comma-separated list of regparams to search using mean c-index.",
    )
    args = parser.parse_args()

    kd = pd.read_csv(DATA_PROCESSED / "Kd_matrices" / args.state / "kd_target_matrix.csv")
    manifest = pd.read_csv(DATA_PROCESSED / "ligands_3d" / "ligand_manifest.csv")[["ligand_id", "smiles_anionic", "DTXSID", "preferred_name"]]
    data = kd.merge(manifest, on="ligand_id", how="left")
    data = data.dropna(subset=["smiles_anionic"]).copy()

    target_order = [t for t in TARGETS.keys() if f"{t}_pKd" in data.columns]
    y_df = data[[f"{t}_pKd" for t in target_order]].copy()

    kc = chemical_kernel(data["smiles_anionic"].tolist())

    seq_map = {}
    seq_rows = []
    for t in target_order:
        pdb_id = TARGETS[t].pdb_id
        ligand = TARGETS[t].ligand_resname if TARGETS[t].ligand_resname else "MYR"
        seq, meta = extract_pocket_sequence(PDB_DIR / f"{pdb_id}.pdb", ligand_resname=ligand)
        seq_map[t] = seq if seq else "X"
        seq_rows.append(
            {
                "target": t,
                "pdb_id": pdb_id,
                "ligand_resname": ligand,
                "seq_len": int(len(seq_map[t])),
                "sequence": seq_map[t],
                "mode": meta.get("mode", ""),
                "chain": meta.get("chain", ""),
                "segment_start": meta.get("segment_start", ""),
                "segment_end": meta.get("segment_end", ""),
                "pocket_residue_count": meta.get("pocket_residue_count", ""),
                "segment_len": meta.get("segment_len", ""),
            }
        )
    kp, kp_targets = protein_kernel_from_sequences(seq_map)

    idx_map = [kp_targets.index(t) for t in target_order]
    kp = kp[np.ix_(idx_map, idx_map)]

    out_dir = MODELS / "kronrls" / args.state
    out_dir.mkdir(parents=True, exist_ok=True)

    reg_grid = [x for x in (args.reg_grid.split(",") if args.reg_grid else []) if x]
    best_reg = args.reg
    reg_selection_rows = []
    best_eval = None
    best_pred = None
    if reg_grid:
        for reg_str in reg_grid:
            reg = float(reg_str)
            eval_df_tmp, pred_df_tmp = run_loto_rlscore(y_df, kc, kp, target_order, reg=reg)
            cidx_mean = float(eval_df_tmp["cindex"].dropna().mean())
            spr_mean = float(eval_df_tmp["spearman"].dropna().mean())
            reg_selection_rows.append(
                {
                    "reg": reg,
                    "mean_cindex": cidx_mean,
                    "mean_spearman": spr_mean,
                }
            )
            if best_eval is None or cidx_mean > float(best_eval["cindex"].dropna().mean()):
                best_reg = reg
                best_eval = eval_df_tmp
                best_pred = pred_df_tmp
        pd.DataFrame(reg_selection_rows).sort_values("mean_cindex", ascending=False).to_csv(
            out_dir / "reg_selection.csv", index=False
        )

    if best_eval is None:
        eval_df, pred_df = run_loto_rlscore(y_df, kc, kp, target_order, reg=best_reg)
    else:
        eval_df, pred_df = best_eval, best_pred
    eval_df = eval_df.sort_values("spearman", ascending=False)
    eval_df.to_csv(out_dir / "loto_metrics.csv", index=False)
    pd.DataFrame(seq_rows).to_csv(out_dir / "pocket_sequences.csv", index=False)

    pred_df = pd.concat([data[["ligand_id", "DTXSID", "preferred_name"]].loc[pred_df.index % len(data)].reset_index(drop=True), pred_df], axis=1)
    pred_df.to_csv(out_dir / "loto_predictions.csv", index=False)

    for showcase in ["GPR40", "URAT1"]:
        if showcase not in target_order:
            continue
        show = pred_df[pred_df["holdout_target"] == showcase].copy()
        show = show.sort_values("pred_pKd", ascending=False)
        show.head(30).to_csv(out_dir / f"showcase_top30_{showcase}.csv", index=False)

    (out_dir / "kernel_shapes.json").write_text(
        json.dumps(
            {
                "n_ligands": int(kc.shape[0]),
                "n_targets": int(kp.shape[0]),
                "reg": float(best_reg),
                "reg_grid": reg_grid,
                "implementation": "RLScore KronRLS",
                "missing_policy": "no target-median imputation; train on complete observed blocks and evaluate on observed labels only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Kron-RLS LOTO metrics: {out_dir / 'loto_metrics.csv'}")
    print(f"Predictions: {out_dir / 'loto_predictions.csv'}")


if __name__ == "__main__":
    main()
