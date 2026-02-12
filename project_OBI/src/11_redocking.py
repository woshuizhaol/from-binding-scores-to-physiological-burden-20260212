#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openbabel import openbabel as ob

from config import DATA_PROCESSED, PDB_DIR, RESULTS, TARGETS


@dataclass
class RedockTarget:
    target: str
    pdb_id: str
    ligand_resname: str
    receptor_pdbqt: Path
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float


def extract_ligand_from_pdb(
    pdb_path: Path, ligand_resname: str, center: Optional[Tuple[float, float, float]] = None
) -> Tuple[ob.OBMol, str]:
    conv = ob.OBConversion()
    conv.SetInFormat("pdb")
    mol = ob.OBMol()
    if not conv.ReadFile(mol, str(pdb_path)):
        raise RuntimeError(f"Failed to read PDB: {pdb_path}")

    residue_atoms: Dict[Tuple[str, int, str], List[int]] = {}
    residue_xyz: Dict[Tuple[str, int, str], List[np.ndarray]] = {}

    for atom in ob.OBMolAtomIter(mol):
        residue = atom.GetResidue()
        if not residue:
            continue
        if residue.GetName().strip() != ligand_resname:
            continue
        key = (str(residue.GetChain()).strip() or "_", int(residue.GetNum()), residue.GetName().strip())
        residue_atoms.setdefault(key, []).append(atom.GetIdx())
        residue_xyz.setdefault(key, []).append(np.array([atom.GetX(), atom.GetY(), atom.GetZ()], dtype=float))

    if not residue_atoms:
        raise RuntimeError(f"Ligand {ligand_resname} not found in {pdb_path.name}")

    keys = list(residue_atoms.keys())
    if len(keys) == 1:
        chosen_key = keys[0]
    else:
        if center is None:
            chosen_key = sorted(keys, key=lambda k: len(residue_atoms[k]), reverse=True)[0]
        else:
            c = np.array(center, dtype=float)
            chosen_key = min(keys, key=lambda k: float(np.linalg.norm(np.mean(residue_xyz[k], axis=0) - c)))

    selected = residue_atoms[chosen_key]

    lig = ob.OBMol()
    idx_map: Dict[int, int] = {}
    for old_idx in selected:
        old_atom = mol.GetAtom(old_idx)
        new_atom = lig.NewAtom()
        new_atom.SetAtomicNum(old_atom.GetAtomicNum())
        new_atom.SetFormalCharge(old_atom.GetFormalCharge())
        new_atom.SetVector(old_atom.GetX(), old_atom.GetY(), old_atom.GetZ())
        idx_map[old_idx] = new_atom.GetIdx()

    for bond in ob.OBMolBondIter(mol):
        b = bond.GetBeginAtomIdx()
        e = bond.GetEndAtomIdx()
        if b in idx_map and e in idx_map:
            lig.AddBond(idx_map[b], idx_map[e], bond.GetBondOrder())

    if lig.NumBonds() == 0:
        lig.ConnectTheDots()
        lig.PerceiveBondOrders()

    residue_tag = f"{chosen_key[0]}_{chosen_key[1]}_{chosen_key[2]}"
    lig.SetTitle(f"{pdb_path.stem}_{residue_tag}")
    return lig, residue_tag


def write_pdbqt(mol: ob.OBMol, out_path: Path) -> None:
    if out_path.exists():
        out_path.unlink()

    conv = ob.OBConversion()
    conv.SetOutFormat("pdbqt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not conv.WriteFile(mol, str(out_path)):
        raise RuntimeError(f"Failed to write PDBQT: {out_path}")


def read_all_poses_pdbqt(path: Path) -> List[ob.OBMol]:
    conv = ob.OBConversion()
    conv.SetInFormat("pdbqt")
    mol = ob.OBMol()
    if not conv.ReadFile(mol, str(path)):
        raise RuntimeError(f"Failed to read PDBQT: {path}")
    mols = [ob.OBMol(mol)]
    while True:
        nxt = ob.OBMol()
        if not conv.Read(nxt):
            break
        mols.append(ob.OBMol(nxt))
    return mols


def heavy_atom_rmsd(ref_mol: ob.OBMol, probe_mol: ob.OBMol) -> float:
    ref = ob.OBMol(ref_mol)
    probe = ob.OBMol(probe_mol)
    ref.DeleteHydrogens()
    probe.DeleteHydrogens()

    align = ob.OBAlign(ref, probe, False, True)
    ok = align.Align()
    if not ok:
        raise RuntimeError("OBAlign failed")
    return float(align.GetRMSD())


def parse_best_affinity(vina_output: str) -> Optional[float]:
    for line in vina_output.splitlines():
        m = re.match(r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+", line)
        if m:
            return float(m.group(1))
    return None


def load_redock_targets(target_subset: Optional[List[str]]) -> List[RedockTarget]:
    boxes = pd.read_csv(DATA_PROCESSED / "receptors_prepared" / "boxes" / "docking_boxes.tsv", sep="\t")
    targets = []
    for target_name, cfg in TARGETS.items():
        if cfg.ligand_resname is None:
            continue
        if target_subset and target_name not in target_subset:
            continue

        rows = boxes[boxes["target"] == target_name].copy()
        if rows.empty:
            raise RuntimeError(f"No docking box found for target {target_name}")
        rows = rows.sort_values("site_id")
        row = rows.iloc[0]

        targets.append(
            RedockTarget(
                target=target_name,
                pdb_id=cfg.pdb_id,
                ligand_resname=cfg.ligand_resname,
                receptor_pdbqt=Path(str(row["receptor_pdbqt"])),
                center_x=float(row["center_x"]),
                center_y=float(row["center_y"]),
                center_z=float(row["center_z"]),
                size_x=float(row["size_x"]),
                size_y=float(row["size_y"]),
                size_z=float(row["size_z"]),
            )
        )

    if not targets:
        raise RuntimeError("No redocking targets selected.")
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Redock co-crystal ligands and compute heavy-atom RMSD.")
    parser.add_argument("--targets", type=str, default="", help="Comma-separated target names (default: all with co-crystal ligand).")
    parser.add_argument("--exhaustiveness", type=int, default=16)
    parser.add_argument("--num-modes", type=int, default=9)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rmsd-pass", type=float, default=2.0, help="RMSD threshold (A) for pass rate summary.")
    args = parser.parse_args()

    target_subset = [x.strip() for x in args.targets.split(",") if x.strip()] if args.targets else None
    targets = load_redock_targets(target_subset)

    out_root = RESULTS / "redocking"
    ref_dir = out_root / "reference_ligands"
    pose_dir = out_root / "docked_poses"
    log_dir = out_root / "logs"
    out_root.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []

    for t in targets:
        status = "ok"
        error_msg = ""
        best_affinity = np.nan
        rmsd = np.nan
        rmsd_mode1 = np.nan
        rmsd_min = np.nan
        best_mode = np.nan
        n_ref_atoms = np.nan
        n_pose_atoms = np.nan
        residue_tag = ""

        ref_pdb = PDB_DIR / f"{t.pdb_id}.pdb"
        ref_lig_pdbqt = ref_dir / f"{t.target}_{t.ligand_resname}_ref.pdbqt"
        dock_out = pose_dir / f"{t.target}_{t.ligand_resname}_redock.pdbqt"
        log_path = log_dir / f"{t.target}_{t.ligand_resname}.log"

        try:
            ref_mol, residue_tag = extract_ligand_from_pdb(
                ref_pdb,
                t.ligand_resname,
                center=(t.center_x, t.center_y, t.center_z),
            )
            n_ref_atoms = int(ref_mol.NumAtoms())
            write_pdbqt(ref_mol, ref_lig_pdbqt)
            if dock_out.exists():
                dock_out.unlink()

            cmd = [
                "vina",
                "--receptor",
                str(t.receptor_pdbqt),
                "--ligand",
                str(ref_lig_pdbqt),
                "--center_x",
                str(t.center_x),
                "--center_y",
                str(t.center_y),
                "--center_z",
                str(t.center_z),
                "--size_x",
                str(t.size_x),
                "--size_y",
                str(t.size_y),
                "--size_z",
                str(t.size_z),
                "--exhaustiveness",
                str(args.exhaustiveness),
                "--num_modes",
                str(args.num_modes),
                "--seed",
                str(args.seed),
                "--out",
                str(dock_out),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
            if proc.returncode != 0:
                raise RuntimeError(f"Vina failed (code {proc.returncode})")

            best_affinity_parsed = parse_best_affinity(proc.stdout)
            if best_affinity_parsed is not None:
                best_affinity = float(best_affinity_parsed)

            pose_mols = read_all_poses_pdbqt(dock_out)
            n_pose_atoms = int(pose_mols[0].NumAtoms()) if pose_mols else np.nan
            rmsds = [heavy_atom_rmsd(ref_mol, pm) for pm in pose_mols]
            if rmsds:
                rmsd_mode1 = float(rmsds[0])
                rmsd_min = float(np.min(rmsds))
                best_mode = int(np.argmin(rmsds) + 1)
                rmsd = rmsd_mode1

        except Exception as exc:
            status = "failed"
            error_msg = str(exc)

        rows.append(
            {
                "target": t.target,
                "pdb_id": t.pdb_id,
                "ligand_resname": t.ligand_resname,
                "selected_residue": residue_tag,
                "reference_pdb": str(ref_pdb),
                "receptor_pdbqt": str(t.receptor_pdbqt),
                "reference_ligand_pdbqt": str(ref_lig_pdbqt),
                "docked_pose_pdbqt": str(dock_out),
                "log_file": str(log_path),
                "center_x": t.center_x,
                "center_y": t.center_y,
                "center_z": t.center_z,
                "size_x": t.size_x,
                "size_y": t.size_y,
                "size_z": t.size_z,
                "exhaustiveness": args.exhaustiveness,
                "num_modes": args.num_modes,
                "seed": args.seed,
                "n_ref_atoms": n_ref_atoms,
                "n_pose_atoms": n_pose_atoms,
                "best_affinity_kcal_mol": best_affinity,
                "rmsd_heavy_A": rmsd,
                "rmsd_mode1_A": rmsd_mode1,
                "rmsd_min_A": rmsd_min,
                "mode_of_best_rmsd": best_mode,
                "pass_rmsd_mode1": int(np.isfinite(rmsd_mode1) and rmsd_mode1 <= args.rmsd_pass),
                "pass_rmsd_min": int(np.isfinite(rmsd_min) and rmsd_min <= args.rmsd_pass),
                "status": status,
                "error": error_msg,
            }
        )

    out_csv = out_root / "redocking_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    df = pd.DataFrame(rows)
    ok_df = df[df["status"] == "ok"].copy()

    summary = {
        "n_targets": int(len(df)),
        "n_success": int((df["status"] == "ok").sum()),
        "n_failed": int((df["status"] != "ok").sum()),
        "rmsd_pass_threshold_A": float(args.rmsd_pass),
        "pass_rate_mode1": float(ok_df["pass_rmsd_mode1"].mean()) if not ok_df.empty else float("nan"),
        "pass_rate_min": float(ok_df["pass_rmsd_min"].mean()) if not ok_df.empty else float("nan"),
        "median_rmsd_mode1_A": float(ok_df["rmsd_mode1_A"].median()) if not ok_df.empty else float("nan"),
        "median_rmsd_min_A": float(ok_df["rmsd_min_A"].median()) if not ok_df.empty else float("nan"),
        "mean_rmsd_mode1_A": float(ok_df["rmsd_mode1_A"].mean()) if not ok_df.empty else float("nan"),
        "mean_rmsd_min_A": float(ok_df["rmsd_min_A"].mean()) if not ok_df.empty else float("nan"),
        "max_rmsd_mode1_A": float(ok_df["rmsd_mode1_A"].max()) if not ok_df.empty else float("nan"),
        "max_rmsd_min_A": float(ok_df["rmsd_min_A"].max()) if not ok_df.empty else float("nan"),
    }

    out_json = out_root / "redocking_metrics.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Redocking targets: {summary['n_targets']}")
    print(f"Success: {summary['n_success']} | Failed: {summary['n_failed']}")
    print(f"Summary CSV: {out_csv}")
    print(f"Metrics JSON: {out_json}")


if __name__ == "__main__":
    main()
