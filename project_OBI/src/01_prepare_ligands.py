#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from tqdm import tqdm

from config import DATA_PROCESSED, SOURCE_DIR
from utils import ensure_dirs, sanitize_id


def maybe_deprotonate_acidic_groups(mol: Chem.Mol) -> Chem.Mol:
    rw = Chem.RWMol(mol)
    changed = False
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() != 8:
            continue
        if atom.GetFormalCharge() != 0:
            continue
        if atom.GetTotalNumHs() < 1:
            continue
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() == 6:
                has_carbonyl = any(
                    b.GetBondType() == Chem.BondType.DOUBLE and other.GetAtomicNum() == 8
                    for b in nbr.GetBonds()
                    for other in [b.GetOtherAtom(nbr)]
                )
                if has_carbonyl:
                    atom.SetFormalCharge(-1)
                    atom.SetNumExplicitHs(0)
                    atom.SetNoImplicit(True)
                    changed = True
                    break
            if nbr.GetAtomicNum() == 16:
                double_o = sum(
                    1
                    for b in nbr.GetBonds()
                    for other in [b.GetOtherAtom(nbr)]
                    if b.GetBondType() == Chem.BondType.DOUBLE and other.GetAtomicNum() == 8
                )
                if double_o >= 2:
                    atom.SetFormalCharge(-1)
                    atom.SetNumExplicitHs(0)
                    atom.SetNoImplicit(True)
                    changed = True
                    break
    out = rw.GetMol()
    if changed:
        Chem.SanitizeMol(out)
    return out


def build_3d(mol: Chem.Mol, seed: int = 2026) -> Chem.Mol:
    mol = Chem.AddHs(mol, addCoords=True)
    if mol.GetNumConformers() == 0:
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        AllChem.EmbedMolecule(mol, params)
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=400)
    except Exception:
        pass
    return mol


def convert_one_sdf_to_pdbqt(job: Tuple[str, Path, Path]) -> Tuple[str, bool, str]:
    ligand_id, sdf_path, pdbqt_path = job
    cmd = [
        "obabel",
        str(sdf_path),
        "-O",
        str(pdbqt_path),
        "--partialcharge",
        "gasteiger",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = proc.returncode == 0 and pdbqt_path.exists()
    msg = proc.stderr.strip() if proc.stderr.strip() else proc.stdout.strip()
    return ligand_id, ok, msg


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PFAS ligands (neutral + anionic) and PDBQT files.")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    lig_root = DATA_PROCESSED / "ligands_3d"
    sdf_root = lig_root / "sdf"
    pdbqt_root = lig_root / "pdbqt"
    ensure_dirs(
        [
            lig_root,
            sdf_root / "neutral",
            sdf_root / "anionic",
            pdbqt_root / "neutral",
            pdbqt_root / "anionic",
        ]
    )

    sdf_file = SOURCE_DIR / "CCD-Batch-Search_2026-02-11_03_07_57.sdf"
    csv_file = SOURCE_DIR / "Chemical List EPAPFASINV-2026-02-11.csv"

    meta_df = pd.read_csv(csv_file)
    meta_df["DTXSID_SHORT"] = meta_df["DTXSID"].astype(str).str.extract(r"(DTXSID\d+)")
    meta_map = meta_df.set_index("DTXSID_SHORT").to_dict(orient="index")

    suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
    rows: List[Dict[str, str]] = []
    neutral_writer = Chem.SDWriter(str(lig_root / "ligands_neutral.sdf"))
    anionic_writer = Chem.SDWriter(str(lig_root / "ligands_anionic.sdf"))

    conversion_jobs: List[Tuple[str, Path, Path]] = []
    n_ok = 0
    for idx, mol in enumerate(tqdm(suppl, desc="Processing PFAS", total=430), start=1):
        if mol is None:
            continue
        dtxsid = mol.GetProp("DTXSID") if mol.HasProp("DTXSID") else f"UNK_{idx:04d}"
        pref_name = mol.GetProp("PREFERRED_NAME") if mol.HasProp("PREFERRED_NAME") else ""
        if not pref_name and dtxsid in meta_map:
            pref_name = str(meta_map[dtxsid].get("PREFERRED NAME", ""))
        ligand_id = sanitize_id(f"{idx:03d}_{dtxsid}")

        base = Chem.Mol(mol)
        base = Chem.RemoveHs(base)
        anionic = maybe_deprotonate_acidic_groups(base)
        neutral = build_3d(Chem.Mol(base))
        anionic = build_3d(Chem.Mol(anionic))

        neutral.SetProp("_Name", ligand_id)
        anionic.SetProp("_Name", ligand_id)
        neutral.SetProp("ligand_id", ligand_id)
        anionic.SetProp("ligand_id", ligand_id)
        neutral.SetProp("DTXSID", dtxsid)
        anionic.SetProp("DTXSID", dtxsid)

        neutral_writer.write(neutral)
        anionic_writer.write(anionic)

        neutral_sdf = sdf_root / "neutral" / f"{ligand_id}.sdf"
        anionic_sdf = sdf_root / "anionic" / f"{ligand_id}.sdf"
        w1 = Chem.SDWriter(str(neutral_sdf))
        w1.write(neutral)
        w1.close()
        w2 = Chem.SDWriter(str(anionic_sdf))
        w2.write(anionic)
        w2.close()

        conversion_jobs.append((f"{ligand_id}|neutral", neutral_sdf, pdbqt_root / "neutral" / f"{ligand_id}.pdbqt"))
        conversion_jobs.append((f"{ligand_id}|anionic", anionic_sdf, pdbqt_root / "anionic" / f"{ligand_id}.pdbqt"))

        rows.append(
            {
                "ligand_id": ligand_id,
                "DTXSID": dtxsid,
                "preferred_name": pref_name,
                "smiles_neutral": Chem.MolToSmiles(Chem.RemoveHs(neutral), isomericSmiles=True),
                "smiles_anionic": Chem.MolToSmiles(Chem.RemoveHs(anionic), isomericSmiles=True),
                "mw": f"{Descriptors.MolWt(Chem.RemoveHs(neutral)):.4f}",
            }
        )
        n_ok += 1

    neutral_writer.close()
    anionic_writer.close()

    failures: List[Tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_one_sdf_to_pdbqt, job): job for job in conversion_jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="SDF->PDBQT"):
            key, ok, msg = fut.result()
            if not ok:
                failures.append((key, msg))

    manifest = lig_root / "ligand_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fail_file = lig_root / "ligand_pdbqt_failures.tsv"
    with fail_file.open("w", encoding="utf-8") as handle:
        for key, msg in failures:
            handle.write(f"{key}\t{msg}\n")

    print(f"Prepared ligands: {n_ok}")
    print(f"PDBQT conversions: {len(conversion_jobs) - len(failures)}/{len(conversion_jobs)}")
    print(f"Manifest: {manifest}")
    if failures:
        print(f"Failures logged to: {fail_file}")


if __name__ == "__main__":
    main()

