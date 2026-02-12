#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List

from config import DATA_PROCESSED, PDB_DIR, TARGETS
from utils import ensure_dirs, ligand_center_from_pdb, write_json


def download_pdb(pdb_id: str, out_dir: Path) -> Path:
    out_path = out_dir / f"{pdb_id}.pdb"
    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    cmd = ["curl", "-fsSL", url, "-o", str(out_path)]
    subprocess.run(cmd, check=True)
    return out_path


def clean_receptor_pdb(input_pdb: Path, output_pdb: Path) -> None:
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    with input_pdb.open("r", encoding="utf-8", errors="ignore") as inp, output_pdb.open(
        "w", encoding="utf-8"
    ) as out:
        for line in inp:
            if line.startswith("ATOM"):
                out.write(line)
        out.write("TER\nEND\n")


def pdb_to_pdbqt(input_pdb: Path, output_pdbqt: Path) -> None:
    cmd = [
        "obabel",
        str(input_pdb),
        "-O",
        str(output_pdbqt),
        "-xr",
        "-p",
        "7.4",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare receptors and docking boxes.")
    parser.add_argument("--size", type=float, default=26.0, help="Default box size (A).")
    args = parser.parse_args()

    rec_root = DATA_PROCESSED / "receptors_prepared"
    ensure_dirs([PDB_DIR, rec_root, rec_root / "clean_pdb", rec_root / "pdbqt", rec_root / "boxes"])

    boxes: List[Dict[str, object]] = []
    targets_summary: List[Dict[str, object]] = []

    for key, target in TARGETS.items():
        pdb_path = download_pdb(target.pdb_id, PDB_DIR)
        clean_pdb = rec_root / "clean_pdb" / f"{target.pdb_id}_{key}.pdb"
        pdbqt_path = rec_root / "pdbqt" / f"{target.pdb_id}_{key}.pdbqt"

        clean_receptor_pdb(pdb_path, clean_pdb)
        pdb_to_pdbqt(clean_pdb, pdbqt_path)

        if target.multi_site_centers:
            for i, center in enumerate(target.multi_site_centers, start=1):
                boxes.append(
                    {
                        "site_id": f"{key}_site{i}",
                        "target": key,
                        "axis": target.axis,
                        "pdb_id": target.pdb_id,
                        "receptor_pdbqt": str(pdbqt_path),
                        "center_x": center[0],
                        "center_y": center[1],
                        "center_z": center[2],
                        "size_x": target.box_size[0],
                        "size_y": target.box_size[1],
                        "size_z": target.box_size[2],
                    }
                )
        else:
            if not target.ligand_resname:
                raise ValueError(f"Target {key} has no ligand definition for center extraction.")
            center = ligand_center_from_pdb(pdb_path, target.ligand_resname)
            boxes.append(
                {
                    "site_id": f"{key}_site1",
                    "target": key,
                    "axis": target.axis,
                    "pdb_id": target.pdb_id,
                    "ligand_resname": target.ligand_resname,
                    "receptor_pdbqt": str(pdbqt_path),
                    "center_x": center[0],
                    "center_y": center[1],
                    "center_z": center[2],
                    "size_x": target.box_size[0] if target.box_size else args.size,
                    "size_y": target.box_size[1] if target.box_size else args.size,
                    "size_z": target.box_size[2] if target.box_size else args.size,
                }
            )

        targets_summary.append(
            {
                "target": key,
                "pdb_id": target.pdb_id,
                "clean_pdb": str(clean_pdb),
                "pdbqt": str(pdbqt_path),
            }
        )

    boxes_path = rec_root / "boxes" / "docking_boxes.json"
    write_json(boxes_path, {"boxes": boxes})
    write_json(rec_root / "receptor_manifest.json", {"targets": targets_summary})

    # Also write TSV for easier shell inspection.
    tsv = rec_root / "boxes" / "docking_boxes.tsv"
    with tsv.open("w", encoding="utf-8") as handle:
        cols = [
            "site_id",
            "target",
            "axis",
            "pdb_id",
            "ligand_resname",
            "center_x",
            "center_y",
            "center_z",
            "size_x",
            "size_y",
            "size_z",
            "receptor_pdbqt",
        ]
        handle.write("\t".join(cols) + "\n")
        for row in boxes:
            handle.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")

    print(f"Prepared receptors: {len(targets_summary)}")
    print(f"Prepared docking sites: {len(boxes)}")
    print(f"Boxes file: {boxes_path}")


if __name__ == "__main__":
    main()

