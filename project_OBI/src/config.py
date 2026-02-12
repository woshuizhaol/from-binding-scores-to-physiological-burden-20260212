from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ProteinTarget:
    name: str
    pdb_id: str
    ligand_resname: Optional[str]
    box_size: Tuple[float, float, float] = (26.0, 26.0, 26.0)
    multi_site_centers: Optional[List[Tuple[float, float, float]]] = None
    axis: str = "generic"


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA_RAW = ROOT / "data_raw"
DATA_PROCESSED = ROOT / "data_processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

# Provided by user.
SOURCE_DIR = DATA_RAW / "source"
PDB_DIR = DATA_RAW / "pdb"

TARGETS: Dict[str, ProteinTarget] = {
    "HSA": ProteinTarget(
        name="HSA",
        pdb_id="1E7G",
        ligand_resname=None,
        multi_site_centers=[
            (-1.214, 4.148, 39.286),
            (11.889, 10.431, 14.193),
            (10.076, 4.110, 19.817),
            (23.744, 5.663, -1.731),
            (35.563, 15.558, 34.937),
            (33.231, 14.498, 7.791),
            (47.731, 14.448, 18.111),
            (54.050, 8.238, 21.460),
        ],
        axis="thyroid",
    ),
    "TTR": ProteinTarget(name="TTR", pdb_id="2F7I", ligand_resname="26C", axis="thyroid"),
    "TBG": ProteinTarget(name="TBG", pdb_id="2CEO", ligand_resname="T44", axis="thyroid"),
    "PPARG": ProteinTarget(name="PPARG", pdb_id="3U9Q", ligand_resname="DKA"),
    "GPR40": ProteinTarget(name="GPR40", pdb_id="4PHU", ligand_resname="2YB"),
    "OAT1": ProteinTarget(name="OAT1", pdb_id="9KL5", ligand_resname="RTO", axis="renal"),
    "OAT4": ProteinTarget(name="OAT4", pdb_id="9U5A", ligand_resname="ZWY", axis="renal"),
    "URAT1": ProteinTarget(name="URAT1", pdb_id="9IRW", ligand_resname="URC", axis="renal"),
}

# Physiological priors used in OBI computation.
# Physiological priors used in OBI computation (baseline values).
# Physiological priors used in OBI computation (baseline values).
PHYSIOLOGY = {
    "HSA_M": 0.6e-3,  # 0.6 mM
    "TTR_M": 5.0e-6,  # 5 uM
    "TBG_M": 2.6e-7,  # 260 nM
    "T4_total_M": 1.0e-7,  # 100 nM
    "T4_free_M": 1.6e-11,  # 16 pM
    # Approximate Kd of T4 with transport proteins.
    "Kd_T4_HSA_M": 1.0e-6,
    "Kd_T4_TTR_M": 1.0e-8,
    "Kd_T4_TBG_M": 1.0e-10,
}

# Physiological ranges for uncertainty propagation (min, max).
PHYSIOLOGY_RANGES = {
    "HSA_M": (0.50e-3, 0.78e-3),
    "TTR_M": (3.6e-6, 7.3e-6),
    "TBG_M": (1.8e-7, 3.5e-7),
    "T4_total_M": (60e-9, 140e-9),
}

# Thyroid hormone transport share for weighted TTII.
THYROID_WEIGHTS = {
    "TBG": 0.75,
    "TTR": 0.15,
    "HSA": 0.10,
}

# Default HSA calibration mode for Kd conversion.
DEFAULT_HSA_CALIB = "strong"

# Expression-informed renal transporter contribution weights (sum to 1).
# Used for RTII_weighted; RTII_equal is still reported for sensitivity analysis.
RENAL_WEIGHTS = {
    "OAT1": 0.50,
    "OAT4": 0.20,
    "URAT1": 0.30,
}

# Component weights for OBI aggregation.
OBI_COMPONENT_WEIGHTS = {
    "TTII": 0.50,
    "RTII": 0.50,
}

# NHANES P_PFAS analytes used in this project.
# Each entry must map uniquely by DTXSID in ligand_manifest; no silent fallback.
NHANES_ANALYTE_SPECS: Dict[str, Dict[str, str]] = {
    "LBXPFDE": {
        "short_name": "PFDeA",
        "preferred_name": "Perfluorodecanoic acid",
        "casrn": "335-76-2",
        "dtxsid": "DTXSID3031860",
    },
    "LBXPFHS": {
        "short_name": "PFHxS",
        "preferred_name": "Perfluorohexanesulfonic acid",
        "casrn": "355-46-4",
        "dtxsid": "DTXSID7040150",
    },
    "LBXMPAH": {
        "short_name": "MeFOSAA",
        "preferred_name": "2-(N-methylperfluorooctane sulfonamido)acetic acid",
        "casrn": "2355-31-9",
        "dtxsid": "DTXSID10624392",
    },
    "LBXPFNA": {
        "short_name": "PFNA",
        "preferred_name": "Perfluorononanoic acid",
        "casrn": "375-95-1",
        "dtxsid": "DTXSID8031863",
    },
    "LBXPFUA": {
        "short_name": "PFUA",
        "preferred_name": "Perfluoroundecanoic acid",
        "casrn": "2058-94-8",
        "dtxsid": "DTXSID8047553",
    },
    "LBXNFOA": {
        "short_name": "n-PFOA",
        "preferred_name": "Perfluorooctanoic acid",
        "casrn": "335-67-1",
        "dtxsid": "DTXSID8031865",
    },
    "LBXBFOA": {
        "short_name": "Sb-PFOA",
        "preferred_name": "Perfluorooctanoic acid",
        "casrn": "335-67-1",
        "dtxsid": "DTXSID8031865",
    },
    "LBXNFOS": {
        "short_name": "n-PFOS",
        "preferred_name": "Perfluorooctanesulfonic acid",
        "casrn": "1763-23-1",
        "dtxsid": "DTXSID3031864",
    },
    "LBXMFOS": {
        "short_name": "Sm-PFOS",
        "preferred_name": "Perfluorooctanesulfonic acid",
        "casrn": "1763-23-1",
        "dtxsid": "DTXSID3031864",
    },
}

NHANES_ANALYTES = {k: v["short_name"] for k, v in NHANES_ANALYTE_SPECS.items()}
NHANES_WEIGHT_VAR = "WTSBAPRP"

# This analyte is in NHANES P_PFAS but absent from the provided 430-compound inventory.
# Add it explicitly so the exposure model has no silent drop.
SUPPLEMENTAL_LIGANDS: List[Dict[str, str]] = [
    {
        "dtxsid": "DTXSID10624392",
        "preferred_name": "2-(N-methylperfluorooctane sulfonamido)acetic acid",
        "smiles": "CN(CC(=O)O)S(=O)(=O)C(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F",
    }
]
