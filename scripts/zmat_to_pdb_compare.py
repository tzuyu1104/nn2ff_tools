#!/usr/bin/env python3
"""Convert Gaussian Z-matrix GJF to Cartesian PDB and compare to a reference PDB.

Example:
    python scripts/zmat_to_pdb_compare.py \
      --gjf samples/nBME_opt_zmat.gjf \
      --reference samples/nBME_opt.pdb \
      --output-pdb samples/nBME_opt_trans2.pdb
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

try:
    from .common import zmat
except ImportError:
    try:
        from scripts.common.importing import ensure_scripts_importable
    except ImportError:
        import sys
        from pathlib import Path

        _repo_root = Path(__file__).resolve().parent.parent
        repo_root_str = str(_repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
        sys.modules.pop("scripts", None)
        from scripts.common.importing import ensure_scripts_importable

    ensure_scripts_importable(__file__)
    from scripts.common import zmat


def kabsch_align(mobile: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Kabsch RMS optimal rotation. Returns (rmsd, aligned_mobile, aligned_target)."""
    centroid_m = mobile.mean(axis=0)
    centroid_t = target.mean(axis=0)

    mobile_c = mobile - centroid_m
    target_c = target - centroid_t

    cov = mobile_c.T @ target_c
    u, s, vh = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(np.outer(u, vh)))
    rotation = np.outer(u, vh) * np.diag([1, 1, d])

    aligned = mobile @ rotation.T
    return float(np.linalg.norm(aligned - target_c) / math.sqrt(len(target))), aligned, target_c


def rmsd(positions1: np.ndarray, positions2: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((positions1 - positions2) ** 2, axis=1))))


def map_atoms_by_element_nearest(
    mobile: np.ndarray,
    target: np.ndarray,
    mobile_symbols: list[str],
    target_symbols: list[str],
) -> list[int]:
    mob_by_elem: dict[str, list[int]] = {}
    tar_by_elem: dict[str, list[int]] = {}

    for i, s in enumerate(mobile_symbols):
        mob_by_elem.setdefault(s, []).append(i)
    for j, s in enumerate(target_symbols):
        tar_by_elem.setdefault(s, []).append(j)

    if set(mob_by_elem.keys()) != set(tar_by_elem.keys()):
        raise ValueError("Reference and transformed structures have different element sets.")

    mapping: list[int] = [-1] * len(mobile_symbols)

    for elem, mob_idx in mob_by_elem.items():
        tar_idx = tar_by_elem[elem]
        if len(mob_idx) != len(tar_idx):
            raise ValueError(f"Element count mismatch for {elem}: {len(mob_idx)} vs {len(tar_idx)}")

        unused_tar = set(tar_idx)
        for i in mob_idx:
            dlist = sorted((float(np.linalg.norm(mobile[i] - target[j])), j) for j in unused_tar)
            chosen = dlist[0][1]
            mapping[i] = chosen
            unused_tar.remove(chosen)

    if any(v < 0 for v in mapping):
        raise RuntimeError("Internal mapping failure: some atoms were not assigned.")
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform Gaussian Z-matrix GJF to Cartesian PDB and compare against reference PDB."
    )
    parser.add_argument("--gjf", type=Path, required=True, help="Input Gaussian Z-matrix GJF file.")
    parser.add_argument("--reference", type=Path, required=True, help="Reference Cartesian PDB file.")
    parser.add_argument(
        "--output-pdb",
        type=Path,
        default=Path("samples/nBME_opt_trans2.pdb"),
        help="Output transformed Cartesian PDB file.",
    )
    parser.add_argument(
        "--aligned-pdb",
        type=Path,
        default=Path("samples/nBME_opt_trans2_aligned.pdb"),
        help="Output aligned transformed PDB file.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("samples/nBME_opt_trans2_compare.txt"),
        help="Output text report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gjf_path = args.gjf.expanduser().resolve()
    ref_path = args.reference.expanduser().resolve()
    out_pdb = args.output_pdb.expanduser().resolve()
    out_aligned = args.aligned_pdb.expanduser().resolve()
    out_report = args.report.expanduser().resolve()

    _, _, rows, variables = zmat.parse_gjf_zmatrix(gjf_path)
    transformed_atoms = zmat.zmat_to_atoms(rows, variables)

    reference_atoms = read(str(ref_path))
    ref_symbols = reference_atoms.get_chemical_symbols()
    trans_symbols = transformed_atoms.get_chemical_symbols()

    if len(ref_symbols) != len(trans_symbols):
        raise ValueError("Reference and transformed structures have different atom counts.")

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_pdb), transformed_atoms)

    ref_pos = reference_atoms.get_positions()
    trans_pos = transformed_atoms.get_positions()

    # Structures can have different atom ordering; build a symbol-aware mapping first.
    mapping = map_atoms_by_element_nearest(trans_pos, ref_pos, trans_symbols, ref_symbols)
    ref_pos_mapped = ref_pos[np.array(mapping, dtype=int)]

    rmsd_before = rmsd(trans_pos, ref_pos_mapped)

    _, aligned_pos, _ = kabsch_align(trans_pos, ref_pos_mapped)
    rmsd_after = rmsd(aligned_pos, ref_pos_mapped)

    aligned_atoms = transformed_atoms.copy()
    aligned_atoms.set_positions(aligned_pos)

    out_aligned.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_aligned), aligned_atoms)

    per_atom_dev = np.linalg.norm(aligned_pos - ref_pos_mapped, axis=1)
    max_idx = int(np.argmax(per_atom_dev))

    out_report.parent.mkdir(parents=True, exist_ok=True)
    with out_report.open("w", encoding="utf-8") as f:
        f.write("# ZMAT to Cartesian comparison report\n")
        f.write(f"input_gjf: {gjf_path}\n")
        f.write(f"reference_pdb: {ref_path}\n")
        f.write(f"transformed_pdb: {out_pdb}\n")
        f.write(f"aligned_pdb: {out_aligned}\n\n")
        f.write(f"atom_count: {len(ref_symbols)}\n")
        f.write("atom_mapping: transformed_index -> reference_index (1-based)\n")
        f.write(
            "mapping_pairs: "
            + ", ".join(f"{i + 1}->{mapping[i] + 1}" for i in range(len(mapping)))
            + "\n"
        )
        f.write(f"rmsd_before_alignment_A: {rmsd_before:.10f}\n")
        f.write(f"rmsd_after_alignment_A: {rmsd_after:.10f}\n")
        f.write(f"max_atom_deviation_after_alignment_A: {per_atom_dev[max_idx]:.10f}\n")
        f.write(f"max_atom_index_1based: {max_idx + 1}\n")

    print(f"Transformed PDB written: {out_pdb}")
    print(f"Aligned PDB written: {out_aligned}")
    print(f"Comparison report written: {out_report}")
    print(f"RMSD before alignment (A): {rmsd_before:.10f}")
    print(f"RMSD after alignment (A): {rmsd_after:.10f}")


if __name__ == "__main__":
    main()
