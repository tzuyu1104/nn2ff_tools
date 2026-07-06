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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.io import read, write


@dataclass
class ZRow:
    symbol: str
    b: Optional[int] = None
    r_token: Optional[str] = None
    a: Optional[int] = None
    ang_token: Optional[str] = None
    d: Optional[int] = None
    dih_token: Optional[str] = None


def is_number_token(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def resolve_token(tok: str, variables: Dict[str, float]) -> float:
    if tok is None:
        raise ValueError("Missing Z-matrix token.")
    if is_number_token(tok):
        return float(tok)
    if tok not in variables:
        raise ValueError(f"Variable {tok} not found in GJF variable block.")
    return float(variables[tok])


def parse_gjf_zmatrix(gjf_path: Path) -> Tuple[int, int, List[ZRow], Dict[str, float]]:
    lines = gjf_path.read_text(encoding="utf-8").splitlines()
    idx = 0
    n = len(lines)

    while idx < n and (
        lines[idx].strip().startswith("%")
        or lines[idx].strip().startswith("#")
        or lines[idx].strip() == ""
    ):
        idx += 1

    while idx < n and lines[idx].strip() != "":
        idx += 1

    while idx < n and lines[idx].strip() == "":
        idx += 1

    if idx >= n:
        raise ValueError("Could not find charge/multiplicity line.")

    charge_mult = lines[idx].split()
    if len(charge_mult) < 2:
        raise ValueError("Invalid charge/multiplicity line.")
    charge = int(charge_mult[0])
    multiplicity = int(charge_mult[1])
    idx += 1

    z_lines: List[str] = []
    while idx < n and lines[idx].strip() != "":
        z_lines.append(lines[idx].strip())
        idx += 1

    while idx < n and lines[idx].strip() == "":
        idx += 1

    var_lines: List[str] = []
    while idx < n and lines[idx].strip() != "":
        var_lines.append(lines[idx].strip())
        idx += 1

    rows: List[ZRow] = []
    for i, zl in enumerate(z_lines):
        t = zl.split()
        if len(t) == 1:
            rows.append(ZRow(symbol=t[0]))
        elif len(t) == 3:
            rows.append(ZRow(symbol=t[0], b=int(t[1]), r_token=t[2]))
        elif len(t) == 5:
            rows.append(ZRow(symbol=t[0], b=int(t[1]), r_token=t[2], a=int(t[3]), ang_token=t[4]))
        elif len(t) == 7:
            rows.append(
                ZRow(
                    symbol=t[0],
                    b=int(t[1]),
                    r_token=t[2],
                    a=int(t[3]),
                    ang_token=t[4],
                    d=int(t[5]),
                    dih_token=t[6],
                )
            )
        else:
            raise ValueError(f"Unsupported Z-matrix line format at row {i+1}: {zl}")

    variables: Dict[str, float] = {}
    for vl in var_lines:
        m = re.match(r"^\s*([A-Za-z]\w*)\s*=\s*([-+0-9.eE]+)\s*$", vl)
        if m:
            variables[m.group(1)] = float(m.group(2))

    return charge, multiplicity, rows, variables


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-14:
        raise ValueError("Encountered near-zero vector in coordinate construction.")
    return v / n


def place_atom(
    pb: np.ndarray,
    pa: np.ndarray,
    pd: np.ndarray,
    r: float,
    ang_deg: float,
    dih_deg: float,
) -> np.ndarray:
    theta = math.radians(ang_deg)
    phi = math.radians(dih_deg)

    ez = unit(pb - pa)
    v = pa - pd
    en = np.cross(v, ez)
    if np.linalg.norm(en) < 1e-10:
        trial = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(trial, ez)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        en = np.cross(trial, ez)
    ey = unit(en)
    ex = unit(np.cross(ey, ez))

    v_local = -math.cos(theta) * ez + math.sin(theta) * (
        math.cos(phi) * ex + math.sin(phi) * ey
    )
    return pb + r * v_local


def zmat_to_atoms(rows: List[ZRow], variables: Dict[str, float]) -> Atoms:
    symbols: List[str] = []
    coords: List[np.ndarray] = []

    for i, row in enumerate(rows):
        symbols.append(row.symbol)

        if i == 0:
            coords.append(np.array([0.0, 0.0, 0.0], dtype=float))
            continue

        if i == 1:
            r = resolve_token(row.r_token, variables)
            coords.append(np.array([0.0, 0.0, r], dtype=float))
            continue

        if i == 2:
            r = resolve_token(row.r_token, variables)
            ang = resolve_token(row.ang_token, variables)
            b = row.b - 1
            a = row.a - 1

            pb = coords[b]
            pa = coords[a]
            pd = pa + np.array([1.0, 0.0, 0.0], dtype=float)
            coords.append(place_atom(pb, pa, pd, r, ang, 0.0))
            continue

        r = resolve_token(row.r_token, variables)
        ang = resolve_token(row.ang_token, variables)
        dih = resolve_token(row.dih_token, variables)

        b = row.b - 1
        a = row.a - 1
        d = row.d - 1

        pb = coords[b]
        pa = coords[a]
        pd = coords[d]
        coords.append(place_atom(pb, pa, pd, r, ang, dih))

    positions = np.vstack(coords)
    return Atoms(symbols=symbols, positions=positions)


def kabsch_align(mobile: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rotation, translated mobile-aligned coords, and target-centered coords."""
    mob_centroid = mobile.mean(axis=0)
    tar_centroid = target.mean(axis=0)

    mob_centered = mobile - mob_centroid
    tar_centered = target - tar_centroid

    h = mob_centered.T @ tar_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T

    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    aligned = mob_centered @ r + tar_centroid
    return r, aligned, tar_centered + tar_centroid


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    diff2 = np.sum((a - b) ** 2, axis=1)
    return float(np.sqrt(np.mean(diff2)))


def map_atoms_by_element_nearest(
    mobile: np.ndarray,
    target: np.ndarray,
    mobile_symbols: List[str],
    target_symbols: List[str],
) -> List[int]:
    """Map mobile atom indices to target indices using element-wise nearest matching.

    Returns a list mapping each mobile index i -> target index mapping[i].
    """
    mob_by_elem: Dict[str, List[int]] = {}
    tar_by_elem: Dict[str, List[int]] = {}

    for i, s in enumerate(mobile_symbols):
        mob_by_elem.setdefault(s, []).append(i)
    for j, s in enumerate(target_symbols):
        tar_by_elem.setdefault(s, []).append(j)

    if set(mob_by_elem.keys()) != set(tar_by_elem.keys()):
        raise ValueError("Reference and transformed structures have different element sets.")

    mapping = [-1] * len(mobile_symbols)

    for elem, mob_idx in mob_by_elem.items():
        tar_idx = tar_by_elem[elem]
        if len(mob_idx) != len(tar_idx):
            raise ValueError(f"Element count mismatch for {elem}: {len(mob_idx)} vs {len(tar_idx)}")

        # Greedy nearest pairing within each element type.
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

    _, _, rows, variables = parse_gjf_zmatrix(gjf_path)
    transformed_atoms = zmat_to_atoms(rows, variables)

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
