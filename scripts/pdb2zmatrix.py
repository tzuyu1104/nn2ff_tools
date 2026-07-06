#!/usr/bin/env python3
"""Convert a PDB structure to a simple Z-matrix text format.

Usage:
    python scripts/pdb2zmatrix.py \
        --input samples/nBME_opt.pdb \
        --output samples/nBME_opt.zmat
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from ase import Atoms
from ase.io import read


@dataclass
class ZMatrixRow:
    symbol: str
    b: Optional[int] = None
    a: Optional[int] = None
    d: Optional[int] = None
    r: Optional[float] = None
    angle: Optional[float] = None
    dihedral: Optional[float] = None


def parse_pdb_conect(pdb_path: Path, natoms: int) -> Dict[int, Set[int]]:
    """Parse CONECT records from a PDB file into a symmetric adjacency map."""
    adjacency: Dict[int, Set[int]] = {i: set() for i in range(natoms)}

    with pdb_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("CONECT"):
                continue

            fields = line.split()
            if len(fields) < 3:
                continue

            try:
                src = int(fields[1]) - 1
            except ValueError:
                continue

            if src < 0 or src >= natoms:
                continue

            for token in fields[2:]:
                try:
                    dst = int(token) - 1
                except ValueError:
                    continue
                if 0 <= dst < natoms and dst != src:
                    adjacency[src].add(dst)
                    adjacency[dst].add(src)

    return adjacency


def unit_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm


def calc_angle_deg(p_i: np.ndarray, p_j: np.ndarray, p_k: np.ndarray) -> float:
    """Angle i-j-k in degrees (vertex at j)."""
    v1 = p_i - p_j
    v2 = p_k - p_j
    u1 = unit_vector(v1)
    u2 = unit_vector(v2)
    cosine = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def calc_dihedral_deg(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """Dihedral angle for points 1-2-3-4 in degrees."""
    b0 = p2 - p1
    b1 = p3 - p2
    b2 = p4 - p3

    b1_u = unit_vector(b1)

    v = b0 - np.dot(b0, b1_u) * b1_u
    w = b2 - np.dot(b2, b1_u) * b1_u

    x = np.dot(v, w)
    y = np.dot(np.cross(b1_u, v), w)

    return math.degrees(math.atan2(y, x))


def calc_zmatrix_dihedral_deg(p_i: np.ndarray, p_b: np.ndarray, p_a: np.ndarray, p_d: np.ndarray) -> float:
    """Dihedral consistent with common Z-matrix local-frame placement convention."""
    ez = unit_vector(p_b - p_a)
    n = np.cross(p_a - p_d, ez)
    if np.linalg.norm(n) < 1e-10:
        trial = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(trial, ez)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        n = np.cross(trial, ez)
    ey = unit_vector(n)
    ex = unit_vector(np.cross(ey, ez))

    v = unit_vector(p_i - p_b)
    x = float(np.dot(v, ex))
    y = float(np.dot(v, ey))
    return math.degrees(math.atan2(y, x))


def build_atom_order_and_parent(atoms: Atoms, adjacency: Dict[int, Set[int]]) -> Tuple[List[int], Dict[int, Optional[int]]]:
    """Build connectivity-first atom order and a BFS parent map."""
    symbols = atoms.get_chemical_symbols()
    natoms = len(symbols)

    order: List[int] = []
    visited: Set[int] = set()
    parent: Dict[int, Optional[int]] = {}

    def enqueue_component(start: int) -> None:
        queue: deque[int] = deque([start])
        visited.add(start)
        parent[start] = None
        while queue:
            node = queue.popleft()
            order.append(node)

            # Visit heavy atoms first, then hydrogens.
            neighbors = sorted(adjacency[node], key=lambda n: (symbols[n] == "H", n))
            for nei in neighbors:
                if nei not in visited:
                    visited.add(nei)
                    parent[nei] = node
                    queue.append(nei)

    enqueue_component(0)
    for idx in range(natoms):
        if idx not in visited:
            enqueue_component(idx)

    return order, parent


def reframe_to_gaussian_anchor(positions: np.ndarray, atom_order: List[int]) -> np.ndarray:
    """Rotate/translate coordinates to Gaussian-style anchor frame for first 3 atoms."""
    if len(atom_order) < 3:
        return positions.copy()

    p1 = positions[atom_order[0]]
    p2 = positions[atom_order[1]]
    p3 = positions[atom_order[2]]

    ez = unit_vector(p2 - p1)
    v13 = p3 - p1
    v13_perp = v13 - np.dot(v13, ez) * ez
    if np.linalg.norm(v13_perp) < 1e-10:
        trial = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(trial, ez)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        v13_perp = trial - np.dot(trial, ez) * ez
    ex = unit_vector(v13_perp)
    # Enforce atom 3 with positive x in the anchor frame.
    if np.dot(v13, ex) < 0:
        ex = -ex
    ey = unit_vector(np.cross(ez, ex))

    basis = np.column_stack((ex, ey, ez))
    shifted = positions - p1
    return shifted @ basis


def pick_distinct_previous(prev_rows: List[int], forbidden: Set[int], fallback: int) -> int:
    for cand in reversed(prev_rows):
        if cand not in forbidden:
            return cand
    return fallback


def build_zmatrix_rows(atoms: Atoms, adjacency: Dict[int, Set[int]]) -> List[ZMatrixRow]:
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    atom_order, parent = build_atom_order_and_parent(atoms, adjacency)
    positions_anchor = reframe_to_gaussian_anchor(positions, atom_order)

    rows: List[ZMatrixRow] = []
    placed: Set[int] = set()
    placed_order: List[int] = []
    row_map: Dict[int, int] = {}

    for row_idx, atom_idx in enumerate(atom_order):
        symbol = symbols[atom_idx]

        if row_idx == 0:
            rows.append(ZMatrixRow(symbol=symbol))
            placed.add(atom_idx)
            placed_order.append(atom_idx)
            row_map[atom_idx] = row_idx + 1
            continue

        if row_idx == 1:
            b = parent.get(atom_idx)
            if b is None or b not in placed:
                b = placed_order[-1]
            r = np.linalg.norm(positions_anchor[atom_idx] - positions_anchor[b])
            rows.append(ZMatrixRow(symbol=symbol, b=row_map[b], r=float(r)))
            placed.add(atom_idx)
            placed_order.append(atom_idx)
            row_map[atom_idx] = row_idx + 1
            continue

        if row_idx == 2:
            # Keep first three atoms in canonical Gaussian pattern for stable reconstruction.
            b = placed_order[1]
            a = placed_order[0]
            r = np.linalg.norm(positions_anchor[atom_idx] - positions_anchor[b])
            ang = calc_angle_deg(positions_anchor[atom_idx], positions_anchor[b], positions_anchor[a])
            rows.append(
                ZMatrixRow(symbol=symbol, b=row_map[b], r=float(r), a=row_map[a], angle=float(ang))
            )
            placed.add(atom_idx)
            placed_order.append(atom_idx)
            row_map[atom_idx] = row_idx + 1
            continue

        b = parent.get(atom_idx)
        if b is None or b not in placed:
            b = placed_order[-1]

        a = parent.get(b)
        if a is None or a not in placed or a == b:
            a = pick_distinct_previous(placed_order, forbidden={b}, fallback=placed_order[0])

        d = parent.get(a)
        if d is None or d not in placed or d in {a, b}:
            d = pick_distinct_previous(placed_order, forbidden={a, b}, fallback=placed_order[0])

        r = np.linalg.norm(positions_anchor[atom_idx] - positions_anchor[b])
        ang = calc_angle_deg(positions_anchor[atom_idx], positions_anchor[b], positions_anchor[a])
        dih = calc_zmatrix_dihedral_deg(
            positions_anchor[atom_idx], positions_anchor[b], positions_anchor[a], positions_anchor[d]
        )

        rows.append(
            ZMatrixRow(
                symbol=symbol,
                b=row_map[b],
                r=float(r),
                a=row_map[a],
                angle=float(ang),
                d=row_map[d],
                dihedral=float(dih),
            )
        )
        placed.add(atom_idx)
        placed_order.append(atom_idx)
        row_map[atom_idx] = row_idx + 1

    return rows


def write_plain_zmatrix(output_path: Path, input_name: str, rows: List[ZMatrixRow]) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Z-matrix generated from {input_name}\n")
        handle.write("# Columns: atom  b  r(Ang)  a  angle(deg)  d  dihedral(deg)\n")
        for idx, row in enumerate(rows, start=1):
            if idx == 1:
                handle.write(f"{idx:3d}  {row.symbol}\n")
            elif idx == 2:
                handle.write(f"{idx:3d}  {row.symbol:2s} {row.b:3d} {row.r:10.6f}\n")
            elif idx == 3:
                handle.write(
                    f"{idx:3d}  {row.symbol:2s} {row.b:3d} {row.r:10.6f} {row.a:3d} {row.angle:10.4f}\n"
                )
            else:
                handle.write(
                    f"{idx:3d}  {row.symbol:2s} {row.b:3d} {row.r:10.6f} {row.a:3d} {row.angle:10.4f}"
                    f" {row.d:3d} {row.dihedral:10.4f}\n"
                )


def write_gjf_zmatrix(
    output_path: Path,
    input_name: str,
    rows: List[ZMatrixRow],
    charge: int,
    multiplicity: int,
) -> None:
    r_count = 0
    a_count = 0
    d_count = 0
    variable_lines: List[str] = []
    zmat_lines: List[str] = []

    for idx, row in enumerate(rows, start=1):
        if idx == 1:
            zmat_lines.append(f"{row.symbol}")
            continue

        r_count += 1
        r_name = f"R{r_count}"
        variable_lines.append(f"{r_name}={row.r:.6f}")

        if idx == 2:
            zmat_lines.append(f"{row.symbol} {row.b} {r_name}")
            continue

        a_count += 1
        a_name = f"A{a_count}"
        variable_lines.append(f"{a_name}={row.angle:.4f}")

        if idx == 3:
            zmat_lines.append(f"{row.symbol} {row.b} {r_name} {row.a} {a_name}")
            continue

        d_count += 1
        d_name = f"D{d_count}"
        variable_lines.append(f"{d_name}={row.dihedral:.4f}")
        zmat_lines.append(f"{row.symbol} {row.b} {r_name} {row.a} {a_name} {row.d} {d_name}")

    chk_name = output_path.stem + ".chk"
    title = f"Z-matrix generated from {input_name}"

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"%chk={chk_name}\n")
        handle.write("#p opt\n\n")
        handle.write(f"{title}\n\n")
        handle.write(f"{charge} {multiplicity}\n")
        for line in zmat_lines:
            handle.write(f"{line}\n")
        handle.write("\n")
        for line in variable_lines:
            handle.write(f"{line}\n")
        handle.write("\n")


def convert_pdb_to_zmatrix(
    input_path: Path,
    output_path: Path,
    output_format: str,
    charge: int,
    multiplicity: int,
) -> None:
    atoms = read(str(input_path))
    if not isinstance(atoms, Atoms):
        raise TypeError("ASE did not return an Atoms object for this PDB file.")

    adjacency = parse_pdb_conect(input_path, len(atoms))
    rows = build_zmatrix_rows(atoms, adjacency)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "gjf":
        write_gjf_zmatrix(output_path, input_path.name, rows, charge=charge, multiplicity=multiplicity)
    else:
        write_plain_zmatrix(output_path, input_path.name, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PDB to Z-matrix format.")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to input PDB file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to output Z-matrix file. Defaults to input name with .zmat extension.",
    )
    parser.add_argument(
        "--format",
        choices=["plain", "gjf"],
        default="plain",
        help="Output format: plain Z-matrix text or Gaussian GJF Z-matrix.",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Molecular charge for GJF output.",
    )
    parser.add_argument(
        "--multiplicity",
        type=int,
        default=1,
        help="Spin multiplicity for GJF output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if args.output is None:
        output_path = input_path.with_suffix(".zmat" if args.format == "plain" else ".gjf")
    else:
        output_path = args.output.resolve()

    convert_pdb_to_zmatrix(
        input_path,
        output_path,
        output_format=args.format,
        charge=args.charge,
        multiplicity=args.multiplicity,
    )
    print(f"Z-matrix written to: {output_path}")


if __name__ == "__main__":
    main()
