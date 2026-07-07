"""Z-matrix GJF parsing and coordinate reconstruction.

Unified from the duplicates found in:
- scripts/mace_polar_ff_pot.py
- scripts/zmat_to_pdb_compare.py
- scripts/rot_fragment_lib.py
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms


@dataclass
class ZRow:
    """One row of a Gaussian Z-matrix entry."""

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
    """Resolve a Z-matrix token to a numeric value.

    Handles numeric literals and variable look-ups.
    """
    if tok is None:
        raise ValueError("Missing Z-matrix token.")
    if is_number_token(tok):
        return float(tok)
    if tok not in variables:
        raise ValueError(f"Variable {tok} not found in GJF variable block.")
    return float(variables[tok])


def parse_gjf_zmatrix(gjf_path: Path) -> Tuple[int, int, List[ZRow], Dict[str, float]]:
    """Parse a Gaussian Z-matrix GJF file.

    Returns (charge, multiplicity, rows, variables).
    """
    lines = gjf_path.read_text(encoding="utf-8").splitlines()
    idx = 0
    n = len(lines)

    # Skip link0 and route lines and leading blanks.
    while idx < n and (
        lines[idx].strip().startswith("%")
        or lines[idx].strip().startswith("#")
        or lines[idx].strip() == ""
    ):
        idx += 1

    # Title block until blank line.
    while idx < n and lines[idx].strip() != "":
        idx += 1

    # Blank after title.
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
            rows.append(
                ZRow(symbol=t[0], b=int(t[1]), r_token=t[2], a=int(t[3]), ang_token=t[4])
            )
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
        if not m:
            continue
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


def zmat_rows_to_atoms(rows: List[ZRow], variables: Dict[str, float]) -> Atoms:
    """Convert parsed Z-matrix rows and variables to ASE Atoms Cartesian structure."""
    symbols: List[str] = []
    coords: List[np.ndarray] = []

    for i, row in enumerate(rows):
        symbols.append(row.symbol)

        if i == 0:
            coords.append(np.array([0.0, 0.0, 0.0], dtype=float))
            continue

        if i == 1:
            r = resolve_token(row.r_token or "", variables)
            coords.append(np.array([0.0, 0.0, r], dtype=float))
            continue

        if i == 2:
            r = resolve_token(row.r_token or "", variables)
            ang = resolve_token(row.ang_token or "", variables)
            b = (row.b or 1) - 1
            a = (row.a or 1) - 1

            pb = coords[b]
            pa = coords[a]
            pd = pa + np.array([1.0, 0.0, 0.0], dtype=float)
            coords.append(place_atom(pb, pa, pd, r, ang, 0.0))
            continue

        r = resolve_token(row.r_token or "", variables)
        ang = resolve_token(row.ang_token or "", variables)
        dih = resolve_token(row.dih_token or "", variables)
        b = (row.b or 1) - 1
        a = (row.a or 1) - 1
        d = (row.d or 1) - 1

        pb = coords[b]
        pa = coords[a]
        pd = coords[d]
        coords.append(place_atom(pb, pa, pd, r, ang, dih))

    return Atoms(symbols=symbols, positions=np.vstack(coords))


def zmat_to_atoms(rows: List[ZRow], variables: Dict[str, float]) -> Atoms:
    """Alias for zmat_rows_to_atoms for backward compatibility."""
    return zmat_rows_to_atoms(rows, variables)


def adjacency_from_zmat_rows(rows: List[ZRow]) -> Dict[int, Set[int]]:
    """Build an undirected adjacency from Z-matrix rows via bond references (row.b)."""
    from typing import Set
    natoms = len(rows)
    adjacency: Dict[int, Set[int]] = {i: set() for i in range(natoms)}
    for i, row in enumerate(rows):
        if i == 0:
            continue
        if row.b is None:
            continue
        j = row.b - 1
        if 0 <= j < natoms and i != j:
            adjacency[i].add(j)
            adjacency[j].add(i)
    return adjacency
