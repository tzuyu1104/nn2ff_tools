#!/usr/bin/env python3
"""Utilities for connectivity-aware GJF writing and fragment partitioning.

This module is designed for non-ring molecules in its first version.
It provides:
1) GJF writing with explicit bond connectivity.
2) DFS-based fragment extraction for bond/angle/dihedral pivots.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.io import read

Adjacency = Dict[int, Set[int]]


@dataclass
class ZRow:
    symbol: str
    b: Optional[int] = None
    r_token: Optional[str] = None
    a: Optional[int] = None
    ang_token: Optional[str] = None
    d: Optional[int] = None
    dih_token: Optional[str] = None


def parse_pdb_conect(pdb_path: Path, natoms: int) -> Adjacency:
    """Parse PDB CONECT records into a 0-based undirected adjacency map."""
    adjacency: Adjacency = {i: set() for i in range(natoms)}
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


def infer_adjacency_from_geometry(atoms: Atoms, scale: float = 1.2) -> Adjacency:
    """Infer bonds from distances and covalent radii.

    scale controls sensitivity; larger values produce more bonds.
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    natoms = len(symbols)
    adjacency: Adjacency = {i: set() for i in range(natoms)}

    radii = [covalent_radii[atomic_numbers[s]] for s in symbols]
    for i in range(natoms):
        for j in range(i + 1, natoms):
            cutoff = scale * (radii[i] + radii[j])
            d = float(np.linalg.norm(positions[i] - positions[j]))
            if d <= cutoff:
                adjacency[i].add(j)
                adjacency[j].add(i)
    return adjacency


def load_atoms_and_adjacency(input_path: Path) -> Tuple[Atoms, Adjacency]:
    """Load atoms and best-effort adjacency.

    For PDB files, tries CONECT first and falls back to geometric inference.
    For non-PDB files, uses geometric inference.
    """
    atoms = read(str(input_path))
    if not isinstance(atoms, Atoms):
        raise TypeError(f"Could not load Atoms from {input_path}")

    if input_path.suffix.lower() == ".pdb":
        adjacency = parse_pdb_conect(input_path, len(atoms))
        if all(len(v) == 0 for v in adjacency.values()):
            adjacency = infer_adjacency_from_geometry(atoms)
    else:
        adjacency = infer_adjacency_from_geometry(atoms)

    return atoms, adjacency


def _bfs_tree_order(adjacency: Adjacency, start: int = 0) -> Tuple[List[int], Dict[int, Optional[int]]]:
    """Build a BFS atom order and parent map (0-based)."""
    natoms = len(adjacency)
    order: List[int] = []
    parent: Dict[int, Optional[int]] = {}
    visited: Set[int] = set()

    def visit_component(root: int) -> None:
        q: deque[int] = deque([root])
        visited.add(root)
        parent[root] = None
        while q:
            node = q.popleft()
            order.append(node)
            for nei in sorted(adjacency[node]):
                if nei not in visited:
                    visited.add(nei)
                    parent[nei] = node
                    q.append(nei)

    if start < natoms:
        visit_component(start)
    for i in range(natoms):
        if i not in visited:
            visit_component(i)

    return order, parent


def write_connectivity_gjf(
    atoms: Atoms,
    adjacency: Adjacency,
    output_path: Path,
    charge: int = 0,
    multiplicity: int = 1,
    route: str = "#p hf/sto-3g geom=connectivity",
    title: str = "Connectivity geometry",
) -> None:
    """Write Gaussian Cartesian GJF with explicit connectivity section.

    Connectivity lines use Gaussian format:
    atom_i  atom_j  bond_order ...
    Here we write bond order as 1.0 for all edges.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_positions()

    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"%chk={output_path.stem}.chk\n")
        f.write(route + "\n\n")
        f.write(title + "\n\n")
        f.write(f"{charge} {multiplicity}\n")
        for s, p in zip(symbols, pos):
            f.write(f"{s:2s} {p[0]:16.10f} {p[1]:16.10f} {p[2]:16.10f}\n")
        f.write("\n")

        natoms = len(symbols)
        for i in range(natoms):
            neigh = sorted(j for j in adjacency[i] if j > i)
            if not neigh:
                continue
            fields = [str(i + 1)]
            for j in neigh:
                fields.extend([str(j + 1), "1.0"])
            f.write(" ".join(fields) + "\n")
        f.write("\n")


def dfs_component(
    adjacency: Adjacency,
    start: int,
    blocked_edges: Iterable[Tuple[int, int]] = (),
    blocked_nodes: Iterable[int] = (),
) -> Set[int]:
    """Return connected component from start under edge/node blocking.

    blocked_edges should contain undirected 0-based edges.
    """
    blocked_edge_set = {tuple(sorted(e)) for e in blocked_edges}
    blocked_node_set = set(blocked_nodes)

    if start in blocked_node_set:
        return set()

    comp: Set[int] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in comp or node in blocked_node_set:
            continue
        comp.add(node)
        for nei in adjacency[node]:
            if tuple(sorted((node, nei))) in blocked_edge_set:
                continue
            if nei not in comp and nei not in blocked_node_set:
                stack.append(nei)
    return comp


def split_by_bond(adjacency: Adjacency, i: int, j: int) -> Tuple[Set[int], Set[int]]:
    """Split molecule into two parts by removing bond (i, j), 0-based indices."""
    if j not in adjacency.get(i, set()):
        raise ValueError(f"Bond ({i + 1}, {j + 1}) not present in adjacency.")
    blocked = [(i, j)]
    part_i = dfs_component(adjacency, i, blocked_edges=blocked)
    part_j = dfs_component(adjacency, j, blocked_edges=blocked)
    return part_i, part_j


def split_by_angle(adjacency: Adjacency, i: int, j: int, k: int) -> Tuple[Set[int], Set[int], Set[int]]:
    """Split into three parts around angle i-j-k by removing (j,i) and (j,k)."""
    missing = [
        edge for edge in [(j, i), (j, k)] if edge[1] not in adjacency.get(edge[0], set())
    ]
    if missing:
        bad = ", ".join(f"({a + 1},{b + 1})" for a, b in missing)
        raise ValueError(f"Angle edges not present: {bad}")

    blocked = [(j, i), (j, k)]
    part_i = dfs_component(adjacency, i, blocked_edges=blocked)
    part_k = dfs_component(adjacency, k, blocked_edges=blocked)

    # Center-side fragment: component attached to j after detaching both sides.
    blocked_nodes = set(part_i) | set(part_k)
    part_center = dfs_component(adjacency, j, blocked_nodes=blocked_nodes)

    return part_i, part_center, part_k


def split_by_dihedral(adjacency: Adjacency, i: int, j: int, k: int, l: int) -> Tuple[Set[int], Set[int]]:
    """Split into two parts by cutting central bond (j, k) of dihedral i-j-k-l."""
    if k not in adjacency.get(j, set()):
        raise ValueError(f"Central bond ({j + 1}, {k + 1}) not present in adjacency.")
    return split_by_bond(adjacency, j, k)


def format_fragment(fragment: Set[int]) -> str:
    """Format 0-based fragment indices as sorted 1-based list string."""
    vals = sorted(v + 1 for v in fragment)
    return "[" + ", ".join(str(v) for v in vals) + "]"


def is_number_token(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def resolve_token(tok: str, variables: Dict[str, float]) -> float:
    if is_number_token(tok):
        return float(tok)
    if tok not in variables:
        raise ValueError(f"Variable {tok} not found in GJF variable block.")
    return float(variables[tok])


def parse_gjf_zmatrix(gjf_path: Path) -> Tuple[int, int, List[ZRow], Dict[str, float]]:
    """Parse Gaussian Z-matrix GJF into rows and variable dictionary."""
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
            raise ValueError(f"Unsupported Z-matrix line format at row {i + 1}: {zl}")

    variables: Dict[str, float] = {}
    for vl in var_lines:
        m = re.match(r"^\s*([A-Za-z]\w*)\s*=\s*([-+0-9.eE]+)\s*$", vl)
        if m:
            variables[m.group(1)] = float(m.group(2))

    return charge, multiplicity, rows, variables


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-14:
        raise ValueError("Encountered near-zero vector in coordinate construction.")
    return v / n


def _place_atom(
    pb: np.ndarray,
    pa: np.ndarray,
    pd: np.ndarray,
    r: float,
    ang_deg: float,
    dih_deg: float,
) -> np.ndarray:
    theta = math.radians(ang_deg)
    phi = math.radians(dih_deg)

    ez = _unit(pb - pa)
    v = pa - pd
    en = np.cross(v, ez)
    if np.linalg.norm(en) < 1e-10:
        trial = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(trial, ez)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        en = np.cross(trial, ez)
    ey = _unit(en)
    ex = _unit(np.cross(ey, ez))

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
            coords.append(_place_atom(pb, pa, pd, r, ang, 0.0))
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
        coords.append(_place_atom(pb, pa, pd, r, ang, dih))

    return Atoms(symbols=symbols, positions=np.vstack(coords))


def adjacency_from_zmat_rows(rows: List[ZRow]) -> Adjacency:
    """Build an undirected adjacency from Z-matrix rows via bond references (row.b)."""
    natoms = len(rows)
    adjacency: Adjacency = {i: set() for i in range(natoms)}
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


def calc_dihedral_deg(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    b0 = p2 - p1
    b1 = p3 - p2
    b2 = p4 - p3

    b1_u = _unit(b1)
    v = b0 - np.dot(b0, b1_u) * b1_u
    w = b2 - np.dot(b2, b1_u) * b1_u
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1_u, v), w))
    return math.degrees(math.atan2(y, x))


def wrap_deg(x: float) -> float:
    return ((x + 180.0) % 360.0) - 180.0


def rotate_points_about_axis(points: np.ndarray, axis_a: np.ndarray, axis_b: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate points around axis (axis_a -> axis_b) by angle_deg using Rodrigues formula."""
    theta = math.radians(angle_deg)
    k = _unit(axis_b - axis_a)
    p = points - axis_a
    c = math.cos(theta)
    s = math.sin(theta)
    p_rot = p * c + np.cross(k, p) * s + np.outer(np.dot(p, k), k) * (1.0 - c)
    return p_rot + axis_a


def set_dihedral_by_fragment_rotation(
    atoms: Atoms,
    adjacency: Adjacency,
    i: int,
    j: int,
    k: int,
    l: int,
    target_deg: float,
) -> Atoms:
    """Set dihedral i-j-k-l by rotating the full j-side fragment around bond j-k.

    Indices are 0-based.
    """
    part_j, part_k = split_by_bond(adjacency, j, k)
    if i not in part_j or l not in part_k:
        raise ValueError(
            "Dihedral orientation mismatch for fragment rotation: "
            f"i={i + 1} must be on j-side and l={l + 1} on k-side."
        )

    pos = atoms.get_positions().copy()
    cur = calc_dihedral_deg(pos[i], pos[j], pos[k], pos[l])

    # Determine positive rotation direction sensitivity.
    eps = 1.0
    test_idx = sorted(part_j)
    rotated_test = rotate_points_about_axis(pos[test_idx], pos[j], pos[k], eps)
    pos_test = pos.copy()
    pos_test[test_idx] = rotated_test
    cur_plus = calc_dihedral_deg(pos_test[i], pos_test[j], pos_test[k], pos_test[l])
    dcur = wrap_deg(cur_plus - cur)
    sign = 1.0 if dcur >= 0 else -1.0

    needed = wrap_deg(target_deg - cur)
    apply_angle = needed * sign

    rotated = rotate_points_about_axis(pos[test_idx], pos[j], pos[k], apply_angle)
    pos[test_idx] = rotated

    out = atoms.copy()
    out.set_positions(pos)
    return out
