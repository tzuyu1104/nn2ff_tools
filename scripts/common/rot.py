"""Fragment rotation, splitting, and adjacency utilities.

Unified from:
- scripts/rot_fragment_lib.py (core fragment operations)
- scripts/mace_polar_ff_pot.py (parse_specifier, resolve_parameter_name, default_scan_range)
- scripts/mace_polar_pot_rot.py (parse_specifier)

Imports from common.zmat for ZRow and parse_gjf_zmatrix.
"""

from __future__ import annotations

import math
import re
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.io import read

from .zmat import ZRow, parse_gjf_zmatrix, resolve_token, zmat_rows_to_atoms, adjacency_from_zmat_rows  # noqa: F401

Adjacency = Dict[int, Set[int]]


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


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-14:
        raise ValueError("Encountered near-zero vector in coordinate construction.")
    return v / n


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


def rotate_points_about_axis(
    points: np.ndarray, axis_a: np.ndarray, axis_b: np.ndarray, angle_deg: float
) -> np.ndarray:
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


def calc_angle_deg(p_i: np.ndarray, p_j: np.ndarray, p_k: np.ndarray) -> float:
    """Calculate angle i-j-k in degrees."""
    v_i = p_i - p_j
    v_k = p_k - p_j
    v_i_u = _unit(v_i)
    v_k_u = _unit(v_k)
    cos_angle = float(np.clip(np.dot(v_i_u, v_k_u), -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def change_angle(
    atoms: Atoms,
    adjacency: Adjacency,
    i: int,
    j: int,
    k: int,
    target_deg: float,
) -> Atoms:
    """Change angle i-j-k to target_deg by symmetric fragment rotation.

    Both i-side and k-side fragments rotate around the normal vector
    of the angle plane (i-j-k), each by half the angle difference
    (symmetric rotation).

    Indices are 0-based.
    """
    part_i, part_center, part_k = split_by_angle(adjacency, i, j, k)

    pos = atoms.get_positions().copy()
    cur = calc_angle_deg(pos[i], pos[j], pos[k])

    half_diff = (cur - target_deg) / 2.0

    # Normal vector of the angle plane (i-j) x (k-j)
    v_i = pos[i] - pos[j]
    v_k = pos[k] - pos[j]
    normal = np.cross(v_i, v_k)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-14:
        raise ValueError(
            f"Angle i-j-k is collinear (atoms {i+1}-{j+1}-{k+1}), "
            "cannot compute normal vector for rotation."
        )
    normal = normal / norm_len

    # Rotate i-side around normal by +half_diff
    if part_i:
        i_side = sorted(part_i)
        rotated_i = rotate_points_about_axis(pos[i_side], pos[j], pos[j] + normal, half_diff)
        pos[i_side] = rotated_i

    # Rotate k-side around normal by -half_diff
    if part_k:
        k_side = sorted(part_k)
        rotated_k = rotate_points_about_axis(pos[k_side], pos[j], pos[j] + normal, -half_diff)
        pos[k_side] = rotated_k

    out = atoms.copy()
    out.set_positions(pos)
    return out


def stretch_bond(
    atoms: Atoms,
    adjacency: Adjacency,
    i: int,
    j: int,
    target_dist: float,
) -> Atoms:
    """Stretch/compress bond i-j to target_dist by symmetric translation.

    The i-side fragment moves by -offset along the i->j direction.
    The j-side fragment moves by +offset along the i->j direction.
    This preserves angles (no rotation) and center of mass.

    Indices are 0-based.
    """
    part_i, part_j = split_by_bond(adjacency, i, j)

    pos = atoms.get_positions().copy()
    cur_dist = float(np.linalg.norm(pos[j] - pos[i]))

    if cur_dist < 1e-14:
        raise ValueError(
            f"Bond ({i + 1}, {j + 1}) has near-zero length, cannot stretch."
        )

    offset = (target_dist - cur_dist) / 2.0
    direction = (pos[j] - pos[i]) / cur_dist

    # Move i-side by -offset along bond direction
    if part_i:
        i_side = sorted(part_i)
        pos[i_side] -= offset * direction

    # Move j-side by +offset along bond direction
    if part_j:
        j_side = sorted(part_j)
        pos[j_side] += offset * direction

    out = atoms.copy()
    out.set_positions(pos)
    return out


def write_connectivity_gjf(
    atoms: Atoms,
    adjacency: Adjacency,
    output_path: Path,
    charge: int = 0,
    multiplicity: int = 1,
    route: str = "#p hf/sto-3g geom=connectivity",
    title: str = "Connectivity geometry",
) -> None:
    """Write Gaussian Cartesian GJF with explicit connectivity section."""
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


def parse_specifier(specifier: str) -> List[int]:
    """Parse an atom-index specifier like 'i-j', 'i-j-k', or 'i-j-k-l'."""
    parts = [p.strip() for p in specifier.split("-") if p.strip()]
    if len(parts) not in (2, 3, 4):
        raise ValueError(
            f"Invalid specifier '{specifier}'. Use i-j, i-j-k, or i-j-k-l."
        )

    try:
        idx = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid specifier '{specifier}': indices must be integers.") from exc

    if any(v <= 0 for v in idx):
        raise ValueError(f"Invalid specifier '{specifier}': indices must be >= 1.")
    return idx


def default_scan_range(param: str, center: float) -> Tuple[float, float]:
    """Return a default scan range based on parameter type."""
    ptype = param[0]
    if ptype == "R":
        return center * 0.5, center * 1.5
    elif ptype == "A":
        return max(10.0, center - 30.0), min(170.0, center + 30.0)
    elif ptype == "D":
        return -180.0, 180.0
    else:
        raise ValueError(f"Unknown parameter type '{ptype}' in '{param}'.")


def resolve_parameter_name(
    key: str, rows: List[ZRow], variables: Dict[str, float]
) -> Tuple[str, str]:
    """Resolve a parameter specifier to a concrete GJF variable name.

    Accepts:
    - Direct variable name (e.g. R1, A5, D7)
    - Atom-index specifier: 'i-j' (bond), 'i-j-k' (angle), 'i-j-k-l' (dihedral)

    Returns (resolved_variable_name, mode).
    """
    # Check if key is already a valid variable name.
    if key in variables:
        return key, key[0].lower() + "parameter"

    idx = parse_specifier(key)

    # Determine which variable references are needed.
    set_candidates: List[str] = []
    for i, row in enumerate(rows):
        if row.b == idx[0] + 1 and row.r_token is not None:
            set_candidates.append(row.r_token)
        if row.a == idx[1] + 1 and row.ang_token is not None:
            set_candidates.append(row.ang_token)
        if row.d == idx[2] + 1 and row.dih_token is not None:
            set_candidates.append(row.dih_token)

    unique_candidates = sorted(set(set_candidates))

    n = len(idx)
    mode = {2: "bond", 3: "angle", 4: "dihedral"}[n]

    if len(unique_candidates) == 1:
        return unique_candidates[0], mode

    if len(unique_candidates) > 1:
        raise ValueError(
            f"Specifier '{key}' matches multiple parameters: {', '.join(unique_candidates)}. "
            "Use --param with an explicit variable name."
        )

    raise ValueError(
        f"Could not resolve '{key}' to a scan variable. "
        "Use a valid parameter (e.g. D7) or a valid specifier (e.g. 1-2, 3-4-5, 6-7-8-10)."
    )
