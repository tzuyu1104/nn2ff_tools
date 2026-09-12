"""Graph and neighborhood utilities for molecular topologies."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

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


def dfs_component(
    adjacency: Adjacency,
    start: int,
    blocked_edges: Iterable[Tuple[int, int]] = (),
    blocked_nodes: Iterable[int] = (),
) -> Set[int]:
    """Return a connected component from start under edge/node blocking."""
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
        for nei in adjacency.get(node, set()):
            if tuple(sorted((node, nei))) in blocked_edge_set:
                continue
            if nei not in comp and nei not in blocked_node_set:
                stack.append(nei)
    return comp


def connected_components(adjacency: Adjacency) -> List[Set[int]]:
    """Find all connected components in an undirected adjacency map."""
    visited: Set[int] = set()
    comps: List[Set[int]] = []
    for node in adjacency:
        if node in visited:
            continue
        comp = dfs_component(adjacency, node)
        comps.append(comp)
        visited.update(comp)
    return comps


def has_cycle_undirected(nodes: Iterable[int], adjacency: Adjacency) -> bool:
    """Return True when the undirected subgraph over nodes contains a cycle."""
    node_set = set(nodes)
    visited: Set[int] = set()

    def _dfs(node: int, parent: Optional[int]) -> bool:
        visited.add(node)
        for nei in adjacency.get(node, set()):
            if nei not in node_set:
                continue
            if nei not in visited:
                if _dfs(nei, node):
                    return True
            elif nei != parent:
                return True
        return False

    for node in node_set:
        if node not in visited and _dfs(node, None):
            return True
    return False


def bridge_bonds(adjacency: Adjacency) -> Set[Tuple[int, int]]:
    """Find cut edges (bridge bonds) with Tarjan low-link traversal."""
    bridges: Set[Tuple[int, int]] = set()
    visited: Set[int] = set()
    disc: Dict[int, int] = {}
    low: Dict[int, int] = {}
    timer = 0

    def _dfs(u: int, parent: Optional[int]) -> None:
        nonlocal timer
        visited.add(u)
        disc[u] = timer
        low[u] = timer
        timer += 1

        for v in adjacency.get(u, set()):
            if v not in visited:
                _dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.add(tuple(sorted((u, v))))
            elif v != parent:
                low[u] = min(low[u], disc[v])

    for node in adjacency:
        if node not in visited:
            _dfs(node, None)

    return bridges


def format_fragment(fragment: Set[int]) -> str:
    """Format 0-based atom indices as a 1-based sorted list string."""
    vals = sorted(v + 1 for v in fragment)
    return "[" + ", ".join(str(v) for v in vals) + "]"


def normalize_atom_label(label: str) -> str:
    """Normalize atom labels by removing stereo/charge decorations and case."""
    if label is None:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", str(label)).upper()
    return cleaned


def _build_adjacency(atom_types: List[str], bonds: object) -> Adjacency:
    """Build a symmetric adjacency map from atom labels and bond data."""
    n = len(atom_types)
    adjacency: Adjacency = {i: set() for i in range(n)}

    if isinstance(bonds, dict):
        for src, neighbors in bonds.items():
            src_idx = int(src)
            if 0 <= src_idx < n:
                adjacency[src_idx] = {
                    int(dst) for dst in neighbors if 0 <= int(dst) < n and int(dst) != src_idx
                }
        return adjacency

    for edge in bonds:
        if len(edge) != 2:
            continue
        u, v = edge
        u = int(u)
        v = int(v)
        if 0 <= u < n and 0 <= v < n and u != v:
            adjacency[u].add(v)
            adjacency[v].add(u)

    return adjacency


def _graph_signature(adjacency: Adjacency) -> Tuple[object, ...]:
    """Create a deterministic, order-invariant graph signature for the unlabeled bond graph."""
    colors = [len(adjacency[i]) for i in range(len(adjacency))]
    for _ in range(min(4, len(adjacency))):
        signatures = []
        for index in range(len(adjacency)):
            neighbor_colors = tuple(sorted(colors[j] for j in adjacency[index]))
            signatures.append((colors[index], neighbor_colors))
        unique_signatures = sorted(set(signatures))
        color_map = {signature: idx for idx, signature in enumerate(unique_signatures)}
        colors = [color_map[signature] for signature in signatures]
    return tuple(sorted(colors))


def same_topology(
    atom_types_a: List[str],
    bonds_a: object,
    atom_types_b: List[str],
    bonds_b: object,
) -> bool:
    """Return True when two structures share the same atom-type multiset and bond topology."""
    labels_a = [normalize_atom_label(x) for x in atom_types_a]
    labels_b = [normalize_atom_label(x) for x in atom_types_b]

    if len(labels_a) != len(labels_b):
        return False

    if Counter(labels_a) != Counter(labels_b):
        return False

    adjacency_a = _build_adjacency(labels_a, bonds_a)
    adjacency_b = _build_adjacency(labels_b, bonds_b)

    if sum(len(neighbors) for neighbors in adjacency_a.values()) != sum(
        len(neighbors) for neighbors in adjacency_b.values()
    ):
        return False

    if sorted(len(neighbors) for neighbors in adjacency_a.values()) != sorted(
        len(neighbors) for neighbors in adjacency_b.values()
    ):
        return False

    return _graph_signature(adjacency_a) == _graph_signature(adjacency_b)


def parse_pdb_atoms(pdb_path: Path) -> List[str]:
    """Parse ATOM/HETATM records and return element-like labels in order."""
    atom_types: List[str] = []
    with pdb_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            element = line[76:78].strip()
            if not element:
                atom_name = line[12:16].strip()
                if atom_name:
                    element = atom_name[0]

            if element:
                atom_types.append(element)

    return atom_types
