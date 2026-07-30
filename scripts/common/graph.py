"""Graph and neighborhood utilities for molecular topologies."""

from __future__ import annotations

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
