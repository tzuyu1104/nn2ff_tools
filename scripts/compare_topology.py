#!/usr/bin/env python3
"""Compare the topology of two PDB structures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common.graph import (
    normalize_atom_label,
    parse_pdb_atoms,
    parse_pdb_conect,
    same_topology,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two PDB files by atom types and bond connectivity only."
    )
    parser.add_argument("left", type=Path, help="First PDB file")
    parser.add_argument("right", type=Path, help="Second PDB file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    left_path = args.left.resolve()
    right_path = args.right.resolve()

    if not left_path.exists():
        raise FileNotFoundError(f"File not found: {left_path}")
    if not right_path.exists():
        raise FileNotFoundError(f"File not found: {right_path}")

    atom_types_a = parse_pdb_atoms(left_path)
    atom_types_b = parse_pdb_atoms(right_path)

    if not atom_types_a:
        raise ValueError(f"No atoms found in {left_path}")
    if not atom_types_b:
        raise ValueError(f"No atoms found in {right_path}")

    bonds_a = parse_pdb_conect(left_path, len(atom_types_a))
    bonds_b = parse_pdb_conect(right_path, len(atom_types_b))

    is_same = same_topology(atom_types_a, bonds_a, atom_types_b, bonds_b)

    print(f"Left:  {left_path}")
    print(f"  atoms: {len(atom_types_a)}")
    print(f"  labels: {sorted(normalize_atom_label(x) for x in atom_types_a)}")

    print(f"Right: {right_path}")
    print(f"  atoms: {len(atom_types_b)}")
    print(f"  labels: {sorted(normalize_atom_label(x) for x in atom_types_b)}")

    print(f"Same topology: {is_same}")
    return 0 if is_same else 1


if __name__ == "__main__":
    raise SystemExit(main())
