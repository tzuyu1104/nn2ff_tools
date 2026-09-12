#!/usr/bin/env python3
"""Thin compatibility wrapper re-exporting common.rot and common.zmat.

For backward compatibility: scripts that previously imported from
rot_fragment_lib will continue to work.
"""

from __future__ import annotations

from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import rot, zmat  # noqa: F401
from scripts.common.rot import (  # noqa: F401
    Adjacency,
    _bfs_tree_order,
    calc_dihedral_deg,
    default_scan_range,
    dfs_component,
    format_fragment,
    infer_adjacency_from_geometry,
    load_atoms_and_adjacency,
    parse_pdb_conect,
    parse_specifier,
    resolve_parameter_name,
    rotate_points_about_axis,
    set_dihedral_by_fragment_rotation,
    split_by_angle,
    split_by_bond,
    split_by_dihedral,
    wrap_deg,
    write_connectivity_gjf,
)
from scripts.common.zmat import (  # noqa: F401
    ZRow,
    adjacency_from_zmat_rows,
    is_number_token,
    parse_gjf_zmatrix,
    place_atom,
    resolve_token,
    unit,
    zmat_rows_to_atoms,
    zmat_to_atoms,
)
