#!/usr/bin/env python3
"""Thin compatibility wrapper re-exporting common.rot and common.zmat.

For backward compatibility: scripts that previously imported from
rot_fragment_lib will continue to work.
"""

from __future__ import annotations


try:
    from .common.rot import (  # noqa: F401
        Adjacency,
        calc_dihedral_deg,
        dfs_component,
        default_scan_range,
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
        write_connectivity_gjf,
        wrap_deg,
        _bfs_tree_order,
    )
    from .common.zmat import (  # noqa: F401
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
    from .common import rot  # noqa: F401
    from .common import zmat  # noqa: F401
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
    from scripts.common.rot import (  # noqa: F401
        Adjacency,
        calc_dihedral_deg,
        dfs_component,
        default_scan_range,
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
        write_connectivity_gjf,
        wrap_deg,
        _bfs_tree_order,
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
    from scripts.common import rot  # noqa: F401
    from scripts.common import zmat  # noqa: F401
