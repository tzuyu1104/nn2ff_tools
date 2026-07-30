#!/usr/bin/env python3
"""First-step rotation/fragment preparation utility (no MACE integration).

Provides two subcommands:
1) write-gjf: write a Cartesian Gaussian input with explicit connectivity.
2) split: use DFS to split fragments by bond/angle/dihedral specifier.

Specifier format (1-based atom indices):
- bond: i-j
- angle: i-j-k
- dihedral: i-j-k-l
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
from ase import Atoms
from ase.io import write

try:
    from .common import zmat
    from .common import rot
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
    from scripts.common import zmat
    from scripts.common import rot


def load_structure_with_adjacency(input_path: Path) -> tuple[Atoms, dict[int, set[int]]]:
    """Load structure and adjacency from either GJF z-matrix or Cartesian file."""
    if input_path.suffix.lower() == ".gjf":
        _, _, rows, variables = zmat.parse_gjf_zmatrix(input_path)
        atoms = zmat.zmat_rows_to_atoms(rows, variables)
        adjacency = zmat.adjacency_from_zmat_rows(rows)
    else:
        atoms, adjacency = rot.load_atoms_and_adjacency(input_path)
    return atoms, adjacency


def parse_specifier(specifier: str) -> List[int]:
    return rot.parse_specifier(specifier)


def cmd_write_gjf(args: argparse.Namespace) -> None:
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    atoms, adjacency = rot.load_atoms_and_adjacency(input_path)
    rot.write_connectivity_gjf(
        atoms,
        adjacency,
        output_path=output_path,
        charge=args.charge,
        multiplicity=args.multiplicity,
        route=args.route,
        title=args.title,
    )

    print(f"Input structure: {input_path}")
    print(f"Output GJF: {output_path}")
    print(f"Atoms: {len(atoms)}")
    print(f"Bonds: {sum(len(v) for v in adjacency.values()) // 2}")


def cmd_split(args: argparse.Namespace) -> None:
    input_path = args.input.expanduser().resolve()
    atoms, adjacency = rot.load_atoms_and_adjacency(input_path)

    idx = parse_specifier(args.spec)
    idx0 = [v - 1 for v in idx]

    print(f"Input structure: {input_path}")
    print(f"Atoms: {len(atoms)}")

    if len(idx0) == 2:
        part_a, part_b = rot.split_by_bond(adjacency, idx0[0], idx0[1])
        print("Mode: bond")
        print(f"Part A (side of {idx[0]}): {rot.format_fragment(part_a)}")
        print(f"Part B (side of {idx[1]}): {rot.format_fragment(part_b)}")

    elif len(idx0) == 3:
        part_i, part_center, part_k = rot.split_by_angle(adjacency, idx0[0], idx0[1], idx0[2])
        print("Mode: angle")
        print(f"Side I (side of {idx[0]}): {rot.format_fragment(part_i)}")
        print(f"Center side (around {idx[1]}): {rot.format_fragment(part_center)}")
        print(f"Side K (side of {idx[2]}): {rot.format_fragment(part_k)}")

    else:
        part_j, part_k = rot.split_by_dihedral(adjacency, idx0[0], idx0[1], idx0[2], idx0[3])
        print("Mode: dihedral")
        print(f"J-side (side of {idx[1]}): {rot.format_fragment(part_j)}")
        print(f"K-side (side of {idx[2]}): {rot.format_fragment(part_k)}")


def cmd_generate_dihedral_series(args: argparse.Namespace) -> None:
    input_path = args.input.expanduser().resolve()
    out_dir = args.outdir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = parse_specifier(args.spec)
    if len(idx) != 4:
        raise ValueError("--spec for generate-dihedral-series must be i-j-k-l.")

    atoms0, adjacency = load_structure_with_adjacency(input_path)

    i, j, k, l = [v - 1 for v in idx]
    angles = list(range(args.start, args.stop + 1, args.step))

    print(f"Input GJF: {input_path}")
    print(f"Dihedral spec: {args.spec}")
    print(f"Output dir: {out_dir}")
    print(f"Frames: {len(angles)}")

    for ang in angles:
        atoms_rot = rot.set_dihedral_by_fragment_rotation(
            atoms0,
            adjacency,
            i=i,
            j=j,
            k=k,
            l=l,
            target_deg=float(ang),
        )
        pos = atoms_rot.get_positions()
        realized = rot.calc_dihedral_deg(pos[i], pos[j], pos[k], pos[l])

        out_name = f"{args.prefix}{ang:03d}.pdb"
        out_path = out_dir / out_name
        write(str(out_path), atoms_rot)
        print(f"Wrote {out_path.name}  target={ang:7.2f}  realized={realized:9.4f}")


def cmd_scan_dihedral_potential(args: argparse.Namespace) -> None:
    from mace.calculators import mace_polar
    import matplotlib.pyplot as plt
    from ase.io import write as ase_write

    input_path = args.input.expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()

    idx = parse_specifier(args.spec)
    if len(idx) != 4:
        raise ValueError("--spec must be i-j-k-l for scan-dihedral-potential.")

    atoms0, adjacency = load_structure_with_adjacency(input_path)

    i, j, k, l = [v - 1 for v in idx]
    angles = np.linspace(args.start, args.stop, args.bins)

    calc = mace_polar(
        model=str(model_path),
        device=args.device,
        default_dtype=args.dtype,
    )

    energies = []
    realized_list = []
    snapshots = []
    for ang in angles:
        atoms_rot = rot.set_dihedral_by_fragment_rotation(
            atoms0,
            adjacency,
            i=i,
            j=j,
            k=k,
            l=l,
            target_deg=float(ang),
        )
        pos = atoms_rot.get_positions()
        realized = rot.calc_dihedral_deg(pos[i], pos[j], pos[k], pos[l])
        realized_list.append(realized)

        atoms_rot.info["charge"] = args.charge
        atoms_rot.info["spin"] = args.spin
        atoms_rot.info["external_field"] = list(args.field)
        atoms_rot.calc = calc

        e = atoms_rot.get_potential_energy()
        energies.append(float(e))
        snapshots.append(atoms_rot)

    energies = np.array(energies, dtype=float)

    csv_out = args.csv_out.expanduser().resolve()
    fig_out = args.fig_out.expanduser().resolve()

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    fig_out.parent.mkdir(parents=True, exist_ok=True)

    with csv_out.open("w", encoding="utf-8") as f:
        f.write("target_dihedral_deg,realized_dihedral_deg,potential_energy_eV\n")
        for target, realized, energy in zip(angles, realized_list, energies):
            f.write(f"{target:.10f},{realized:.10f},{energy:.12f}\n")

    plt.figure(figsize=(7, 5))
    plt.plot(angles, energies, marker="o", ms=3)
    plt.xlabel("Target dihedral (deg)")
    plt.ylabel("Potential Energy (eV)")
    plt.title(f"Rigid-fragment dihedral scan: {args.spec}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_out, dpi=180)

    # Write multi-frame PDB snapshots (OVITO-compatible)
    if args.snapshots_out:
        snap_path = args.snapshots_out.expanduser().resolve()
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        ase_write(str(snap_path), snapshots, format="proteindatabank")
        print(f"Snapshots written: {snap_path} ({len(snapshots)} frames)")

    print(f"Input structure: {input_path}")
    print(f"Dihedral spec: {args.spec}")
    print(f"Model: {model_path}")
    print(f"Device: {args.device}")
    print(f"Points: {args.bins}")
    print(f"CSV written: {csv_out}")
    print(f"Figure written: {fig_out}")


def cmd_scan_angle_potential(args: argparse.Namespace) -> None:
    from mace.calculators import mace_polar
    import matplotlib.pyplot as plt
    from ase.io import write as ase_write

    input_path = args.input.expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()

    idx = parse_specifier(args.spec)
    if len(idx) != 3:
        raise ValueError("--spec must be i-j-k for scan-angle-potential.")

    atoms0, adjacency = load_structure_with_adjacency(input_path)

    i, j, k = [v - 1 for v in idx]
    cur_angle = rot.calc_angle_deg(
        atoms0.get_positions()[i],
        atoms0.get_positions()[j],
        atoms0.get_positions()[k],
    )

    # Default scan range: symmetric around current angle, ±5°
    start = args.start if args.start is not None else cur_angle - 5.0
    stop = args.stop if args.stop is not None else cur_angle + 5.0

    angles = np.linspace(start, stop, args.bins)

    calc = mace_polar(
        model=str(model_path),
        device=args.device,
        default_dtype=args.dtype,
    )

    energies = []
    realized_list = []
    snapshots = []
    for ang in angles:
        atoms_rot = rot.change_angle(
            atoms0,
            adjacency,
            i=i,
            j=j,
            k=k,
            target_deg=float(ang),
        )
        pos = atoms_rot.get_positions()
        realized = rot.calc_angle_deg(pos[i], pos[j], pos[k])
        realized_list.append(realized)

        atoms_rot.info["charge"] = args.charge
        atoms_rot.info["spin"] = args.spin
        atoms_rot.info["external_field"] = list(args.field)
        atoms_rot.calc = calc

        e = atoms_rot.get_potential_energy()
        energies.append(float(e))
        snapshots.append(atoms_rot)

    energies = np.array(energies, dtype=float)

    csv_out = args.csv_out.expanduser().resolve()
    fig_out = args.fig_out.expanduser().resolve()

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    fig_out.parent.mkdir(parents=True, exist_ok=True)

    with csv_out.open("w", encoding="utf-8") as f:
        f.write("target_angle_deg,realized_angle_deg,potential_energy_eV\n")
        for target, realized, energy in zip(angles, realized_list, energies):
            f.write(f"{target:.10f},{realized:.10f},{energy:.12f}\n")

    plt.figure(figsize=(7, 5))
    plt.plot(angles, energies, marker="o", ms=3)
    plt.xlabel("Target angle (deg)")
    plt.ylabel("Potential Energy (eV)")
    plt.title(f"Rigid-fragment angle scan: {args.spec}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_out, dpi=180)

    # Write multi-frame PDB snapshots (OVITO-compatible)
    if args.snapshots_out:
        snap_path = args.snapshots_out.expanduser().resolve()
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        ase_write(str(snap_path), snapshots, format="proteindatabank")
        print(f"Snapshots written: {snap_path} ({len(snapshots)} frames)")

    print(f"Input structure: {input_path}")
    print(f"Angle spec: {args.spec}")
    print(f"Model: {model_path}")
    print(f"Device: {args.device}")
    print(f"Points: {args.bins}")
    print(f"CSV written: {csv_out}")
    print(f"Figure written: {fig_out}")


def cmd_scan_bond_potential(args: argparse.Namespace) -> None:
    from mace.calculators import mace_polar
    import matplotlib.pyplot as plt
    from ase.io import write as ase_write

    input_path = args.input.expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()

    idx = parse_specifier(args.spec)
    if len(idx) != 2:
        raise ValueError("--spec must be i-j for scan-bond-potential.")

    atoms0, adjacency = load_structure_with_adjacency(input_path)

    i, j = [v - 1 for v in idx]
    cur_dist = float(np.linalg.norm(
        atoms0.get_positions()[j] - atoms0.get_positions()[i]
    ))

    # Default scan range: symmetric around current bond length, ±0.3 A
    start = args.start if args.start is not None else cur_dist - 0.3
    stop = args.stop if args.stop is not None else cur_dist + 0.3

    bond_lengths = np.linspace(start, stop, args.bins)

    calc = mace_polar(
        model=str(model_path),
        device=args.device,
        default_dtype=args.dtype,
    )

    energies = []
    realized_list = []
    snapshots = []
    for dist in bond_lengths:
        atoms_stretched = rot.stretch_bond(
            atoms0,
            adjacency,
            i=i,
            j=j,
            target_dist=float(dist),
        )
        pos = atoms_stretched.get_positions()
        realized = float(np.linalg.norm(pos[j] - pos[i]))
        realized_list.append(realized)

        atoms_stretched.info["charge"] = args.charge
        atoms_stretched.info["spin"] = args.spin
        atoms_stretched.info["external_field"] = list(args.field)
        atoms_stretched.calc = calc

        e = atoms_stretched.get_potential_energy()
        energies.append(float(e))
        snapshots.append(atoms_stretched)

    energies = np.array(energies, dtype=float)

    csv_out = args.csv_out.expanduser().resolve()
    fig_out = args.fig_out.expanduser().resolve()

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    fig_out.parent.mkdir(parents=True, exist_ok=True)

    with csv_out.open("w", encoding="utf-8") as f:
        f.write("target_bond_length_A,realized_bond_length_A,potential_energy_eV\n")
        for target, realized, energy in zip(bond_lengths, realized_list, energies):
            f.write(f"{target:.10f},{realized:.10f},{energy:.12f}\n")

    plt.figure(figsize=(7, 5))
    plt.plot(bond_lengths, energies, marker="o", ms=3)
    plt.xlabel("Bond length (A)")
    plt.ylabel("Potential Energy (eV)")
    plt.title(f"Rigid-fragment bond scan: {args.spec}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_out, dpi=180)

    # Write multi-frame PDB snapshots (OVITO-compatible)
    if args.snapshots_out:
        snap_path = args.snapshots_out.expanduser().resolve()
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        ase_write(str(snap_path), snapshots, format="proteindatabank")
        print(f"Snapshots written: {snap_path} ({len(snapshots)} frames)")

    print(f"Input structure: {input_path}")
    print(f"Bond spec: {args.spec}")
    print(f"Model: {model_path}")
    print(f"Device: {args.device}")
    print(f"Points: {args.bins}")
    print(f"CSV written: {csv_out}")
    print(f"Figure written: {fig_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fragment-based rotational tools with optional MACE-POLAR potential scans."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_gjf = sub.add_parser("write-gjf", help="Write Cartesian GJF with explicit connectivity.")
    p_gjf.add_argument("--input", "-i", type=Path, required=True, help="Input structure file (PDB preferred).")
    p_gjf.add_argument("--output", "-o", type=Path, required=True, help="Output GJF path.")
    p_gjf.add_argument("--charge", type=int, default=0, help="Charge for GJF header.")
    p_gjf.add_argument("--multiplicity", type=int, default=1, help="Multiplicity for GJF header.")
    p_gjf.add_argument(
        "--route",
        type=str,
        default="#p hf/sto-3g geom=connectivity",
        help="Gaussian route section.",
    )
    p_gjf.add_argument("--title", type=str, default="Connectivity geometry", help="Gaussian title line.")
    p_gjf.set_defaults(func=cmd_write_gjf)

    p_split = sub.add_parser("split", help="Split fragments by bond/angle/dihedral specifier.")
    p_split.add_argument("--input", "-i", type=Path, required=True, help="Input structure file.")
    p_split.add_argument(
        "--spec",
        "-s",
        type=str,
        required=True,
        help="Specifier: i-j (bond), i-j-k (angle), i-j-k-l (dihedral).",
    )
    p_split.set_defaults(func=cmd_split)

    p_series = sub.add_parser(
        "generate-dihedral-series",
        help="Generate rigid-fragment rotated PDB series for a dihedral i-j-k-l.",
    )
    p_series.add_argument("--input", "-i", type=Path, required=True, help="Input Gaussian Z-matrix GJF file.")
    p_series.add_argument("--spec", "-s", type=str, required=True, help="Dihedral specifier i-j-k-l (1-based).")
    p_series.add_argument("--outdir", "-o", type=Path, required=True, help="Output directory for PDB frames.")
    p_series.add_argument("--start", type=int, default=0, help="Start dihedral value (deg).")
    p_series.add_argument("--stop", type=int, default=330, help="Stop dihedral value (deg), inclusive.")
    p_series.add_argument("--step", type=int, default=30, help="Step size (deg).")
    p_series.add_argument("--prefix", type=str, default="nBME_rotD6_", help="Output filename prefix.")
    p_series.set_defaults(func=cmd_generate_dihedral_series)

    p_scan = sub.add_parser(
        "scan-dihedral-potential",
        help="Rigid-fragment dihedral scan with MACE-POLAR potential evaluation.",
    )
    p_scan.add_argument("--input", "-i", type=Path, required=True, help="Input structure (.gjf z-matrix or .pdb).")
    p_scan.add_argument("--spec", "-s", type=str, required=True, help="Dihedral specifier i-j-k-l (1-based).")
    p_scan.add_argument(
        "--model",
        type=str,
        default="~/workspace/mace_polar_1/MACE-POLAR-1-M.model",
        help="Path to local MACE-POLAR model file.",
    )
    p_scan.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu.")
    p_scan.add_argument("--dtype", type=str, default="float64", help="Calculator dtype.")
    p_scan.add_argument("--charge", type=int, default=0, help="Charge passed to atoms.info.")
    p_scan.add_argument("--spin", type=int, default=1, help="Spin passed to atoms.info.")
    p_scan.add_argument("--field", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="External field Ex Ey Ez.")
    p_scan.add_argument("--start", type=float, default=0.0, help="Start dihedral value (deg).")
    p_scan.add_argument("--stop", type=float, default=360.0, help="Stop dihedral value (deg).")
    p_scan.add_argument("--bins", type=int, default=36, help="Number of points in scan.")
    p_scan.add_argument("--csv-out", type=Path, required=True, help="Output CSV path.")
    p_scan.add_argument("--fig-out", type=Path, required=True, help="Output figure path.")
    p_scan.add_argument("--snapshots-out", type=Path, default=None, help="Output multi-frame PDB path for OVITO visualization (optional).")
    p_scan.set_defaults(func=cmd_scan_dihedral_potential)

    p_scan_angle = sub.add_parser(
        "scan-angle-potential",
        help="Rigid-fragment angle scan with MACE-POLAR potential evaluation.",
    )
    p_scan_angle.add_argument("--input", "-i", type=Path, required=True, help="Input structure (.gjf z-matrix or .pdb).")
    p_scan_angle.add_argument("--spec", "-s", type=str, required=True, help="Angle specifier i-j-k (1-based).")
    p_scan_angle.add_argument(
        "--model",
        type=str,
        default="~/workspace/mace_polar_1/MACE-POLAR-1-M.model",
        help="Path to local MACE-POLAR model file.",
    )
    p_scan_angle.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu.")
    p_scan_angle.add_argument("--dtype", type=str, default="float64", help="Calculator dtype.")
    p_scan_angle.add_argument("--charge", type=int, default=0, help="Charge passed to atoms.info.")
    p_scan_angle.add_argument("--spin", type=int, default=1, help="Spin passed to atoms.info.")
    p_scan_angle.add_argument("--field", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="External field Ex Ey Ez.")
    p_scan_angle.add_argument("--start", type=float, default=None, help="Start angle value (deg). Default: current-5°.")
    p_scan_angle.add_argument("--stop", type=float, default=None, help="Stop angle value (deg). Default: current+5°.")
    p_scan_angle.add_argument("--bins", type=int, default=36, help="Number of points in scan.")
    p_scan_angle.add_argument("--csv-out", type=Path, required=True, help="Output CSV path.")
    p_scan_angle.add_argument("--fig-out", type=Path, required=True, help="Output figure path.")
    p_scan_angle.add_argument("--snapshots-out", type=Path, default=None, help="Output multi-frame PDB path for OVITO visualization (optional).")
    p_scan_angle.set_defaults(func=cmd_scan_angle_potential)

    p_scan_bond = sub.add_parser(
        "scan-bond-potential",
        help="Rigid-fragment bond scan with MACE-POLAR potential evaluation.",
    )
    p_scan_bond.add_argument("--input", "-i", type=Path, required=True, help="Input structure (.gjf z-matrix or .pdb).")
    p_scan_bond.add_argument("--spec", "-s", type=str, required=True, help="Bond specifier i-j (1-based).")
    p_scan_bond.add_argument(
        "--model",
        type=str,
        default="~/workspace/mace_polar_1/MACE-POLAR-1-M.model",
        help="Path to local MACE-POLAR model file.",
    )
    p_scan_bond.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu.")
    p_scan_bond.add_argument("--dtype", type=str, default="float64", help="Calculator dtype.")
    p_scan_bond.add_argument("--charge", type=int, default=0, help="Charge passed to atoms.info.")
    p_scan_bond.add_argument("--spin", type=int, default=1, help="Spin passed to atoms.info.")
    p_scan_bond.add_argument("--field", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="External field Ex Ey Ez.")
    p_scan_bond.add_argument("--start", type=float, default=None, help="Start bond length (A). Default: current-0.3 A.")
    p_scan_bond.add_argument("--stop", type=float, default=None, help="Stop bond length (A). Default: current+0.3 A.")
    p_scan_bond.add_argument("--bins", type=int, default=36, help="Number of points in scan.")
    p_scan_bond.add_argument("--csv-out", type=Path, required=True, help="Output CSV path.")
    p_scan_bond.add_argument("--fig-out", type=Path, required=True, help="Output figure path.")
    p_scan_bond.add_argument("--snapshots-out", type=Path, default=None, help="Output multi-frame PDB path for OVITO visualization (optional).")
    p_scan_bond.set_defaults(func=cmd_scan_bond_potential)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
