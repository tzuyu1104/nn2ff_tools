#!/usr/bin/env python3
"""Scan one Z-matrix parameter and compute MACE-POLAR potentials.

Reuses shared logic from scripts.common.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from ase import Atoms

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise RuntimeError(
        "matplotlib is required. Install in your venv with:\n"
        "pip install matplotlib"
    ) from exc

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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scan one Z-matrix parameter and compute MACE-POLAR potentials."
    )
    ap.add_argument("--input", "-i", type=Path, required=True, help="Gaussian Z-matrix GJF file.")
    ap.add_argument(
        "--param",
        "-p",
        type=str,
        required=True,
        help=(
            "Parameter identifier (R1/A1/D1) or atom-index specifier: "
            "'i-j' bond, 'i-j-k' angle, 'i-j-k-l' dihedral/improper."
        ),
    )
    ap.add_argument("--npoints", type=int, default=36, help="Number of scan points.")
    ap.add_argument("--bins", type=int, default=None, help="Alias for --npoints.")
    ap.add_argument(
        "--range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Scan range as two values: MIN MAX.",
    )
    ap.add_argument("--vmin", type=float, default=None, help="Scan minimum. If omitted, auto range is used.")
    ap.add_argument("--vmax", type=float, default=None, help="Scan maximum. If omitted, auto range is used.")
    ap.add_argument(
        "--model",
        type=str,
        default="~/workspace/mace_polar_1/MACE-POLAR-1-M.model",
        help="Local MACE-POLAR model file.",
    )
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    ap.add_argument("--dtype", type=str, default="float64", help="float64 or float32.")
    ap.add_argument("--spin", type=int, default=1, help="Spin info passed to atoms.info.")
    ap.add_argument("--field", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="External field Ex Ey Ez.")
    ap.add_argument("--csv-out", type=Path, default=None, help="CSV output path.")
    ap.add_argument("--fig-out", type=Path, default=None, help="Figure output path (PNG).")
    args = ap.parse_args()

    input_path = args.input.expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()

    charge, _multiplicity, rows, variables = zmat.parse_gjf_zmatrix(input_path)

    param, param_mode = rot.resolve_parameter_name(args.param, rows, variables)

    npoints = args.bins if args.bins is not None else args.npoints
    if npoints <= 0:
        raise ValueError("Number of scan points/bins must be > 0.")

    center = float(variables[param])
    if args.range is not None:
        vmin, vmax = float(args.range[0]), float(args.range[1])
    elif args.vmin is None or args.vmax is None:
        vmin, vmax = rot.default_scan_range(param, center)
    else:
        vmin, vmax = args.vmin, args.vmax

    values = np.linspace(vmin, vmax, npoints)

    stem = input_path.stem + f"_{param}_scan"
    csv_out = args.csv_out.expanduser().resolve() if args.csv_out else input_path.with_name(stem + ".csv")
    fig_out = args.fig_out.expanduser().resolve() if args.fig_out else input_path.with_name(stem + ".png")

    from mace.calculators import mace_polar

    calc = mace_polar(
        model=str(model_path),
        device=args.device,
        default_dtype=args.dtype,
    )

    energies = []
    for v in values:
        local_vars = dict(variables)
        local_vars[param] = float(v)

        atoms = zmat.zmat_to_atoms(rows, local_vars)
        atoms.info["charge"] = charge
        atoms.info["spin"] = args.spin
        atoms.info["external_field"] = list(args.field)
        atoms.calc = calc

        e = atoms.get_potential_energy()
        energies.append(float(e))

    energies = np.array(energies, dtype=float)

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", encoding="utf-8") as f:
        f.write("parameter,value,potential_energy_eV\n")
        for v, e in zip(values, energies):
            f.write(f"{param},{v:.10f},{e:.12f}\n")

    plt.figure(figsize=(7, 5))
    plt.plot(values, energies, marker="o", ms=3)
    plt.xlabel(param)
    plt.ylabel("Potential Energy (eV)")
    plt.title(f"MACE-POLAR scan: {param}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_out, dpi=180)

    print(f"Input GJF: {input_path}")
    print(f"Requested parameter: {args.param}")
    print(f"Resolved parameter: {param} ({param_mode})")
    print(f"Points: {npoints}")
    print(f"Model: {model_path}")
    print(f"Device: {args.device}")
    print(f"CSV written: {csv_out}")
    print(f"Figure written: {fig_out}")


if __name__ == "__main__":
    main()
