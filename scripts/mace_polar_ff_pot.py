#!/usr/bin/env python3
"""Scan one Z-matrix parameter and compute MACE-POLAR potentials.

Reuses shared logic from scripts.common.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import rot, scanning, zmat


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

    calculator = scanning.create_mace_calculator(model_path, args.device, args.dtype)

    def transform(value: float):
        local_vars = dict(variables)
        local_vars[param] = value

        atoms = zmat.zmat_to_atoms(rows, local_vars)
        return atoms, value

    result = scanning.evaluate_scan(
        values,
        transform,
        calculator,
        charge=charge,
        spin=args.spin,
        field=args.field,
    )
    scanning.write_scan_csv(
        csv_out,
        result,
        target_column="value",
        parameter=param,
    )
    scanning.write_scan_plot(
        fig_out,
        result,
        xlabel=param,
        title=f"MACE-POLAR scan: {param}",
    )

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
