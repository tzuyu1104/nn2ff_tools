#!/usr/bin/env python3
"""Run MACE-POLAR calculation (energy/forces/stress/charges) on a structure file.

Example:
    python scripts/mace_polar_calc.py \
        --input samples/nBME_opt.pdb \
        --model ~/workspace/mace_polar_1/MACE-POLAR-1-M.model \
        --device cuda \
        --output samples/nBME_opt_mace_polar_results.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from ase.io import read

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common.scanning import create_mace_calculator


def to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MACE-POLAR on an ASE-readable structure.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input structure file (e.g. PDB).")
    parser.add_argument("--model", type=str, required=True, help="Path to local MACE-POLAR model file.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: cuda or cpu.")
    parser.add_argument("--dtype", type=str, default="float64", help="Default dtype for MACE calculator.")
    parser.add_argument("--charge", type=int, default=0, help="Total molecular charge.")
    parser.add_argument("--spin", type=int, default=1, help="Spin multiplicity-like spin setting expected by MACE-POLAR.")
    parser.add_argument(
        "--external-field",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("Ex", "Ey", "Ez"),
        help="External electric field vector.",
    )
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output text report path.")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    json_path = args.json.expanduser().resolve() if args.json else None

    atoms = read(str(input_path))

    calc = create_mace_calculator(model_path, args.device, args.dtype)

    atoms.info["charge"] = args.charge
    atoms.info["spin"] = args.spin
    atoms.info["external_field"] = list(args.external_field)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()

    results = calc.results
    dipole = results.get("dipole")
    charges = results.get("charges")
    density_coefficients = results.get("density_coefficients")
    spin_charge_density = results.get("spin_charge_density")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# MACE-POLAR calculation report\n")
        f.write(f"input_file: {input_path}\n")
        f.write(f"model_file: {model_path}\n")
        f.write(f"device: {args.device}\n")
        f.write(f"dtype: {args.dtype}\n")
        f.write(f"charge: {args.charge}\n")
        f.write(f"spin: {args.spin}\n")
        f.write(f"external_field: {list(args.external_field)}\n\n")

        f.write(f"potential_energy_eV: {energy:.12f}\n")
        f.write("forces_eV_per_A:\n")
        for idx, vec in enumerate(forces, start=1):
            f.write(f"  {idx:3d}: [{vec[0]: .12f}, {vec[1]: .12f}, {vec[2]: .12f}]\n")

        f.write("stress_eV_per_A3: ")
        f.write(np.array2string(stress, precision=12, separator=", "))
        f.write("\n\n")

        if dipole is not None:
            d = np.asarray(dipole)
            f.write(f"dipole: [{d[0]: .12f}, {d[1]: .12f}, {d[2]: .12f}]\n")
        else:
            f.write("dipole: None\n")

        if charges is not None:
            q = np.asarray(charges)
            f.write("charges_e:\n")
            for idx, qi in enumerate(q, start=1):
                f.write(f"  {idx:3d}: {float(qi): .12f}\n")
        else:
            f.write("charges_e: None\n")

        if density_coefficients is not None:
            rho = np.asarray(density_coefficients)
            f.write(f"density_coefficients_shape: {tuple(rho.shape)}\n")
        else:
            f.write("density_coefficients_shape: None\n")

        if spin_charge_density is not None:
            rho_s = np.asarray(spin_charge_density)
            f.write(f"spin_charge_density_shape: {tuple(rho_s.shape)}\n")
        else:
            f.write("spin_charge_density_shape: None\n")

    if json_path is not None:
        payload: Dict[str, Any] = {
            "input_file": str(input_path),
            "model_file": str(model_path),
            "device": args.device,
            "dtype": args.dtype,
            "charge": args.charge,
            "spin": args.spin,
            "external_field": list(args.external_field),
            "potential_energy_eV": float(energy),
            "forces_eV_per_A": to_serializable(forces),
            "stress_eV_per_A3": to_serializable(stress),
            "dipole": to_serializable(dipole) if dipole is not None else None,
            "charges_e": to_serializable(charges) if charges is not None else None,
            "density_coefficients_shape": list(np.asarray(density_coefficients).shape)
            if density_coefficients is not None
            else None,
            "spin_charge_density_shape": list(np.asarray(spin_charge_density).shape)
            if spin_charge_density is not None
            else None,
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(f"Report written to: {output_path}")
    if json_path is not None:
        print(f"JSON written to: {json_path}")


if __name__ == "__main__":
    main()
