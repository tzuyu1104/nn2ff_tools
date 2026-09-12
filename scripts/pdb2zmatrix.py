#!/usr/bin/env python3
"""Convert a PDB structure to a simple Z-matrix text format.

Usage:
    python scripts/pdb2zmatrix.py \
        --input samples/nBME_opt.pdb \
        --output samples/nBME_opt.zmat
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import pdb2zmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PDB to Z-matrix format.")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to input PDB file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to output Z-matrix file. Defaults to input name with .zmat extension.",
    )
    parser.add_argument(
        "--format",
        choices=["plain", "gjf"],
        default="plain",
        help="Output format: plain Z-matrix text or Gaussian GJF Z-matrix.",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Molecular charge for GJF output.",
    )
    parser.add_argument(
        "--multiplicity",
        type=int,
        default=1,
        help="Spin multiplicity for GJF output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if args.output is None:
        output_path = input_path.with_suffix(".zmat" if args.format == "plain" else ".gjf")
    else:
        output_path = args.output.resolve()

    pdb2zmat.convert_pdb_to_zmatrix(
        input_path,
        output_path,
        output_format=args.format,
        charge=args.charge,
        multiplicity=args.multiplicity,
    )
    print(f"Z-matrix written to: {output_path}")


if __name__ == "__main__":
    main()
