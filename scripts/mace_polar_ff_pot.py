#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms
from mace.calculators import mace_polar

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise RuntimeError(
        "matplotlib is required. Install in your venv with:\n"
        "pip install matplotlib"
    ) from exc


class ZRow:
    def __init__(
        self,
        symbol: str,
        b: Optional[int] = None,
        r_token: Optional[str] = None,
        a: Optional[int] = None,
        ang_token: Optional[str] = None,
        d: Optional[int] = None,
        dih_token: Optional[str] = None,
    ):
        self.symbol = symbol
        self.b = b
        self.r_token = r_token
        self.a = a
        self.ang_token = ang_token
        self.d = d
        self.dih_token = dih_token


def is_number_token(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def resolve_token(tok: str, variables: Dict[str, float]) -> float:
    if is_number_token(tok):
        return float(tok)
    if tok not in variables:
        raise ValueError(f"Variable {tok} not found in GJF variable block.")
    return float(variables[tok])


def parse_gjf_zmatrix(gjf_path: Path) -> Tuple[int, int, List[ZRow], Dict[str, float]]:
    lines = gjf_path.read_text(encoding="utf-8").splitlines()

    idx = 0
    n = len(lines)

    # Skip link0 and route lines and leading blanks.
    while idx < n and (
        lines[idx].strip().startswith("%")
        or lines[idx].strip().startswith("#")
        or lines[idx].strip() == ""
    ):
        idx += 1

    # Title block until blank line.
    while idx < n and lines[idx].strip() != "":
        idx += 1

    # Blank after title.
    while idx < n and lines[idx].strip() == "":
        idx += 1

    if idx >= n:
        raise ValueError("Could not find charge/multiplicity line.")

    charge_mult = lines[idx].split()
    if len(charge_mult) < 2:
        raise ValueError("Invalid charge/multiplicity line.")
    charge = int(charge_mult[0])
    multiplicity = int(charge_mult[1])
    idx += 1

    z_lines: List[str] = []
    while idx < n and lines[idx].strip() != "":
        z_lines.append(lines[idx].strip())
        idx += 1

    while idx < n and lines[idx].strip() == "":
        idx += 1

    var_lines: List[str] = []
    while idx < n and lines[idx].strip() != "":
        var_lines.append(lines[idx].strip())
        idx += 1

    rows: List[ZRow] = []
    for i, zl in enumerate(z_lines):
        t = zl.split()
        if len(t) == 1:
            rows.append(ZRow(symbol=t[0]))
        elif len(t) == 3:
            rows.append(ZRow(symbol=t[0], b=int(t[1]), r_token=t[2]))
        elif len(t) == 5:
            rows.append(
                ZRow(symbol=t[0], b=int(t[1]), r_token=t[2], a=int(t[3]), ang_token=t[4])
            )
        elif len(t) == 7:
            rows.append(
                ZRow(
                    symbol=t[0],
                    b=int(t[1]),
                    r_token=t[2],
                    a=int(t[3]),
                    ang_token=t[4],
                    d=int(t[5]),
                    dih_token=t[6],
                )
            )
        else:
            raise ValueError(f"Unsupported Z-matrix line format at row {i+1}: {zl}")

    variables: Dict[str, float] = {}
    for vl in var_lines:
        m = re.match(r"^\s*([A-Za-z]\w*)\s*=\s*([-+0-9.eE]+)\s*$", vl)
        if not m:
            continue
        variables[m.group(1)] = float(m.group(2))

    return charge, multiplicity, rows, variables


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-14:
        raise ValueError("Encountered near-zero vector in coordinate construction.")
    return v / n


def place_atom(
    pb: np.ndarray,
    pa: np.ndarray,
    pd: np.ndarray,
    r: float,
    ang_deg: float,
    dih_deg: float,
) -> np.ndarray:
    theta = math.radians(ang_deg)
    phi = math.radians(dih_deg)

    ez = unit(pb - pa)
    v = pa - pd
    en = np.cross(v, ez)
    if np.linalg.norm(en) < 1e-10:
        trial = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(trial, ez)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        en = np.cross(trial, ez)
    ey = unit(en)
    ex = unit(np.cross(ey, ez))

    v_local = -math.cos(theta) * ez + math.sin(theta) * (
        math.cos(phi) * ex + math.sin(phi) * ey
    )
    return pb + r * v_local


def zmat_to_atoms(rows: List[ZRow], variables: Dict[str, float]) -> Atoms:
    symbols: List[str] = []
    coords: List[np.ndarray] = []

    for i, row in enumerate(rows):
        symbols.append(row.symbol)

        if i == 0:
            coords.append(np.array([0.0, 0.0, 0.0], dtype=float))
            continue

        if i == 1:
            r = resolve_token(row.r_token, variables)
            coords.append(np.array([0.0, 0.0, r], dtype=float))
            continue

        if i == 2:
            r = resolve_token(row.r_token, variables)
            ang = resolve_token(row.ang_token, variables)
            b = row.b - 1
            a = row.a - 1

            pb = coords[b]
            pa = coords[a]
            pd = pa + np.array([1.0, 0.0, 0.0], dtype=float)
            p = place_atom(pb, pa, pd, r, ang, 0.0)
            coords.append(p)
            continue

        r = resolve_token(row.r_token, variables)
        ang = resolve_token(row.ang_token, variables)
        dih = resolve_token(row.dih_token, variables)
        b = row.b - 1
        a = row.a - 1
        d = row.d - 1

        pb = coords[b]
        pa = coords[a]
        pd = coords[d]
        p = place_atom(pb, pa, pd, r, ang, dih)
        coords.append(p)

    pos = np.vstack(coords)
    return Atoms(symbols=symbols, positions=pos)


def default_scan_range(param_name: str, center: float) -> Tuple[float, float]:
    lead = param_name[0].upper()
    if lead == "R":
        delta = max(0.02, abs(center) * 0.10)
        return center - delta, center + delta
    if lead in ("A", "D"):
        return center - 30.0, center + 30.0
    raise ValueError("Parameter name must start with R, A, or D.")


def parse_specifier(specifier: str) -> List[int]:
    parts = [p.strip() for p in specifier.split("-") if p.strip()]
    if len(parts) not in (2, 3, 4):
        raise ValueError(
            f"Invalid specifier '{specifier}'. Use 'i-j' (bond), 'i-j-k' (angle), or 'i-j-k-l' (dihedral)."
        )

    try:
        idx = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid specifier '{specifier}': atom indices must be integers.") from exc

    if any(i <= 0 for i in idx):
        raise ValueError(f"Invalid specifier '{specifier}': atom indices must be >= 1.")
    return idx


def resolve_parameter_name(param_or_specifier: str, rows: List[ZRow], variables: Dict[str, float]) -> Tuple[str, str]:
    """Resolve user input to a Z-matrix variable name.

    Returns (resolved_parameter_name, mode) where mode is one of: param|bond|angle|dihedral.
    """
    key = param_or_specifier.strip()
    if key in variables:
        return key, "param"

    idx = parse_specifier(key)
    n = len(idx)
    candidates: List[str] = []

    for atom_i, row in enumerate(rows, start=1):
        if n == 2 and row.r_token is not None:
            pair = (atom_i, row.b)
            if pair == (idx[0], idx[1]) or pair == (idx[1], idx[0]):
                if row.r_token in variables:
                    candidates.append(row.r_token)

        elif n == 3 and row.ang_token is not None:
            triple = (atom_i, row.b, row.a)
            if triple == (idx[0], idx[1], idx[2]) or triple == (idx[2], idx[1], idx[0]):
                if row.ang_token in variables:
                    candidates.append(row.ang_token)

        elif n == 4 and row.dih_token is not None:
            quad = (atom_i, row.b, row.a, row.d)
            rev = (row.d, row.a, row.b, atom_i)
            if quad == tuple(idx) or rev == tuple(idx):
                if row.dih_token in variables:
                    candidates.append(row.dih_token)

    # Fallback for 4-atom impropers: match by atom set if unique.
    if n == 4 and not candidates:
        set_target = set(idx)
        set_candidates: List[str] = []
        for atom_i, row in enumerate(rows, start=1):
            if row.dih_token is None or row.dih_token not in variables:
                continue
            set_quad = {atom_i, row.b, row.a, row.d}
            if set_quad == set_target:
                set_candidates.append(row.dih_token)
        candidates = set_candidates

    unique = sorted(set(candidates))
    if len(unique) == 1:
        mode = {2: "bond", 3: "angle", 4: "dihedral"}[n]
        return unique[0], mode

    if len(unique) > 1:
        raise ValueError(
            f"Specifier '{key}' matches multiple parameters: {', '.join(unique)}. "
            "Use --param with an explicit variable name."
        )

    raise ValueError(
        f"Could not resolve '{key}' to a scan variable. "
        "Use a valid parameter (e.g. D7) or a valid specifier (e.g. 1-2, 3-4-5, 6-7-8-10)."
    )


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

    charge, _multiplicity, rows, variables = parse_gjf_zmatrix(input_path)

    param, param_mode = resolve_parameter_name(args.param, rows, variables)

    npoints = args.bins if args.bins is not None else args.npoints
    if npoints <= 0:
        raise ValueError("Number of scan points/bins must be > 0.")

    center = float(variables[param])
    if args.range is not None:
        vmin, vmax = float(args.range[0]), float(args.range[1])
    elif args.vmin is None or args.vmax is None:
        vmin, vmax = default_scan_range(param, center)
    else:
        vmin, vmax = args.vmin, args.vmax

    values = np.linspace(vmin, vmax, npoints)

    stem = input_path.stem + f"_{param}_scan"
    csv_out = args.csv_out.expanduser().resolve() if args.csv_out else input_path.with_name(stem + ".csv")
    fig_out = args.fig_out.expanduser().resolve() if args.fig_out else input_path.with_name(stem + ".png")

    calc = mace_polar(
        model=str(model_path),
        device=args.device,
        default_dtype=args.dtype,
    )

    energies = []
    for v in values:
        local_vars = dict(variables)
        local_vars[param] = float(v)

        atoms = zmat_to_atoms(rows, local_vars)
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
