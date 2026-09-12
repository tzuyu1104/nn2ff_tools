"""Reusable potential-scan execution and output utilities.

This module deliberately does not import MACE or matplotlib at import time.  The
geometry code and scan orchestration can therefore be tested on CPU with a
small fake calculator, while command-line tools load optional dependencies only
when they actually need them.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from ase import Atoms


Transform = Callable[[float], tuple[Atoms, float]]


@dataclass(frozen=True)
class ScanResult:
    """Values and generated structures from one potential scan."""

    targets: np.ndarray
    realized: np.ndarray
    energies: np.ndarray
    snapshots: tuple[Atoms, ...]


def create_mace_calculator(model: Path, device: str, dtype: str):
    """Construct a MACE-POLAR calculator while keeping MACE optional."""
    try:
        from mace.calculators import mace_polar
    except ImportError as exc:
        raise RuntimeError(
            "MACE-POLAR support is not installed. Install this project with "
            "the 'mace' extra or install the project-specific MACE package."
        ) from exc

    return mace_polar(model=str(model), device=device, default_dtype=dtype)


def evaluate_scan(
    targets: Iterable[float],
    transform: Transform,
    calculator,
    *,
    charge: int,
    spin: int,
    field: Sequence[float],
) -> ScanResult:
    """Transform and evaluate structures for each requested scan value."""
    target_values = np.asarray(list(targets), dtype=float)
    if target_values.size == 0:
        raise ValueError("A potential scan requires at least one target value.")

    realized_values: list[float] = []
    energies: list[float] = []
    snapshots: list[Atoms] = []

    for target in target_values:
        atoms, realized = transform(float(target))
        atoms.info["charge"] = charge
        atoms.info["spin"] = spin
        atoms.info["external_field"] = list(field)
        atoms.calc = calculator

        realized_values.append(float(realized))
        energies.append(float(atoms.get_potential_energy()))
        snapshots.append(atoms)

    return ScanResult(
        targets=target_values,
        realized=np.asarray(realized_values, dtype=float),
        energies=np.asarray(energies, dtype=float),
        snapshots=tuple(snapshots),
    )


def write_scan_csv(
    path: Path,
    result: ScanResult,
    *,
    target_column: str,
    realized_column: str | None = None,
    parameter: str | None = None,
) -> None:
    """Write a scan result using the existing command-line CSV formats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if parameter is not None:
            writer.writerow(["parameter", target_column, "potential_energy_eV"])
            for target, energy in zip(result.targets, result.energies):
                writer.writerow([parameter, f"{target:.10f}", f"{energy:.12f}"])
        else:
            if realized_column is None:
                raise ValueError("realized_column is required for geometric scans.")
            writer.writerow([target_column, realized_column, "potential_energy_eV"])
            for target, realized, energy in zip(
                result.targets, result.realized, result.energies
            ):
                writer.writerow(
                    [f"{target:.10f}", f"{realized:.10f}", f"{energy:.12f}"]
                )


def write_scan_plot(
    path: Path,
    result: ScanResult,
    *,
    xlabel: str,
    title: str,
) -> None:
    """Write the standard scan plot, importing matplotlib only when requested."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plot output requires matplotlib. Install this project with the 'plot' extra."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(7, 5))
    axes.plot(result.targets, result.energies, marker="o", ms=3)
    axes.set_xlabel(xlabel)
    axes.set_ylabel("Potential Energy (eV)")
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_snapshots(path: Path, result: ScanResult) -> None:
    """Write generated structures as an OVITO-compatible multi-frame PDB."""
    from ase.io import write

    path.parent.mkdir(parents=True, exist_ok=True)
    write(str(path), list(result.snapshots), format="proteindatabank")
