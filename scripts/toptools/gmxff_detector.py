#!/usr/bin/env python3
"""Detect atom types for a PDB and emit a GROMACS ITP file.

The detector uses local 1-hop bonding environments from a connectivity graph:
- PDB CONECT records are used when available.
- If no CONECT data is present, geometric bond inference is used.

Known atom types are assigned by heuristic rules. If a known type cannot be
assigned, the atom receives a generated fallback type in the sequence
xa, xb, ..., xz, ya, ..., yz, za, ..., zz.
Atoms with identical 1-hop signatures share the same fallback type.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from string import ascii_lowercase
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ase.data import atomic_masses, atomic_numbers

try:
	from ..common.rot import load_atoms_and_adjacency
except ImportError:
	try:
		from scripts.common.importing import ensure_scripts_importable
	except ImportError:
		import sys

		repo_root = Path(__file__).resolve().parents[2]
		repo_root_str = str(repo_root)
		if repo_root_str not in sys.path:
			sys.path.insert(0, repo_root_str)
		sys.modules.pop("scripts", None)
		from scripts.common.importing import ensure_scripts_importable

	ensure_scripts_importable(__file__)
	from scripts.common.rot import load_atoms_and_adjacency

Adjacency = Dict[int, Set[int]]


@dataclass
class PdbAtomRecord:
	serial: int
	atom_name: str
	residue_name: str
	residue_id: int
	element: str


@dataclass
class DetectorSettings:
	ff_glob: str = "/apl/gromacs/2026.1/share/gromas/top/oplsaa.ff/*.itp"
	fallback_ff_globs: Tuple[str, ...] = ("/apl/gromacs/2026.1/share/gromacs/top/oplsaa.ff/*.itp",)
	require_type_in_ff: bool = False
	molecule_name: str = "MOL"
	nrexcl: int = 3
	default_charge: float = 0.0
	bond_funct: int = 1


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Detect atom types from PDB connectivity and write an ITP file.")
	parser.add_argument("--input", "-i", type=Path, required=True, help="Input PDB file path.")
	parser.add_argument("--output", "-o", type=Path, required=True, help="Output .itp file path.")
	parser.add_argument(
		"--settings",
		type=Path,
		default=None,
		help="Optional JSON settings path. If missing, internal defaults are used.",
	)
	parser.add_argument(
		"--report",
		type=Path,
		default=None,
		help="Optional text report with assignment diagnostics.",
	)
	return parser.parse_args()


def _normalize_element(raw: str) -> str:
	s = raw.strip()
	if not s:
		return ""
	if len(s) == 1:
		return s.upper()
	return s[0].upper() + s[1:].lower()


def read_settings(settings_path: Optional[Path]) -> DetectorSettings:
	if settings_path is None:
		return DetectorSettings()

	with settings_path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	fallback = data.get("fallback_ff_globs")
	if isinstance(fallback, list):
		fallback_globs = tuple(str(x) for x in fallback)
	elif fallback is None:
		fallback_globs = DetectorSettings.fallback_ff_globs
	else:
		fallback_globs = (str(fallback),)

	return DetectorSettings(
		ff_glob=str(data.get("ff_glob", DetectorSettings.ff_glob)),
		fallback_ff_globs=fallback_globs,
		require_type_in_ff=bool(data.get("require_type_in_ff", False)),
		molecule_name=str(data.get("molecule_name", "MOL")),
		nrexcl=int(data.get("nrexcl", 3)),
		default_charge=float(data.get("default_charge", 0.0)),
		bond_funct=int(data.get("bond_funct", 1)),
	)


def parse_pdb_atom_records(pdb_path: Path) -> List[PdbAtomRecord]:
	records: List[PdbAtomRecord] = []
	with pdb_path.open("r", encoding="utf-8") as handle:
		for line in handle:
			if not (line.startswith("ATOM") or line.startswith("HETATM")):
				continue
			serial = int(line[6:11].strip())
			atom_name = line[12:16].strip() or f"A{serial}"
			residue_name = line[17:20].strip() or "MOL"
			residue_id_raw = line[22:26].strip()
			residue_id = int(residue_id_raw) if residue_id_raw else 1
			element_raw = line[76:78].strip() if len(line) >= 78 else ""
			if not element_raw:
				# Fallback: infer element from atom name prefix.
				alpha = "".join(ch for ch in atom_name if ch.isalpha())
				if not alpha:
					element_raw = "X"
				elif len(alpha) >= 2 and alpha[1].islower():
					element_raw = alpha[:2]
				else:
					element_raw = alpha[:1]

			records.append(
				PdbAtomRecord(
					serial=serial,
					atom_name=atom_name,
					residue_name=residue_name,
					residue_id=residue_id,
					element=_normalize_element(element_raw),
				)
			)
	return records


def _iter_candidate_ff_paths(settings: DetectorSettings) -> Iterable[Path]:
	from glob import glob

	patterns = (settings.ff_glob,) + tuple(settings.fallback_ff_globs)
	seen: Set[Path] = set()
	for pattern in patterns:
		for raw in glob(pattern):
			path = Path(raw).expanduser().resolve()
			if path in seen or not path.is_file():
				continue
			seen.add(path)
			yield path


def load_ff_atom_types(settings: DetectorSettings) -> Set[str]:
	ff_types: Set[str] = set()
	for ff_path in _iter_candidate_ff_paths(settings):
		in_atomtypes = False
		try:
			with ff_path.open("r", encoding="utf-8") as handle:
				for raw_line in handle:
					line = raw_line.strip()
					if not line:
						continue
					if line.startswith(";"):
						continue
					if line.startswith("[") and line.endswith("]"):
						section = line.strip("[]").strip().lower()
						in_atomtypes = section == "atomtypes"
						continue
					if not in_atomtypes:
						continue
					fields = line.split()
					if not fields:
						continue
					if fields[0].startswith(";"):
						continue
					ff_types.add(fields[0])
		except OSError:
			continue
	return ff_types


def signature_for_atom(symbols: List[str], adjacency: Adjacency, idx: int) -> Tuple[str, Tuple[str, ...]]:
	center = symbols[idx]
	neigh = sorted(symbols[n] for n in adjacency[idx])
	return center, tuple(neigh)


def known_type_for_signature(signature: Tuple[str, Tuple[str, ...]]) -> Optional[str]:
	elem, neigh = signature
	counts = Counter(neigh)
	degree = len(neigh)

	if elem == "H" and degree == 1:
		host = neigh[0]
		return {
			"C": "HC",
			"O": "HO",
			"N": "HN",
			"S": "HS",
		}.get(host)

	if elem == "C":
		if degree == 4:
			if counts["C"] == 1 and counts["H"] == 3:
				return "CT"
			if counts["C"] == 2 and counts["H"] == 2:
				return "CT2"
			if counts["C"] == 3 and counts["H"] == 1:
				return "CT3"
			if counts["C"] == 4:
				return "CQ"
			if counts["O"] >= 1:
				return "CTO"
		if degree == 3:
			if counts["O"] >= 1:
				return "C"
			if counts["C"] >= 2:
				return "CA"

	if elem == "O":
		if degree == 2:
			if counts["C"] == 2:
				return "OS"
			if counts["C"] == 1 and counts["H"] == 1:
				return "OH"
		if degree == 1 and counts["C"] == 1:
			return "O"

	if elem == "N":
		if degree == 3:
			return "NT"
		if degree == 4:
			return "NQ"

	if elem == "S":
		if degree == 2:
			return "S"

	return None


def fallback_type_pool() -> List[str]:
	return [f"{first}{second}" for first in "xyz" for second in ascii_lowercase]


def assign_atom_types(symbols: List[str], adjacency: Adjacency, ff_types: Set[str], settings: DetectorSettings) -> Tuple[List[str], List[str], Dict[Tuple[str, Tuple[str, ...]], str]]:
	signatures = [signature_for_atom(symbols, adjacency, i) for i in range(len(symbols))]
	assigned: List[Optional[str]] = [None] * len(symbols)
	source: List[str] = [""] * len(symbols)

	for i, sig in enumerate(signatures):
		known = known_type_for_signature(sig)
		if known is None:
			continue
		if settings.require_type_in_ff and ff_types and known not in ff_types:
			continue
		assigned[i] = known
		source[i] = "rule"

	unknown_signatures = sorted({signatures[i] for i, t in enumerate(assigned) if t is None})
	pool = fallback_type_pool()
	if len(unknown_signatures) > len(pool):
		raise RuntimeError(
			f"Too many unknown signatures ({len(unknown_signatures)}). "
			f"Fallback pool has {len(pool)} types from xa..zz."
		)

	fallback_by_sig: Dict[Tuple[str, Tuple[str, ...]], str] = {
		sig: pool[idx] for idx, sig in enumerate(unknown_signatures)
	}

	for i, sig in enumerate(signatures):
		if assigned[i] is not None:
			continue
		assigned[i] = fallback_by_sig[sig]
		source[i] = "fallback"

	return [x for x in assigned if x is not None], source, fallback_by_sig


def mass_for_element(element: str) -> float:
	try:
		return float(atomic_masses[atomic_numbers[element]])
	except Exception:
		return 0.0


def unique_edges(adjacency: Adjacency) -> List[Tuple[int, int]]:
	edges: List[Tuple[int, int]] = []
	for i, neighs in adjacency.items():
		for j in neighs:
			if i < j:
				edges.append((i, j))
	edges.sort()
	return edges


def write_itp(
	output_path: Path,
	atom_records: List[PdbAtomRecord],
	atom_types: List[str],
	type_source: List[str],
	adjacency: Adjacency,
	settings: DetectorSettings,
) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)

	edges = unique_edges(adjacency)
	molecule_name = settings.molecule_name or atom_records[0].residue_name

	with output_path.open("w", encoding="utf-8") as f:
		f.write("; Generated by gmxff_detector.py\n")
		f.write("\n")
		f.write("[ moleculetype ]\n")
		f.write("; Name            nrexcl\n")
		f.write(f"{molecule_name:<16s} {settings.nrexcl:d}\n")
		f.write("\n")
		f.write("[ atoms ]\n")
		f.write("; nr  type   resnr residue atom   cgnr    charge      mass ; source\n")

		for i, rec in enumerate(atom_records):
			nr = i + 1
			atype = atom_types[i]
			resnr = rec.residue_id
			residue = rec.residue_name
			atom = rec.atom_name
			cgnr = resnr
			charge = settings.default_charge
			mass = mass_for_element(rec.element)
			src = type_source[i]
			f.write(
				f"{nr:5d} {atype:<6s} {resnr:5d} {residue:<6s} {atom:<6s} {cgnr:5d}"
				f" {charge:10.6f} {mass:10.4f} ; {src}\n"
			)

		f.write("\n")
		f.write("[ bonds ]\n")
		f.write(";  ai    aj funct\n")
		for i, j in edges:
			f.write(f"{i + 1:5d} {j + 1:5d} {settings.bond_funct:5d}\n")


def write_report(
	report_path: Path,
	atom_records: List[PdbAtomRecord],
	symbols: List[str],
	adjacency: Adjacency,
	atom_types: List[str],
	type_source: List[str],
	fallback_by_sig: Dict[Tuple[str, Tuple[str, ...]], str],
	ff_types: Set[str],
) -> None:
	report_path.parent.mkdir(parents=True, exist_ok=True)
	with report_path.open("w", encoding="utf-8") as f:
		f.write("# GMX FF detector report\n")
		f.write(f"atoms: {len(symbols)}\n")
		f.write(f"bonds: {len(unique_edges(adjacency))}\n")
		f.write(f"loaded_ff_types: {len(ff_types)}\n")
		f.write(f"fallback_signatures: {len(fallback_by_sig)}\n")
		f.write("\n")
		f.write("## atom_assignments\n")
		f.write("# idx element atom_name type source neighbors\n")
		for i, rec in enumerate(atom_records):
			neigh = ",".join(symbols[n] for n in sorted(adjacency[i]))
			f.write(
				f"{i + 1:4d} {symbols[i]:>2s} {rec.atom_name:<6s} {atom_types[i]:<6s}"
				f" {type_source[i]:<8s} [{neigh}]\n"
			)
		if fallback_by_sig:
			f.write("\n")
			f.write("## fallback_signature_map\n")
			for sig, ftype in sorted(fallback_by_sig.items(), key=lambda x: x[1]):
				center, neigh = sig
				neigh_s = ",".join(neigh)
				f.write(f"{ftype}: {center} -> [{neigh_s}]\n")


def main() -> None:
	args = parse_args()
	input_path = args.input.expanduser().resolve()
	output_path = args.output.expanduser().resolve()
	settings_path = args.settings.expanduser().resolve() if args.settings else None
	report_path = args.report.expanduser().resolve() if args.report else None

	if input_path.suffix.lower() != ".pdb":
		raise ValueError(f"Input must be a PDB file, got: {input_path}")

	settings = read_settings(settings_path)
	atom_records = parse_pdb_atom_records(input_path)
	atoms, adjacency = load_atoms_and_adjacency(input_path)
	symbols = [_normalize_element(s) for s in atoms.get_chemical_symbols()]

	if len(atom_records) != len(symbols):
		raise RuntimeError(
			"PDB atom-record count differs from ASE atom count: "
			f"{len(atom_records)} vs {len(symbols)}"
		)

	ff_types = load_ff_atom_types(settings)
	atom_types, type_source, fallback_by_sig = assign_atom_types(symbols, adjacency, ff_types, settings)
	write_itp(output_path, atom_records, atom_types, type_source, adjacency, settings)

	if report_path is not None:
		write_report(report_path, atom_records, symbols, adjacency, atom_types, type_source, fallback_by_sig, ff_types)

	print(f"ITP written to: {output_path}")
	if report_path is not None:
		print(f"Report written to: {report_path}")
	print(f"Loaded FF atom types: {len(ff_types)}")
	print(f"Fallback signature groups: {len(fallback_by_sig)}")


if __name__ == "__main__":
	main()