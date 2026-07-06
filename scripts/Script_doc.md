# Script Documentation

This document summarizes the Python scripts and libraries in the `scripts` directory, with usage examples, I/O definitions, and recommended workflows.

## Scope

Current files documented:

- `pdb2zmatrix.py`
- `mace_polar_calc.py`
- `mace_polar_ff_pot.py`
- `rot_fragment_lib.py`
- `mace_polar_pot_rot.py`
- `zmat_to_pdb_compare.py`

---

## 1) pdb2zmatrix.py

Purpose:

- Read a PDB structure.
- Build connectivity-aware Z-matrix.
- Export either plain Z-matrix text or Gaussian GJF Z-matrix with symbolic variables (R/A/D).

Main features:

- Parses PDB CONECT records.
- Uses connectivity-first ordering.
- Writes Gaussian-compatible Z-matrix format.

CLI:

```bash
python scripts/pdb2zmatrix.py \
  --input samples/nBME_opt.pdb \
  --output samples/nBME_opt_zmat.gjf \
  --format gjf \
  --charge 0 \
  --multiplicity 1
```

Arguments:

- `--input/-i`: input PDB path (required)
- `--output/-o`: output file path (optional; default by format)
- `--format`: `plain` or `gjf` (default `plain`)
- `--charge`: integer charge for GJF header
- `--multiplicity`: spin multiplicity for GJF header

Outputs:

- Plain mode: `.zmat` text file.
- GJF mode: Gaussian input with Z-matrix body and variable block.

---

## 2) mace_polar_calc.py

Purpose:

- Run a single-point MACE-POLAR calculation on a structure (PDB or any ASE-readable format).
- Export energy, forces, stress, dipole, charges, and selected tensor shapes.

Main features:

- Uses local model file path (no download required).
- Supports device and dtype control.
- Writes text report and optional JSON report.

CLI:

```bash
python scripts/mace_polar_calc.py \
  --input samples/nBME_opt.pdb \
  --model ~/workspace/mace_polar_1/MACE-POLAR-1-M.model \
  --device cuda \
  --dtype float64 \
  --charge 0 \
  --spin 1 \
  --external-field 0.0 0.0 0.0 \
  --output samples/nBME_opt_mace_polar_results.txt \
  --json samples/nBME_opt_mace_polar_results.json
```

Arguments:

- `--input/-i`: input structure (required)
- `--model`: local MACE-POLAR model path (required)
- `--device`: `cpu` or `cuda`
- `--dtype`: `float32` or `float64`
- `--charge`: total molecular charge
- `--spin`: spin flag/info for model
- `--external-field`: `Ex Ey Ez`
- `--output/-o`: text report path (required)
- `--json`: optional JSON report path

Outputs:

- Human-readable text report.
- Optional structured JSON report.

---

## 3) mace_polar_ff_pot.py

Purpose:

- Scan one internal coordinate and evaluate MACE-POLAR potential over a range.
- Generate CSV and a matplotlib plot.

Main features:

- Input is Gaussian Z-matrix GJF.
- Supports parameter by name (`R1`, `A1`, `D7`) or by atom-index specifier:
  - bond: `i-j`
  - angle: `i-j-k`
  - dihedral/improper: `i-j-k-l`
- Resolves specifier to the corresponding Z-matrix variable.

Example 1 (variable name):

```bash
python scripts/mace_polar_ff_pot.py \
  --input samples/nBME_opt_zmat.gjf \
  --param D7 \
  --range 0 360 \
  --bins 36 \
  --model ~/workspace/mace_polar_1/MACE-POLAR-1-M.model \
  --device cuda \
  --csv-out samples/nBME_opt_zmat_D7_0_360_scan.csv \
  --fig-out samples/nBME_opt_zmat_D7_0_360_scan.png
```

Example 2 (atom specifier):

```bash
python scripts/mace_polar_ff_pot.py \
  --input samples/nBME_opt_zmat.gjf \
  --param 10-6-2-1 \
  --range 0 360 \
  --bins 36 \
  --model ~/workspace/mace_polar_1/MACE-POLAR-1-M.model \
  --device cuda \
  --csv-out samples/nBME_opt_zmat_10-6-2-1_0_360_scan.csv \
  --fig-out samples/nBME_opt_zmat_10-6-2-1_0_360_scan.png
```

Arguments:

- `--input/-i`: input GJF
- `--param/-p`: variable name or index specifier
- `--npoints`: number of points (legacy)
- `--bins`: alias for number of points (preferred)
- `--range MIN MAX`: explicit scan range
- `--vmin/--vmax`: alternative range specification
- `--model`, `--device`, `--dtype`, `--spin`, `--field`: calculator options
- `--csv-out`, `--fig-out`: output paths

Outputs:

- CSV: parameter/value/energy table.
- PNG: energy profile.

---

## 4) rot_fragment_lib.py (library)

Purpose:

- Shared geometry and connectivity utilities for fragment-based rotation workflows.

Core capabilities:

- PDB CONECT parsing.
- Geometry-based bond inference fallback.
- DFS component search with blocked edge/node support.
- Fragment split helpers:
  - `split_by_bond`
  - `split_by_angle`
  - `split_by_dihedral`
- Gaussian connectivity GJF writer (Cartesian + `geom=connectivity`).
- GJF Z-matrix parsing and conversion to Cartesian coordinates.
- Adjacency build from Z-matrix rows.
- Rigid-fragment rotation around axis and dihedral setter.

Typical use:

- Import this module from command-line tools (for example `mace_polar_pot_rot.py`).

---

## 5) mace_polar_pot_rot.py

Purpose:

- Fragment-based rotational toolset with CLI subcommands.
- Supports generation of rotated structures and MACE-POLAR potential scans using rigid-fragment rotation rule.

Subcommands:

### 5.1 write-gjf

Write Cartesian Gaussian input with explicit connectivity.

```bash
python scripts/mace_polar_pot_rot.py write-gjf \
  --input samples/nBME_opt.pdb \
  --output samples/nBME_opt_connectivity.gjf
```

### 5.2 split

Print fragment decomposition by specifier (`i-j`, `i-j-k`, `i-j-k-l`).

```bash
python scripts/mace_polar_pot_rot.py split \
  --input samples/nBME_opt.pdb \
  --spec 10-7-1-2
```

### 5.3 generate-dihedral-series

Generate rigid-fragment rotated PDB frames for a dihedral.

```bash
python scripts/mace_polar_pot_rot.py generate-dihedral-series \
  --input samples/nBME_opt_zmat.gjf \
  --spec 10-6-2-1 \
  --outdir samples/nBME_D6 \
  --start 0 --stop 330 --step 30 \
  --prefix nBME_rotD6_
```

### 5.4 scan-dihedral-potential

Run MACE-POLAR potential scan using rigid-fragment rotation (recommended for coupled torsion effects).

```bash
python scripts/mace_polar_pot_rot.py scan-dihedral-potential \
  --input samples/nBME_opt_zmat.gjf \
  --spec 10-6-2-1 \
  --model ~/workspace/mace_polar_1/MACE-POLAR-1-M.model \
  --device cuda --dtype float64 \
  --charge 0 --spin 1 --field 0 0 0 \
  --start 0 --stop 360 --bins 36 \
  --csv-out samples/nBME_opt_rotRule_10-6-2-1_0_360_scan.csv \
  --fig-out samples/nBME_opt_rotRule_10-6-2-1_0_360_scan.png
```

---

## 6) zmat_to_pdb_compare.py

Purpose:

- Convert Gaussian Z-matrix GJF to Cartesian PDB.
- Align against reference PDB (Kabsch alignment).
- Report RMSD and max per-atom deviation.

CLI:

```bash
python scripts/zmat_to_pdb_compare.py \
  --gjf samples/nBME_opt_zmat.gjf \
  --reference samples/nBME_opt.pdb \
  --output-pdb samples/nBME_opt_trans2.pdb \
  --aligned-pdb samples/nBME_opt_trans2_aligned.pdb \
  --report samples/nBME_opt_trans2_compare.txt
```

Outputs:

- Raw transformed PDB.
- Aligned PDB.
- Comparison text report with RMSD metrics.

---

## Recommended Workflow for Current nBME Project

1. Build/update Z-matrix:

```bash
python scripts/pdb2zmatrix.py --input samples/nBME_opt.pdb --format gjf --output samples/nBME_opt_zmat.gjf
```

2. Verify Z-matrix geometry reconstruction quality (optional):

```bash
python scripts/zmat_to_pdb_compare.py --gjf samples/nBME_opt_zmat.gjf --reference samples/nBME_opt.pdb
```

3. For physically meaningful torsion scans, use rigid-fragment scan:

```bash
python scripts/mace_polar_pot_rot.py scan-dihedral-potential \
  --input samples/nBME_opt_zmat.gjf --spec 10-6-2-1 --start 0 --stop 360 --bins 36 \
  --model ~/workspace/mace_polar_1/MACE-POLAR-1-M.model --device cuda \
  --csv-out samples/nBME_opt_rotRule_10-6-2-1_0_360_scan.csv \
  --fig-out samples/nBME_opt_rotRule_10-6-2-1_0_360_scan.png
```

4. If needed, export explicit frame geometries:

```bash
python scripts/mace_polar_pot_rot.py generate-dihedral-series \
  --input samples/nBME_opt_zmat.gjf --spec 10-6-2-1 --outdir samples/nBME_D6
```

---

## Environment Notes

- Activate environment before running scripts:

```bash
source ~/workspace/mace_polar_1/mlplr-venv/bin/activate
```

- Required package families:
  - ASE
  - numpy
  - matplotlib (for plotting)
  - MACE package with `mace.calculators.mace_polar`

---

## Limitations and Current Assumptions

- Fragment split logic is designed first for acyclic/non-ring use cases.
- Ring systems usually need constrained full-molecule protocols, not simple single-bond rigid cuts.
- Index specifiers are 1-based.
- For some workflows, indexing in GJF-derived structures and original PDB may differ; always confirm intended atom mapping.
