# SCIBS 

**SCIBS** is a Python migration of the original SCIBS (C-based) codebase, designed to process 3D meshes (tetrahedrons and triangles), map electrode boundaries, and provide seamless MATLAB interoperability.

## Features

- **Python-Native:** Migrated to Python (>=3.12) for improved maintainability.
- **MATLAB Interoperability:** Read and write mesh geometries directly to/from MATLAB `.mat` structs.
- **CLI Tooling:** Exposes a unified command-line interface for individual mesh operations or full pipeline execution.
- **Visualization:** Integrated 3D mesh rendering using `pyvista`.

## Installation

This project uses standard Python packaging. It requires Python 3.12 or newer.

To install the package and its dependencies (`matplotlib`, `pyvista`, `scipy`):

```bash
# Clone the repository
git clone <your-repo-url>
cd scibs

# Install the package
pip install .
```

If you plan on running tests or contributing to the codebase, install the development dependencies:

```bash
pip install .[dev]
```

## Usage

Installing the package exposes the `scibs` command in your terminal. The CLI provides access to several subcommands for mesh processing.

### Full Pipeline
To run the complete processing pipeline (`tet2tri` $\rightarrow$ `trielec` $\rightarrow$ `tetwarp` $\rightarrow$ `tetcor`):

```bash
python -m scibs pipeline <file_prefix> [-s STRUCT_NAME]
```
* **`file_prefix`**: The prefix of the MATLAB file to run the pipeline on (e.g., if the file is `mesh.mat`, use `mesh`).
* **`-s`, `--struct-name`**: The MATLAB struct name to read from (defaults to `Geometry`).

### Individual Commands

You can also run individual steps of the pipeline standalone:

#### 1. `tet2tri`
Extracts surface triangles from a tetrahedral mesh.

```bash
python -m scibs tet2tri [-s] pts_name [tet_name] [tri_name] [el_name] [tot_name]
```
* **`-s`, `--surface-only`**: Output surface triangles only.

#### 2. `trielec`
Subdivides vertices of a triangulated surface to map electrode boundaries.

```bash
python -m scibs trielec <tri_in> <elec_descr_in> <tri_out> [--plot] [-w WARP_NAME]
```
* **`tri_in`**: Input triangulated surface file (`.tri`).
* **`elec_descr_in`**: Electrode description text file.
* **`tri_out`**: Output triangulated surface file (`.tri`).
* **`--plot`**: Render a 3D visualization of the subdivided mesh.
* **`-w`, `--warp_name`**: Record changes in a warp file (legacy flag).

#### 3. `tetcor`
Corrects tetrahedron volumes in a MATLAB file.

```bash
python -m scibs tetcor <file_name> [-s STRUCT_NAME] [-o OUTPUT]
```
* **`file_name`**: The MATLAB tetrahedron file to read from (`.mat`).
* **`-s`, `--struct-name`**: MATLAB struct name (e.g., `HeadModel`, `Geometry`).
* **`-o`, `--output`**: Output file to write to. Defaults to `<file_name>_corrected.mat`.

## Project Structure

* `src/scibs/`: Main Python package source code.
  * `scripts/`: Implementations of the core pipeline steps (`tet2tri`, `trielec`, `tetwarp`, `tetcor`, `pipeline`).
  * `utilities/`: Helper modules for file I/O (`mat`, `electrode_descr`, etc.) and graphics.
* `tests/`: `pytest` suite (`test_pot.py`, `test_tetcor.py`).

## Author

* **Blake Moody** - [blake.moody001@umb.edu](mailto:blake.moody001@umb.edu)
```