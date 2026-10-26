# AGENT.md — AstroHACK

Guidance for AI agents (and humans) working on **AstroHACK** (Holography and
Antenna Commissioning Kit). Read this before making changes; it captures the
architecture, where the external-data (casacore) boundary lives, how to run the
tests, and the non-obvious gotchas.

---

## 1. What this package is

AstroHACK is a pure-Python package for **NRAO telescope support**: radio
holography (VLA / ALMA, with basic near-field ALMA and emerging ngVLA support)
and **antenna-position corrections** (locit). It ports the AIPS holography tasks
(UVHOL / HOLOG / PANEL) onto a modern **Dask + Numba** stack so reductions
parallelize and run fast.

Inputs are **MeasurementSet v2** (`.ms`) and CASA **calibration tables**;
outputs are self-describing **`.zarr`** datatrees wrapped by the package's own
"mds" file classes (see §2). Coordinate math uses **astropy**; gridding/fitting
use numpy/scipy/scikit-image/numba.

---

## 2. Architecture (three layers + helpers)

```
src/astrohack/
├── __init__.py                # exports the user-facing pipeline functions
├── <task>.py                  # USER-FACING api: param validation + dask graph build
│     extract_pointing, extract_holog, extract_alma_nf_holog, holog, panel,
│     simulate_alma_nf_holog, estimate_nf_distance,
│     locit, extract_locit, combine, beamcut, cassegrain_ray_tracing,
│     image_comparison_tool
├── core/                      # COMPUTE: the per-chunk work the dask graph runs
│     extract_pointing.py, extract_holog.py, extract_locit.py,  ← casacore lives here
│     holog.py, panel.py, combine.py, beamcut.py, ...
├── io/                        # the "*_mds" file classes (zarr datatree wrappers)
│     dio.py, base_mds.py, holog_mds.py, point_mds.py, image_mds.py,
│     locit_mds.py, panel_mds.py, position_mds.py, beamcut_mds.py
├── antenna/                   # telescope + panel geometry/fitting classes
├── utils/                     # algorithms, gridding, conversion, fits, constants, ...
├── visualization/             # plot tools, dashboards
└── config/                    # parameter-json scaffolding
```

A typical pipeline call (e.g. `extract_holog`) lives in the top-level module,
validates parameters, then maps a `core/`-module chunk function over a Dask
graph (via `toolviper`/`graphviper`). The `core/` functions are where the MS is
actually read.

**The usual holography flow:** `extract_pointing` → `extract_holog` → `holog`
(grid + FFT to aperture) → `panel` (panel fitting). The position flow is
`extract_locit` → `locit`.

---

## 3. The casacore boundary — **arcae**, through xradio

**MS / cal-table reading is the ONLY use of casacore in the package, and it is
confined to five files:**

- `src/astrohack/core/extract_holog.py`
- `src/astrohack/core/extract_pointing.py`
- `src/astrohack/core/extract_locit.py`
- `src/astrohack/core/fringefit_locit.py`
- `src/astrohack/scripts/baseline/baseline_reduction_pipeline.py` (import
  inside `make_all_antenna_phases_plot`)

Each does `from xradio._utils._casacore.backend import tables as ctables` and
uses only:
- `ctables.table(path, readonly=True, lockoptions={"option": "usernoread"}, ack=False)`
- `ctables.taql("select ... from $table_obj where ...")` (`$var` substitution)
- `.getcol(col[, startrow=, nrow=])`, `.close()`
- the `path::SUBTABLE` and `path/SUBTABLE` ways of opening MS subtables.

### Why arcae, and why through xradio

[arcae](https://github.com/ska-sa/arcae) ships self-contained wheels for Linux
and macOS (cp311–cp314) with casacore linked in statically and none of its
symbols exported, so it coexists in-process with CASA's `casatools`. Checked
on macOS arm64 and Linux x86_64 with casatools 6.7.6: both import orders,
reads identical to casatools'.

xradio's private `xradio._utils._casacore` package emulates the python-casacore
`tables` API on arcae (`casacore_from_arcae.py`; `backend.py` picks it, env
`XRADIO_CASACORE_BACKEND`, default `arcae`). astrohack depends on
`xradio[casacore]>=1.3.0`, the first release with arcae, and imports that
private module, so changes to it in xradio can break astrohack. Coordinate
conversions are done with **astropy** (`utils/conversion.py`, `utils/fits.py`),
not casacore measures.

Where the shim differs from python-casacore:
- string columns come back as numpy arrays (see §6);
- `getcol` of an array column on an empty selection raises `RuntimeError`
  instead of returning an empty array.

`utils/constants.py` has a casacore **URL in a comment** only — not an import.

### CASA-side example scripts (left untouched on purpose)

`etc/locit/casa/pre-locit-script.py` and `beamcut_CASA_calibration.py` use
`import casatools`; `etc/locit/casa/post-locit-plot-phases.py` uses
`from casacore import tables`. These are **standalone helper scripts meant to be
run inside a CASA session** (where casatools / python-casacore exist), not part
of the installed package or the test suite. Leave them on casatools/casacore.

### ASDM reading — pyasdm

`extract_alma_nf_holog` converts ALMA near field holography ASDMs; the ASDM is
read with [pyasdm](https://github.com/casangi/pyasdm) only in
`core/extract_alma_nf_holog.py` (Holography, SpectralWindow, Antenna, Field,
Flag, Pointing and TotalPower tables).

- pyasdm (0.0.7, 0.0.8) converts the midpoint of a **binary** ArrayTimeInterval
  to its start as `(midpoint - duration) / 2`, which halves the Pointing times
  (casangi/pyasdm#46). `_get_pnt_from_asdm` therefore reads the Pointing table
  with `ArrayTimeInterval.setReadStartTimeDurationInBin(True)` (restored
  afterwards) and subtracts half the duration itself, unless pyasdm's flag says
  the ASDM stores start times (`startTimeDurationInBin` in ASDM.xml; the flag
  is global and never reset, casangi/pyasdm#49). The XML tables are not
  affected.
- The output is one antenna, DDI and map with `near_field=True` and the
  polarization axis `["I", "R2"]`: `holog` needs `to_stokes=False`.
- The unit tests download the ASDM `uid___A002_Xad52b4_X3`, which is not in
  the ToolVIPER registry yet (casangi/toolviper#80). To run them before that,
  copy the ASDM into the test's data folder: `download()` skips existing files.

---

## 4. Build / install / environment

**Use the conda `zinc` environment** (`conda activate zinc`) — it has
xradio, arcae and the scientific stack installed.

```bash
conda activate zinc
cd astrohack
pip install -e .          # setuptools src-layout; deps incl. xradio[casacore] (arcae), toolviper, dask, numba, astropy, zarr>=3
```

Packaging notes / pre-existing quirks (not introduced by the casacore work):
- No `[build-system]` table in `pyproject.toml` → setuptools default backend.
- `python` 3.12–3.14 (`requires-python >= 3.12, < 3.15`, issue #388).
- zarr-python 3 (`zarr>=3.0.0`, issue #300); xarray and numpy are not pinned
  (numba sets its own supported numpy range).

**Zarr formats.** Every function that writes a zarr file takes
`zarr_format` (2 or 3, default 3). Format 2 files — the downloaded unit-test
reference data and the packaged telescopes in `src/astrohack/data/` — are read
as before. An mds object keeps the format of its file: `open()` reads it from
the store and `create_from_input_parameters(..., zarr_format)` sets it, and
`write`, `add_node` and `consolidate` use it, so a store never mixes formats
(e.g. `extract_holog(append=True)` on a format 2 file stays format 2).
Consolidated metadata is written for both formats; `astrohack/__init__.py`
filters zarr's warning that format 3 does not specify it yet.

For quick iteration without installing, `PYTHONPATH=src python -m pytest …`
works (the package imports cleanly from `src/`).

**Lint/format: pre-commit.** `.pre-commit-config.yaml` is xradio's: hygiene
checks, pyupgrade (`--py311-plus`; v2.38.2 has no 3.12 flag, ruff's `UP` rules
cover it), absolufy-imports, and ruff 0.16.10 check + format (config in
`ruff.toml`). `.github/workflows/pre-commit.yml` runs the same hooks on every
push and PR. Run them before committing:

```bash
pip install pre-commit && pre-commit install   # hooks on every git commit
pre-commit run --all-files
```

---

## 5. Tests

```bash
conda activate zinc
cd astrohack
PYTHONPATH=src python -m pytest tests/unit -q          # unit suite
PYTHONPATH=src python -m pytest tests/stakeholder -q   # heavier, downloads bigger MSs
```

- Test data is **downloaded on demand** via `toolviper.utils.data.download(...)`
  in each test's `setup_class` (so the unit suite needs network the first time;
  it caches under a per-test `*_data/` folder and `teardown_class` removes it).
- The arcae boundary is exercised end-to-end by
  `tests/unit/user_facing_functions/test_extract_{pointing,holog,locit}.py` and
  `test_fringefit_locit.py` — run those first when changing anything in the
  `core/` modules that open CASA tables.
- A pre-existing `NumbaPendingDeprecationWarning` (reflected list) is harmless.

CI (`.github/workflows/python-testing-{linux,macos}.yml`) calls the shared
`nrao/gh-actions-templates-public` reusable workflows, so the build/test steps
are defined **outside this repo**. `pip install .` resolves arcae from PyPI on
every supported interpreter: arcae 0.5.8 ships cp311–cp314 wheels for
`manylinux_2_28` x86_64/aarch64, `macosx_14_0 arm64` and `macosx_15_0 x86_64`.

---

## 6. Gotchas

- **Dropbox + casacore table locking.** This tree lives under Dropbox, whose
  sync interferes with casacore lock files ("table is in use" / "Directory not
  empty"). The code already opens tables with `lockoptions={"option":
  "usernoread"}` and `readonly=True`, which avoids most of it. If you write a
  scratch table in a test, put it in `$TMPDIR`, not the Dropbox tree.
- **`conda activate` in non-interactive shells** may not actually switch the
  `python` on `PATH`. Verify with `which python` / use the absolute interpreter
  `/Users/jsteeb/miniforge3/envs/zinc/bin/python` when in doubt.
- **String vs numeric columns.** The code expects string columns as Python
  lists (`.index()`/`.pop()` are used on names/stations, as with
  python-casacore), but xradio's arcae shim returns them as numpy arrays, so
  every whole-column string read calls `.tolist()` — keep that when adding
  string reads. Numeric columns are numpy arrays (`.tolist()` is called
  explicitly where a list is needed).
- **Numba `@njit(cache=...)` functions** (`njit_caching` in
  `utils/constants.py`) hold the hot loops (e.g. `_extract_holog_chunk_jit`).
  Type changes to the arrays they receive can trigger recompiles or reflected-
  list warnings.
- **No `from astrohack import X` inside the package.** `__init__.py` imports
  are absolute and ruff-sorted, so a module it imports cannot rely on another
  name already being bound in the package namespace (circular import). Import
  from the defining module, e.g. `from astrohack.io.position_mds import ...`.

---

## 7. Quick reference

```bash
conda activate zinc

# install (editable) and import-check the arcae boundary
pip install -e .
python -c "import astrohack.core.extract_holog, astrohack.core.extract_pointing, astrohack.core.extract_locit; print('ok')"

# the tests that exercise the arcae boundary end-to-end
PYTHONPATH=src python -m pytest tests/unit/user_facing_functions/test_extract_pointing.py \
    tests/unit/user_facing_functions/test_extract_holog.py \
    tests/unit/user_facing_functions/test_extract_locit.py \
    tests/unit/user_facing_functions/test_fringefit_locit.py -q

# full unit suite
PYTHONPATH=src python -m pytest tests/unit -q

# lint/format (same hooks as CI)
pre-commit run --all-files
```

The table backend lives in **xradio** (`/Users/jsteeb/Dropbox/viper_dev/xradio`,
`src/xradio/_utils/_casacore/`); xradio's AGENT.md documents the arcae shim.
