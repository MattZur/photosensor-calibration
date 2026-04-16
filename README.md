# photosensor-calibration

A Python toolkit for calibrating Silicon Photomultipliers (SiPMs) and PMTs from
oscilloscope waveform data. The pipeline reads raw waveforms, integrates pulse
charges, fits single-photon spectra, and extracts gain and SNR as a function of
bias voltage.

---

## Repository structure

```
photosensor-calibration/
│
├── run_calibration.py      ← ENTRY POINT  pipeline: read → integrate → fit
├── sipm_cali.py            ← ENTRY POINT  analysis & plotting CLI
│
├── calibration/            library modules (not run directly)
│   ├── read_data.py            waveform file I/O
│   ├── compute_area.py         ROI detection, outlier filtering, pulse integration
│   ├── analyze_sipm.py         SiPM spectrum fitting (dependent / independent)
│   ├── analyze_pmt.py          PMT spectrum fitting
│   ├── configuration.py        shared folder paths (DATA_FOLDER, RESULTS_FOLDER, …)
│   └── utils/
│       ├── plotting_utils.py   shared matplotlib helpers
│       └── read_root.py        ROOT file reader (optional dependency)
│
├── iv_curves/
│   ├── sipm_iv_analysis.py     standalone IV-curve analysis (not part of main pipeline)
│   └── README.md
│
├── plots/                  output figures (git-tracked examples)
├── results/                output CSVs – one row per voltage per SiPM/PMT
└── README.md
```

> **`run_cali_mule.py`** at the root is a scratch/development file and is not
> part of the supported pipeline.

---

## Installation

```bash
git clone https://github.com/MattZur/photosensor-calibration.git
cd photosensor-calibration
pip install numpy scipy matplotlib pandas h5py
```

ROOT support (optional, needed only for `read_root.py`):

```bash
pip install uproot   # pure-Python ROOT reader, no ROOT installation required
```

---

## Quick start

### 1 — Full pipeline: `run_calibration.py`

Runs the three-stage pipeline for a single SiPM at one or more bias voltages:

| Stage | Module | What it does |
|-------|--------|--------------|
| 1 | `read_data` | Reads raw oscilloscope waveforms from disk |
| 2 | `compute_area` | Locates ROI, filters outliers, integrates pulse areas → CSV |
| 3 | `analyze_sipm` | Fits the single-photon spectrum, extracts gain & SNR → CSV |

**Configure and run:**

Open `run_calibration.py` and edit the `CONFIG` block near the top:

```python
CONFIG = dict(
    sipm_no      = 419,          # SiPM serial number
    sipm_str     = "dn",         # label used in data filenames
    channel      = "C4",         # oscilloscope channel prefix (e.g. "C1"), or None
    voltage_min  = 4,            # lowest bias voltage (V)
    voltage_max  = 4,            # highest bias voltage, inclusive (V)
    file_start   = 0,            # first waveform file index (inclusive)
    file_stop    = 10,           # last  waveform file index (exclusive)
    segment_no   = 1000,         # waveform segments per file
    data_root    = "/path/to/data/sipm/",
    bin_size     = 4e-12,        # histogram bin size (V·s)
    fitting_mode = "dependent",  # "dependent" or "independent" Gaussian fit
    save_plots   = True,
    debug_plots  = True,
)
```

Then run:

```bash
python run_calibration.py
```

**Expected data layout under `data_root`:**

```
<data_root>/
└── <sipm_no>/
    └── <voltage>V/
        └── [<channel>--]<sipm_str>_00001--0000.trc  ...
```

**Programmatic use** — you can also import and call individual stages:

```python
from run_calibration import run_area_stage, run_fit_stage, run_full_pipeline

run_area_stage(sipm_no=412, voltage_min=55, voltage_max=58, ...)
run_fit_stage( sipm_no=412, voltage_min=55, voltage_max=58, fitting_mode="dependent")
```

**Output:** area CSV files in `DATA_FOLDER` and a results CSV in `RESULTS_FOLDER`
(both configured in `calibration/configuration.py`).

---

### 2 — Analysis & plotting CLI: `sipm_cali.py`

A command-line tool for inspecting data and fitting spectra from `.h5` or `.csv`
files. Every sub-command accepts `--help` for full option details.

```
python sipm_cali.py <command> [options]
```

| Command | Input | What it does |
|---------|-------|--------------|
| `quick-hist` | `.h5` folder | Raw charge histogram from the first file found |
| `fingerplot` | `.h5` folder | Individual finger-plot for every file |
| `combined` | `.h5` folder | Single combined histogram across all files |
| `overlay` | `.h5` folder | Normalised, zero-peak-aligned overlay of all files |
| `subtract` | `.h5` folder | Bin-by-bin difference plot between two files |
| `fit-hdf` | `.h5` folder | Fit signal model to every file; report gain & SNR |
| `fit-csv` | single `.csv` | Fit signal model (bad-SiPM mode) |
| `fit-csv-good` | single `.csv` | Fit signal model (good-SiPM mode) |
| `batch-fit-csv` | `.csv` folder | Fit every matching CSV, then auto-generate summary |
| `summary` | gains CSV | Gain summary plot with error bars |

**Examples:**

```bash
# Quick look at raw data
python sipm_cali.py quick-hist --folder data/run42/ --bins 80

# Fit every HDF5 file in a folder, save figures
python sipm_cali.py fit-hdf --folder data/run42/ --output plots/run42.png

# Batch-fit all CSVs ending in 4_2V.csv, save the summary
python sipm_cali.py batch-fit-csv \
    --folder results/ \
    --suffix 4_2V.csv \
    --title "SiPM gain at 4.2 V OV" \
    --output plots/summary.png

# Standalone gain summary from a pre-built gains CSV
python sipm_cali.py summary \
    --gains-file results/gains.csv \
    --title "Run 42 gains"
```

All plot commands accept `--output <path>` to save a PNG instead of displaying
interactively. For multi-file commands a suffix is appended automatically
(e.g. `plots/run42_sipm419.png`).

---

## Configuration

Shared folder paths (where area CSVs and result CSVs are written) live in
`calibration/configuration.py`. Edit `DATA_FOLDER`, `RESULTS_FOLDER`, and
`PLOTS_FOLDER` there rather than hard-coding paths in the entry-point scripts.

---

## Outputs

| Location | Contents |
|----------|----------|
| `DATA_FOLDER/` | `areas-sipm_<id>-<voltage>.csv` — integrated pulse areas |
| `RESULTS_FOLDER/` | `results_sipm-<id>.csv` — voltage, gain, SNR per row |
| `PLOTS_FOLDER/` | PNG figures (when `save_plots=True` or `--output` is used) |

Example results files are committed under `results/` for reference.

---

## IV curves

The `iv_curves/` sub-folder contains a standalone script for IV-curve analysis
(`sipm_iv_analysis.py`). It is independent of the main pipeline; see
`iv_curves/README.md` for details.