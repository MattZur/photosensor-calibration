# SiPM IV Curve Analysis

A command-line Python script for analysing Silicon Photomultiplier (SiPM) current–voltage (IV) curves. It reads tab-separated `.txt` data files, plots raw IV curves, computes breakdown voltages, and summarises results across multiple devices.

To collect the `.txt` files, use the Keithley ammmeter and ensure the data collected is as formatted with two columns, Voltage (V) and Current (mA).

---

## Background — How Breakdown Voltage is Extracted

The breakdown voltage V_bd is identified as the voltage at which the SiPM begins avalanche multiplication. The quantity:

```
y(V) = (1/I) · (dI/dV)
```

develops a sharp peak at V_bd. This peak is located by:

- **Gaussian fit** — works well when the peak is symmetric and data noise is low.
- **Landau fit** (Moyal approximation) — better for asymmetric peaks or noisy data at high voltages.

If curve fitting fails the script falls back to the raw argmax of `y(V)` and assigns a conservative uncertainty of ±0.03 V.

---

## Requirements

Install dependencies with pip:

```bash
pip install numpy pandas matplotlib scipy
```

Python 3.10 or newer is recommended (the script uses `int | None` type hints).

---

## Usage

```
python sipm_iv_analysis.py --folder <DIR> --mode <MODE> [options]
```

## All Options

| Option | Default | Description |
|--------|---------|-------------|
| `--folder DIR` | *(required)* | Folder containing `.txt` data files |
| `--mode MODE` | *(required)* | `single`, `overlay`, `breakdown`, or `summary` |
| `--index N` | — | File index for `--mode single` |
| `--indices N [N ...]` | all files | Space-separated file indices for other modes |
| `--log` | off | Logarithmic y-axis for IV overlay plots |
| `--fit {gaussian,landau}` | `gaussian` | Fit function for breakdown voltage extraction |
| `--window V` | `0.2` | Half-width (V) for the dynamic x-axis window around the breakdown peak |
| `--xlim XMIN XMAX` | auto | Fixed x-axis limits for breakdown plots (overrides `--window`) |
| `--ylim YMIN YMAX` | auto | Y-axis limits for the summary chart |
| `--save-plots` | off | Save all plots as PNG files |
| `--output-dir DIR` | `./plots` | Output directory for saved plots (used with `--save-plots`) |
| `--list-files` | *(reccomended)* | Print indexed file list and exit — useful for finding indices |

---

## Modes

### `single` — plot one IV curve

Plots the IV curve for a single file chosen by its index.

```bash
python sipm_iv_analysis.py --folder ./data --mode single --index 3
```

Add `--log` to use a logarithmic y-axis:

```bash
python sipm_iv_analysis.py --folder ./data --mode single --index 0 --log
```

---

### `overlay` — overlay multiple curves

Overlays IV curves from several files on one plot. Omit `--indices` to include all files.

```bash
# All files
python sipm_iv_analysis.py --folder ./data --mode overlay

# Specific files (by index), log scale
python sipm_iv_analysis.py --folder ./data --mode overlay --indices 0 2 4 11 13 --log
```

---

### `breakdown` — extract breakdown voltages

For each selected file the script:

1. Computes `y = (1/I) · (dI/dV)` — a quantity that peaks sharply at the breakdown voltage.
2. Fits a **Gaussian** (default) or **Landau** function to locate the peak.
3. Falls back to the raw argmax position if the fit fails.
4. Prints `Vbd ± uncertainty` for each SiPM and shows an individual plot.

```bash
# Gaussian fit on all files
python sipm_iv_analysis.py --folder ./data --mode breakdown --fit gaussian

# Landau fit on a subset, with a fixed x-axis window
python sipm_iv_analysis.py --folder ./data --mode breakdown \
    --fit landau --indices 0 5 6 7 8 9 12 14 --xlim 49 53
```

---

### `summary` — breakdown + summary chart

Runs the full breakdown analysis and then plots a sorted summary chart with error bars.

```bash
python sipm_iv_analysis.py --folder ./data --mode summary \
    --fit gaussian --ylim 51.5 52.0
```

---

## Saving Plots

Add `--save-plots` to any command to write PNG files to `./plots` (or a custom directory with `--output-dir`):

```bash
python sipm_iv_analysis.py --folder ./data --mode summary \
    --fit gaussian --save-plots --output-dir ./results/plots
```

---

## Typical Workflow

```bash
# 1. Check which files are available and note their indices
python sipm_iv_analysis.py --folder ./data --list-files

# 2. Quick visual check of a single device
python sipm_iv_analysis.py --folder ./data --mode single --index 0

# 3. Overlay all devices to spot outliers
python sipm_iv_analysis.py --folder ./data --mode overlay --log

# 4. Re-overlay only the "good" devices after visual inspection
python sipm_iv_analysis.py --folder ./data --mode overlay \
    --indices 0 5 6 7 8 9 12 14 15 16 17 --log

# 5. Breakdown analysis on the good subset
python sipm_iv_analysis.py --folder ./data --mode summary \
    --indices 0 5 6 7 8 9 12 14 15 16 17 \
    --fit gaussian --ylim 51.5 52.0 --save-plots
```

---