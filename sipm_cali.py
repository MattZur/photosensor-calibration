"""
sipm_cali.py — SiPM Calibration Analysis Script
=================================================
Converted from sipm_cali.ipynb.

Usage
-----
  python sipm_cali.py <command> [options]

Commands
--------
  quick-hist       Plot a raw charge histogram for the first .h5 file found.
  fingerplot       Plot individual finger-plots for every .h5 file in a folder.
  combined         Plot a combined finger-plot across all .h5 files in a folder.
  overlay          Overlay normalised histograms from multiple .h5 files.
  subtract         Bin-by-bin difference plot between two .h5 files.
  fit-hdf          Fit the signal model to each .h5 file in a folder.
  fit-csv          Fit the signal model to a single CSV file (bad-SiPM mode).
  fit-csv-good     Fit the signal model to a single CSV file (good-SiPM mode).
  batch-fit-csv    Fit signal model to every matching CSV in a folder and
                   collect gains, then call summary automatically.
  summary          Plot gain summary with error bars from a prior batch-fit-csv run.
                   (Can also be called standalone if you supply --gains-file.)

Run any command with --help for its specific options.
"""

import argparse
import os
import re
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sco
import scipy.stats as scs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def natural_key(text):
    """Sort key that orders strings with embedded numbers naturally."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', text)]


def _list_h5(folder):
    return sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.h5')],
        key=natural_key,
    )


def _list_csv(folder, suffix):
    return sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(suffix)],
        key=natural_key,
    )


# ---------------------------------------------------------------------------
# Signal model  (shared by all fitting functions)
# ---------------------------------------------------------------------------

def signal(xs, bl, A, gain, sigmaq, poismu=1, maxpercent=0.999999999):
    """
    Multi-PE finger-plot model: sum of Poisson-weighted Gaussians.

    Parameters
    ----------
    xs      : array of x values (bin centres)
    bl      : baseline offset
    A       : overall amplitude
    gain    : charge per photo-electron
    sigmaq  : single-PE charge resolution
    poismu  : mean number of photo-electrons (Poisson mean)
    """
    poispeaks_pos = np.arange(0, scs.poisson.ppf(maxpercent, poismu))
    realpeaks_pos = gain * poispeaks_pos + bl
    realpeaks_amp = A * scs.poisson.pmf(poispeaks_pos, poismu)
    result = np.zeros_like(xs, dtype=float)
    for i in range(len(poispeaks_pos)):
        result += realpeaks_amp[i] * scs.norm.pdf(
            xs, loc=realpeaks_pos[i], scale=sigmaq * np.sqrt(i + 1)
        )
    return result


def _run_fit(bin_centers, counts, p0, bounds):
    """
    Wrapper around curve_fit.  Returns (popt, pcov) or (None, None) on failure.
    """
    try:
        popt, pcov = sco.curve_fit(
            signal, bin_centers, counts,
            p0=p0, maxfev=10000, bounds=bounds,
        )
        perr = np.sqrt(np.diag(pcov))
        print("\n--- Fit Results ---")
        for name, val, err in zip(['bl', 'A', 'gain', 'sigmaq', 'poismu'], popt, perr):
            print(f"  {name:>8} = {val:.6f} ± {err:.6f}")
        return popt, pcov
    except RuntimeError as e:
        print(f"  Fit failed: {e}")
        print("  Try adjusting the initial guesses or bounds.")
        return None, None


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def cmd_quick_hist(args):
    """
    Quick raw-charge histogram from the first .h5 file found in --folder.
    Reads the 'CALI/waveform_information' dataset.
    """
    files = _list_h5(args.folder)
    if not files:
        sys.exit(f"No .h5 files found in {args.folder}")
    path = files[0]
    print(f"Reading {path}")
    df = pd.read_hdf(path, 'CALI/waveform_information')
    print(f"Shape: {df.shape}")

    plt.figure()
    plt.hist(df['integrated_Q'], bins=args.bins)
    plt.ylim([0, args.ylim])
    plt.xlabel('Charge released (ADCs)')
    plt.ylabel('Frequency (Counts)')
    plt.title(f'Quick histogram — {os.path.basename(path)}')
    plt.tight_layout()
    _save_or_show(args)


def cmd_fingerplot(args):
    """
    Individual finger-plot histogram for each .h5 file in --folder.
    Reads the 'CALI/wf_info' dataset.
    """
    files = _list_h5(args.folder)
    if not files:
        sys.exit(f"No .h5 files found in {args.folder}")

    for path in files:
        print(f"\nProcessing {path}")
        df = pd.read_hdf(path, 'CALI/wf_info')
        print(f"  Shape: {df.shape}")

        plt.figure()
        plt.hist(df['integrated_Q'], bins=args.bins, alpha=0.7, label='Data')
        plt.ylim([0, args.ylim])
        plt.xlabel('Charge released (ADCs or V)')
        plt.ylabel('Frequency (Counts)')
        plt.title(os.path.splitext(os.path.basename(path))[0])
        plt.legend()
        plt.tight_layout()
        _save_or_show(args, suffix=os.path.splitext(os.path.basename(path))[0])


def cmd_combined(args):
    """
    Concatenate all .h5 files in --folder and plot one combined histogram.
    Reads the 'CALI/wf_info' dataset.
    """
    files = _list_h5(args.folder)
    if not files:
        sys.exit(f"No .h5 files found in {args.folder}")

    combined = pd.DataFrame()
    for path in files:
        print(f"  Loading {path}")
        df = pd.read_hdf(path, 'CALI/wf_info')
        combined = pd.concat([combined, df], ignore_index=True)

    print(f"Combined shape: {combined.shape}")
    plt.figure()
    plt.hist(combined['integrated_Q'], bins=args.bins, alpha=0.7)
    plt.ylim([0, args.ylim])
    plt.xlabel('Charge released (ADCs or V)')
    plt.ylabel('Frequency (Counts)')
    plt.title(f'Combined finger-plot — {len(files)} files')
    plt.tight_layout()
    _save_or_show(args)


def cmd_overlay(args):
    """
    Overlay normalised histograms from multiple .h5 files.
    All histograms are scaled so their zero-charge bin matches the first file.
    Reads the 'CALI/wf_info' dataset.

    Requires: a populated --folder with .h5 files.
    """
    files = _list_h5(args.folder)
    if len(files) < 1:
        sys.exit(f"No .h5 files found in {args.folder}")

    bin_width = args.bin_width

    all_data, labels = [], []
    for path in files:
        df = pd.read_hdf(path, 'CALI/wf_info')
        all_data.append(df['integrated_Q'].values)
        labels.append(os.path.splitext(os.path.basename(path))[0])
        print(f"  {path} — shape: {df.shape}")

    g_min = np.floor(min(d.min() for d in all_data))
    g_max = np.ceil(max(d.max() for d in all_data))
    bin_edges = np.arange(g_min, g_max + bin_width, bin_width)

    all_counts = [np.histogram(d, bins=bin_edges)[0] for d in all_data]

    zero_bin = np.clip(np.searchsorted(bin_edges, 0, side='right') - 1,
                       0, len(all_counts[0]) - 1)
    ref_zero = all_counts[0][zero_bin]

    fig, ax = plt.subplots(figsize=(10, 6))
    for counts, label in zip(all_counts, labels):
        z = counts[zero_bin]
        scale = ref_zero / z if z > 0 else 1.0
        ax.stairs(counts * scale, bin_edges, fill=True, alpha=0.5, label=label)

    ax.set_xlabel('Charge released (ADCs)')
    ax.set_ylabel('Normalised Frequency (zero-peak aligned)')
    ax.set_title(f'Overlay — bin width: {bin_width} ADC, zero-peak normalised')
    ax.legend()
    plt.tight_layout()
    _save_or_show(args)


def cmd_subtract(args):
    """
    Bin-by-bin subtraction of file[1] minus file[0] from --folder.
    Top panel: normalised overlay; bottom panel: difference.
    Reads the 'CALI/wf_info' dataset.

    Requires: at least two .h5 files in --folder.
    Depends on the same normalisation logic as overlay.
    """
    files = _list_h5(args.folder)
    if len(files) < 2:
        sys.exit("Need at least two .h5 files for subtract.  Found: " + str(len(files)))

    bin_width = args.bin_width

    df1 = pd.read_hdf(files[0], 'CALI/wf_info')
    df2 = pd.read_hdf(files[1], 'CALI/wf_info')
    label1 = os.path.splitext(os.path.basename(files[0]))[0]
    label2 = os.path.splitext(os.path.basename(files[1]))[0]

    g_min = np.floor(min(df1['integrated_Q'].min(), df2['integrated_Q'].min()))
    g_max = np.ceil(max(df1['integrated_Q'].max(), df2['integrated_Q'].max()))
    bin_edges = np.arange(g_min, g_max + bin_width, bin_width)

    c1, _ = np.histogram(df1['integrated_Q'], bins=bin_edges)
    c2, _ = np.histogram(df2['integrated_Q'], bins=bin_edges)

    zero_bin = np.clip(np.searchsorted(bin_edges, 0, side='right') - 1,
                       0, len(c1) - 1)
    s1 = c1[zero_bin] if c1[zero_bin] > 0 else 1.0
    s2 = c2[zero_bin] if c2[zero_bin] > 0 else 1.0
    n1, n2 = c1 / s1, c2 / s2
    diff = n2 - n1
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].stairs(n1, bin_edges, fill=True, alpha=0.5, label=label1)
    axes[0].stairs(n2, bin_edges, fill=True, alpha=0.5, label=label2)
    axes[0].set_xlabel('Charge released (ADCs)')
    axes[0].set_ylabel('Normalised Frequency (zero-peak = 1)')
    axes[0].set_title(f'Normalised histograms — bin width: {bin_width} ADC')
    axes[0].legend()

    axes[1].bar(bin_centres, diff, width=bin_width, color='steelblue', edgecolor='none')
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].set_xlabel('Charge released (ADCs)')
    axes[1].set_ylabel('Normalised Difference')
    axes[1].set_title(f'Bin-by-bin difference: {label2} − {label1}')

    plt.tight_layout()
    _save_or_show(args)


def cmd_fit_hdf(args):
    """
    Fit the signal model to each .h5 file in --folder.
    Reads the 'CALI/wf_info' dataset.

    Initial guesses: bl=0, A=max(counts), gain=--gain-guess,
                     sigmaq=gain*0.5, poismu=--poismu-guess.
    """
    files = _list_h5(args.folder)
    if not files:
        sys.exit(f"No .h5 files found in {args.folder}")

    for path in files:
        print(f"\n{'='*60}\nFitting {path}")
        df = pd.read_hdf(path, 'CALI/wf_info')

        fig, ax = plt.subplots()
        counts, bin_edges, _ = ax.hist(df['integrated_Q'], bins=args.bins,
                                       alpha=0.5, label='Data')
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        p0 = [0, counts.max(), args.gain_guess,
              args.gain_guess * 0.5, args.poismu_guess]
        bounds = ([-np.inf, 0, 0, 0, 0],
                  [np.inf, np.inf, np.inf, np.inf, np.inf])

        popt, _ = _run_fit(bin_centers, counts, p0, bounds)
        if popt is not None:
            xs = np.linspace(bin_centers[0], bin_centers[-1], 1000)
            ax.plot(xs, signal(xs, *popt), 'r-', linewidth=2, label='Fit')

        ax.set_ylim([0, args.ylim])
        ax.set_xlabel('Charge released (ADCs or V)')
        ax.set_ylabel('Frequency (Counts)')
        ax.set_title(os.path.splitext(os.path.basename(path))[0])
        ax.legend()
        plt.tight_layout()
        _save_or_show(args, suffix=os.path.splitext(os.path.basename(path))[0])


def cmd_fit_csv(args):
    """
    Fit the signal model to a single CSV file (bad-SiPM settings).

    Reads a plain CSV of charge areas, converts to nC (×1e9), and fits.
    Returns gain ± error printed to stdout.

    --file  : path to the CSV
    --bin-size : physical bin size in the raw (SI) units before ×1e9 scaling
    """
    path = args.file
    if not os.path.isfile(path):
        sys.exit(f"File not found: {path}")
    print(f"Reading {path}")
    areas = np.genfromtxt(path, delimiter=',')
    _fit_csv_core(areas, path, args,
                  p0=[0, 150, 0.075, 0.075 * 0.05, 3],
                  bounds=([-np.inf, 0, 0, 0, 1],
                          [np.inf, 200, 0.1, 0.01, 10]))


def cmd_fit_csv_good(args):
    """
    Fit the signal model to a single CSV file (good-SiPM settings).
    Slightly wider sigmaq bound compared to the bad-SiPM variant.

    --file  : path to the CSV
    """
    path = args.file
    if not os.path.isfile(path):
        sys.exit(f"File not found: {path}")
    print(f"Reading {path}")
    areas = np.genfromtxt(path, delimiter=',')
    _fit_csv_core(areas, path, args,
                  p0=[0, 200, 1e4, 1e4 * 0.1, 6],
                  bounds=([-np.inf, 0, 0, 0, 2],
                          [np.inf, 300, 5e4, 5e4 * 0.1, 15]))
                #   p0=[0, 200, 0.075, 0.075 * 0.1, 3], # for scope data
                #   bounds=([-np.inf, 0, 0, 0, 2],
                #           [np.inf, 300, 0.5, 0.05, 10]))


def _fit_csv_core(areas, path, args, p0, bounds):
    """Shared fitting logic for both CSV variants."""
    fig, ax = plt.subplots()
    counts, bin_edges, _ = ax.hist(areas * 1e9, bins=args.bins,
                                   alpha=0.5, label='Data')
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    popt, _ = _run_fit(bin_centers, counts, p0, bounds)
    if popt is not None:
        xs = np.linspace(bin_centers[0], bin_centers[-1], 5000)
        ax.plot(xs, signal(xs, *popt), 'r-', linewidth=2, label='Fit')
        gain = popt[2]
        gain_err = np.sqrt(np.diag(_)[2]) if _ is not None else float('nan')
        print(f"\nGain = {gain:.6f} ± {gain_err:.6f} (nC units after ×1e9)")

    ax.set_ylim([0, args.ylim])
    ax.set_xlabel('Charge released (nV·s or scaled)')
    ax.set_ylabel('Frequency (Counts)')
    ax.set_title(os.path.splitext(os.path.basename(path))[0])
    ax.legend()
    plt.tight_layout()
    _save_or_show(args)


def cmd_batch_fit_csv(args):
    """
    Fit the signal model to every CSV in --folder whose name ends with --suffix.
    Collects (gain, gain_err) per file then automatically calls the summary plot.

    This command chains batch fitting → plot_breakdown_summary.
    """
    files = _list_csv(args.folder, args.suffix)
    if not files:
        sys.exit(f"No CSV files matching *{args.suffix} found in {args.folder}")

    all_gains, all_g_err = [], []

    for path in files:
        print(f"\n{'='*60}\nFitting {path}")
        areas = np.genfromtxt(path, delimiter=',')

        fig, ax = plt.subplots()
        counts, bin_edges, _ = ax.hist(areas * 1e9, bins=args.bins,
                                       alpha=0.5, label='Data')
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        p0 = [0, 150, 0.075, 0.075 * 0.05, 3]
        bounds = ([-np.inf, 0, 0, 0, 1],
                  [np.inf, 200, 0.1, 0.01, 10])
        popt, pcov = _run_fit(bin_centers, counts, p0, bounds)

        if popt is not None:
            xs = np.linspace(bin_centers[0], bin_centers[-1], 5000)
            ax.plot(xs, signal(xs, *popt), 'r-', linewidth=2, label='Fit')
            all_gains.append(popt[2])
            all_g_err.append(np.sqrt(np.diag(pcov)[2]))
        else:
            all_gains.append(np.nan)
            all_g_err.append(np.nan)

        ax.set_ylim([0, args.ylim])
        ax.set_xlabel('Charge (nV·s scaled)')
        ax.set_ylabel('Counts')
        ax.set_title(os.path.splitext(os.path.basename(path))[0])
        ax.legend()
        plt.tight_layout()
        _save_or_show(args, suffix=os.path.splitext(os.path.basename(path))[0])

    # Automatically call the summary plot after all fits
    _plot_summary(files, all_gains, all_g_err, title=args.title)
    _save_or_show(args, suffix='summary')


def cmd_summary(args):
    """
    Plot gain summary with error bars.

    Standalone usage requires --gains-file pointing to a CSV with columns:
        label, gain, gain_err
    (produced manually or by a prior batch-fit-csv run with --save-gains).
    """
    if not args.gains_file:
        sys.exit("--gains-file is required for standalone summary.  "
                 "Run batch-fit-csv first, or supply a CSV with columns: label,gain,gain_err")
    data = np.genfromtxt(args.gains_file, delimiter=',', dtype=str)
    labels = data[:, 0].tolist()
    gains = data[:, 1].astype(float)
    errs = data[:, 2].astype(float)
    _plot_summary(labels, gains, errs, title=args.title)
    _save_or_show(args)


def _plot_summary(names, gains, errs, title="SiPM gains"):
    """Shared gain-summary error-bar plot."""
    x = np.arange(len(names))
    short_names = [os.path.splitext(os.path.basename(str(n)))[0] for n in names]

    plt.figure(figsize=(max(6, len(names) * 0.8), 5))
    plt.errorbar(x, gains, yerr=errs, fmt='o', capsize=5)
    plt.xticks(x, short_names, rotation=45, ha='right')
    plt.xlabel("Dataset")
    plt.ylabel("Gain (Vs)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()


# ---------------------------------------------------------------------------
# Output helper
# ---------------------------------------------------------------------------

def _save_or_show(args, suffix=None):
    if hasattr(args, 'output') and args.output:
        base, ext = os.path.splitext(args.output)
        fname = f"{base}_{suffix}{ext}" if suffix else args.output
        plt.savefig(fname, dpi=150)
        print(f"  Saved → {fname}")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # ---- shared arguments added to each sub-parser ----
    def add_common(p):
        p.add_argument('--output', '-o', default=None,
                       help='Save figure to this path instead of showing it. '
                            'For multi-file commands a suffix is appended automatically.')

    def add_folder(p, default='.'):
        p.add_argument('--folder', '-f', default=default,
                       help=f'Folder to scan for data files (default: {default})')

    def add_bins(p, default=100):
        p.add_argument('--bins', '-b', type=int, default=default,
                       help=f'Number of histogram bins (default: {default})')

    def add_ylim(p, default=500):
        p.add_argument('--ylim', type=float, default=default,
                       help=f'Upper y-axis limit (default: {default})')

    def add_bin_width(p, default=1.0):
        p.add_argument('--bin-width', type=float, default=default, dest='bin_width',
                       help=f'Bin width in charge units (default: {default})')

    # ---- quick-hist ----
    p = sub.add_parser('quick-hist', help=cmd_quick_hist.__doc__)
    add_folder(p)
    add_bins(p, 50)
    add_ylim(p, 3000)
    add_common(p)
    p.set_defaults(func=cmd_quick_hist)

    # ---- fingerplot ----
    p = sub.add_parser('fingerplot', help=cmd_fingerplot.__doc__)
    add_folder(p)
    add_bins(p, 100)
    add_ylim(p, 400)
    add_common(p)
    p.set_defaults(func=cmd_fingerplot)

    # ---- combined ----
    p = sub.add_parser('combined', help=cmd_combined.__doc__)
    add_folder(p)
    add_bins(p, 60)
    add_ylim(p, 5000)
    add_common(p)
    p.set_defaults(func=cmd_combined)

    # ---- overlay ----
    p = sub.add_parser('overlay', help=cmd_overlay.__doc__)
    add_folder(p)
    add_bin_width(p, 1.0)
    add_common(p)
    p.set_defaults(func=cmd_overlay)

    # ---- subtract ----
    p = sub.add_parser('subtract', help=cmd_subtract.__doc__)
    add_folder(p)
    add_bin_width(p, 1.0)
    add_common(p)
    p.set_defaults(func=cmd_subtract)

    # ---- fit-hdf ----
    p = sub.add_parser('fit-hdf', help=cmd_fit_hdf.__doc__)
    add_folder(p)
    add_bins(p, 100)
    add_ylim(p, 500)
    p.add_argument('--gain-guess', type=float, default=0.03, dest='gain_guess',
                   help='Initial guess for gain parameter (default: 0.03)')
    p.add_argument('--poismu-guess', type=float, default=2.0, dest='poismu_guess',
                   help='Initial guess for Poisson mean (default: 2.0)')
    add_common(p)
    p.set_defaults(func=cmd_fit_hdf)

    # ---- fit-csv ----
    p = sub.add_parser('fit-csv', help=cmd_fit_csv.__doc__)
    p.add_argument('--file', required=True, help='Path to the CSV file')
    p.add_argument('--bin-size', type=float, default=4e-12, dest='bin_size',
                   help='Bin size in raw (SI) units before ×1e9 (default: 4e-12)')
    add_bins(p, 200)
    add_ylim(p, 200)
    add_common(p)
    p.set_defaults(func=cmd_fit_csv)

    # ---- fit-csv-good ----
    p = sub.add_parser('fit-csv-good', help=cmd_fit_csv_good.__doc__)
    p.add_argument('--file', required=True, help='Path to the CSV file')
    p.add_argument('--bin-size', type=float, default=4e-12, dest='bin_size',
                   help='Bin size in raw (SI) units before ×1e9 (default: 4e-12)')
    add_bins(p, 200)
    add_ylim(p, 200)
    add_common(p)
    p.set_defaults(func=cmd_fit_csv_good)

    # ---- batch-fit-csv ----
    p = sub.add_parser('batch-fit-csv', help=cmd_batch_fit_csv.__doc__)
    add_folder(p)
    p.add_argument('--suffix', default='4_2V.csv',
                   help='CSV filename suffix to match (default: 4_2V.csv)')
    p.add_argument('--title', default='SiPM gains',
                   help='Title for the gain summary plot')
    add_bins(p, 200)
    add_ylim(p, 200)
    add_common(p)
    p.set_defaults(func=cmd_batch_fit_csv)

    # ---- summary ----
    p = sub.add_parser('summary', help=cmd_summary.__doc__)
    p.add_argument('--gains-file', default=None, dest='gains_file',
                   help='CSV with columns: label,gain,gain_err')
    p.add_argument('--title', default='SiPM gains',
                   help='Plot title (default: "SiPM gains")')
    add_common(p)
    p.set_defaults(func=cmd_summary)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
