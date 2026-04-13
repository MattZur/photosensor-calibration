"""
SiPM IV Curve Analysis
======================
Analyses Silicon Photomultiplier (SiPM) current-voltage (IV) curves from
tab-separated .txt data files. Supports plotting, overlay comparisons, and
breakdown voltage extraction via Gaussian or Landau fitting.

Usage examples:
    python sipm_iv_analysis.py --folder ./data --mode single --index 3
    python sipm_iv_analysis.py --folder ./data --mode overlay
    python sipm_iv_analysis.py --folder ./data --mode overlay --indices 2 3 --log
    python sipm_iv_analysis.py --folder ./data --mode breakdown --fit gaussian
    python sipm_iv_analysis.py --folder ./data --mode summary --fit landau
    python sipm_iv_analysis.py --folder ./data --mode breakdown --indices 0 5 6 7 8 9 12 14 --fit gaussian --save-plots

Run `python sipm_iv_analysis.py --help` for full option details.
"""

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import moyal


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def natural_key(text: str) -> list:
    """Sort key that orders strings with embedded numbers naturally.
    
    e.g. ['sipm2.txt', 'sipm10.txt'] instead of ['sipm10.txt', 'sipm2.txt']
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', text)]


def first_number_in_filename(name: str) -> int | None:
    """Return the first integer found in a filename, or None if absent."""
    match = re.search(r'\d+', name)
    if match:
        return int(match.group())
    print(f"Warning: no number found in filename '{name}'.")
    return None


def load_iv_file(file_path: str) -> pd.DataFrame:
    """Load a tab-separated IV data file and return a clean DataFrame.
    
    Expected file format: two columns (Voltage [V], Current [mA]), tab-separated.
    """
    df = pd.read_csv(file_path, sep="\t")
    df.columns = ["Voltage", "Current"]
    df["Voltage"] = pd.to_numeric(df["Voltage"], errors="coerce")
    df["Current"] = pd.to_numeric(df["Current"], errors="coerce")
    df.dropna(inplace=True)
    return df


def collect_txt_files(folder: str) -> list[str]:
    """Return full paths of all .txt files in *folder*, sorted naturally."""
    if not os.path.isdir(folder):
        sys.exit(f"Error: folder '{folder}' does not exist.")
    files = sorted(
        [f for f in os.listdir(folder) if f.endswith(".txt")],
        key=natural_key,
    )
    if not files:
        sys.exit(f"Error: no .txt files found in '{folder}'.")
    return [os.path.join(folder, f) for f in files]


def select_files(all_files: list[str], indices: list[int] | None) -> list[str]:
    """Return a subset of *all_files* by index, or the full list if indices is None."""
    if indices is None:
        return all_files
    max_idx = len(all_files) - 1
    for i in indices:
        if i < 0 or i > max_idx:
            sys.exit(f"Error: index {i} out of range (0–{max_idx}).")
    return [all_files[i] for i in indices]


def maybe_save(fig: plt.Figure, save_dir: str | None, name: str) -> None:
    """Save *fig* to *save_dir* if a save directory was requested."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Fit functions
# ---------------------------------------------------------------------------

def gaussian(x, A, mu, sigma):
    """Standard Gaussian function."""
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def landau(x, A, mu, sigma):
    """Landau distribution using SciPy's Moyal approximation."""
    return A * moyal.pdf(x, loc=mu, scale=sigma)


FIT_FUNCTIONS = {"gaussian": gaussian, "landau": landau}


# ---------------------------------------------------------------------------
# Plot modes
# ---------------------------------------------------------------------------

def plot_single(file_path: str, log: bool = False, save_dir: str | None = None) -> None:
    """Plot the IV curve for a single file."""
    df = load_iv_file(file_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df["Voltage"], df["Current"], marker="o", markersize=4)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (mA)")
    ax.set_title(f"IV Curve — {os.path.basename(file_path)}")
    if log:
        ax.set_yscale("log")
    ax.grid(True)
    plt.tight_layout()
    maybe_save(fig, save_dir, f"iv_single_{os.path.splitext(os.path.basename(file_path))[0]}")
    plt.show()


def plot_overlay(
    file_paths: list[str],
    log: bool = False,
    save_dir: str | None = None,
) -> None:
    """Overlay IV curves from multiple files on one plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for fp in file_paths:
        label = first_number_in_filename(os.path.basename(fp))
        df = load_iv_file(fp)
        ax.plot(
            df["Voltage"],
            df["Current"],
            marker="o",
            markersize=4,
            markevery=8,
            linewidth=1,
            label=label,
        )
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (mA)")
    ax.set_title("IV Curves — Overlay")
    if log:
        ax.set_yscale("log")
    ax.grid(True)
    ax.legend(title="SiPM #", fontsize=8)
    plt.tight_layout()
    maybe_save(fig, save_dir, "iv_overlay")
    plt.show()


def analyze_breakdown(
    file_paths: list[str],
    fit_type: str = "gaussian",
    window: float = 0.2,
    xlim: tuple[float, float] | None = None,
    save_dir: str | None = None,
) -> list[tuple[str, float, float]]:
    """
    Compute and plot the breakdown voltage for each file.

    Method:
      1. Compute y = (1/I) * (dI/dV).
      2. Fit a Gaussian or Landau to locate the peak (= breakdown voltage).
      3. Fallback to argmax if fitting fails.

    Parameters
    ----------
    file_paths : list of file paths to analyse
    fit_type   : 'gaussian' or 'landau'
    window     : half-width in volts for dynamic x-axis around breakdown
                 (ignored when xlim is set)
    xlim       : fixed (xmin, xmax) x-axis limits; overrides window
    save_dir   : directory to save plot images (None = don't save)

    Returns
    -------
    list of (sipm_label, breakdown_voltage, uncertainty) tuples
    """
    fit_fn = FIT_FUNCTIONS.get(fit_type)
    if fit_fn is None:
        sys.exit(f"Error: unknown fit type '{fit_type}'. Choose 'gaussian' or 'landau'.")

    results = []

    for fp in file_paths:
        sipm_n = str(first_number_in_filename(os.path.basename(fp)))
        df = load_iv_file(fp)

        V = df["Voltage"].values
        I = df["Current"].values

        # Drop last two points (often noisy at sweep end)
        V = V[:-2]
        I = I[:-2]

        dIdV = np.gradient(I, V)

        # Guard against division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(I != 0, dIdV / I, 0.0)

        # --- Initial parameter guesses ---
        A_guess = np.max(y) if fit_type == "gaussian" else 0.0
        mu_guess = V[np.argmax(y)]
        sigma_guess = (np.max(V) - np.min(V)) / 10

        fit_success = True
        try:
            popt, pcov = curve_fit(
                fit_fn,
                V,
                y,
                p0=[A_guess, mu_guess, sigma_guess],
                maxfev=10_000,
            )
            _A, mu, _sigma = popt
            mu_err = np.sqrt(pcov[1, 1])
        except Exception as exc:
            print(f"  Warning: fit failed for {os.path.basename(fp)} ({exc}). Using argmax fallback.")
            fit_success = False
            mu = mu_guess
            mu_err = 0.03

        results.append((sipm_n, mu, mu_err))
        print(f"  SiPM {sipm_n:>4s}  Vbd = {mu:.3f} ± {mu_err:.3f} V  (fit_ok={fit_success})")

        # --- Per-SiPM plot ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(V, y, "o", markersize=4, label="(1/I) · (dI/dV)")

        if fit_success:
            V_fit = np.linspace(V.min(), V.max(), 500)
            ax.plot(V_fit, fit_fn(V_fit, *popt), label=f"{fit_type.capitalize()} fit")

        if xlim:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(mu - window, mu + window)

        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1e2)
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("(1/I) · (dI/dV)")
        ax.set_title(f"Breakdown Analysis — SiPM {sipm_n}")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        maybe_save(fig, save_dir, f"breakdown_sipm{sipm_n}")
        plt.show()

    return results


def plot_breakdown_summary(
    results: list[tuple[str, float, float]],
    ylim: tuple[float, float] | None = None,
    save_dir: str | None = None,
) -> None:
    """
    Plot a summary of breakdown voltages sorted by voltage value,
    with error bars.

    Parameters
    ----------
    results : list of (label, breakdown_voltage, uncertainty) tuples
    ylim    : optional (ymin, ymax) to zoom in on the y-axis
    save_dir: directory to save the plot (None = don't save)
    """
    if not results:
        print("No results to summarise.")
        return

    results_sorted = sorted(results, key=lambda r: r[1])
    names = [r[0] for r in results_sorted]
    mus = [r[1] for r in results_sorted]
    mu_errs = [r[2] for r in results_sorted]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.7), 5))
    ax.errorbar(x, mus, yerr=mu_errs, fmt="o", capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel("SiPM #")
    ax.set_ylabel("Breakdown Voltage (V)")
    ax.set_title("Breakdown Voltages — Summary")
    ax.grid(True)
    plt.tight_layout()
    maybe_save(fig, save_dir, "breakdown_summary")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sipm_iv_analysis.py",
        description=(
            "Analyse SiPM IV curves from tab-separated .txt files.\n"
            "Four analysis modes are available: single, overlay, breakdown, summary.\n\n"
            "Quick-start examples:\n"
            "  # Plot one IV curve (file index 3)\n"
            "  python sipm_iv_analysis.py --folder ./data --mode single --index 3\n\n"
            "  # Overlay all curves on a log-scale y-axis\n"
            "  python sipm_iv_analysis.py --folder ./data --mode overlay --log\n\n"
            "  # Breakdown voltage for all files, Gaussian fit, save plots\n"
            "  python sipm_iv_analysis.py --folder ./data --mode breakdown --fit gaussian --save-plots\n\n"
            "  # Summary chart for a hand-picked subset\n"
            "  python sipm_iv_analysis.py --folder ./data --mode summary --indices 0 5 6 12 --fit landau"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Required ---
    parser.add_argument(
        "--folder", "-f",
        required=True,
        metavar="DIR",
        help="Path to the folder containing .txt IV-curve data files.",
    )

    # --- Mode ---
    parser.add_argument(
        "--mode", "-m",
        required=True,
        choices=["single", "overlay", "breakdown", "summary"],
        help=(
            "Analysis mode:\n"
            "  single    — plot one IV curve (requires --index)\n"
            "  overlay   — overlay multiple IV curves on one plot\n"
            "  breakdown — per-SiPM breakdown voltage with fit\n"
            "  summary   — breakdown + summary chart (runs breakdown then plots summary)"
        ),
    )

    # --- File selection ---
    parser.add_argument(
        "--index", "-i",
        type=int,
        metavar="N",
        help="Zero-based index of the single file to plot (used with --mode single).",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        metavar="N",
        help=(
            "Zero-based indices of files to include (space-separated). "
            "Omit to use all files. "
            "Example: --indices 0 2 5 11"
        ),
    )

    # --- Plot options ---
    parser.add_argument(
        "--log",
        action="store_true",
        help="Use a logarithmic y-axis for IV overlay plots.",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        metavar=("YMIN", "YMAX"),
        help="Y-axis limits for the breakdown summary chart. Example: --ylim 51.5 52.0",
    )
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        metavar=("XMIN", "XMAX"),
        help=(
            "Fixed x-axis limits for breakdown plots (volts). "
            "Overrides the automatic window around the fitted peak. "
            "Example: --xlim 49 53"
        ),
    )
    parser.add_argument(
        "--window",
        type=float,
        default=0.2,
        metavar="V",
        help=(
            "Half-width in volts for the dynamic x-axis window around the "
            "fitted breakdown voltage (default: 0.2). Ignored if --xlim is set."
        ),
    )

    # --- Fit options ---
    parser.add_argument(
        "--fit",
        choices=["gaussian", "landau"],
        default="gaussian",
        help=(
            "Fit function used to locate the breakdown voltage peak in "
            "(1/I)·(dI/dV) vs V (default: gaussian)."
        ),
    )

    # --- Output ---
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save all generated plots as PNG files to --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default="./plots",
        metavar="DIR",
        help="Directory for saved plots when --save-plots is active (default: ./plots).",
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="Print the indexed list of discovered .txt files and exit.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Collect and list files
    all_files = collect_txt_files(args.folder)

    if args.list_files:
        print(f"Found {len(all_files)} .txt file(s) in '{args.folder}':\n")
        for i, fp in enumerate(all_files):
            print(f"  [{i:>2d}]  {os.path.basename(fp)}")
        sys.exit(0)

    save_dir = args.output_dir if args.save_plots else None

    # -----------------------------------------------------------------------
    # Mode: single
    # -----------------------------------------------------------------------
    if args.mode == "single":
        if args.index is None:
            parser.error("--mode single requires --index N")
        fp = select_files(all_files, [args.index])[0]
        print(f"Plotting single IV curve: {os.path.basename(fp)}")
        plot_single(fp, log=args.log, save_dir=save_dir)

    # -----------------------------------------------------------------------
    # Mode: overlay
    # -----------------------------------------------------------------------
    elif args.mode == "overlay":
        files = select_files(all_files, args.indices)
        print(f"Overlaying {len(files)} IV curve(s) ...")
        plot_overlay(files, log=args.log, save_dir=save_dir)

    # -----------------------------------------------------------------------
    # Mode: breakdown
    # -----------------------------------------------------------------------
    elif args.mode == "breakdown":
        files = select_files(all_files, args.indices)
        print(f"\nBreakdown analysis on {len(files)} file(s), fit = {args.fit}\n")
        analyze_breakdown(
            files,
            fit_type=args.fit,
            window=args.window,
            xlim=tuple(args.xlim) if args.xlim else None,
            save_dir=save_dir,
        )

    # -----------------------------------------------------------------------
    # Mode: summary  (breakdown + summary chart)
    # -----------------------------------------------------------------------
    elif args.mode == "summary":
        files = select_files(all_files, args.indices)
        print(f"\nBreakdown analysis on {len(files)} file(s), fit = {args.fit}\n")
        results = analyze_breakdown(
            files,
            fit_type=args.fit,
            window=args.window,
            xlim=tuple(args.xlim) if args.xlim else None,
            save_dir=save_dir,
        )
        print("\nPlotting breakdown voltage summary ...")
        plot_breakdown_summary(
            results,
            ylim=tuple(args.ylim) if args.ylim else None,
            save_dir=save_dir,
        )


if __name__ == "__main__":
    main()
