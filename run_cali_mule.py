"""
run_calibration.py
==================
Entry point for the photosensor-calibration pipeline.

Full pipeline for a single SiPM:
  1. Read raw oscilloscope waveforms  (read_data)
  2. Determine ROI, filter outliers, integrate pulses  (compute_area)
  3. Fit the single-photon spectrum and extract gain / SNR  (analyze_sipm)

Usage
-----
Edit the CONFIG block below, then run:

    python run_calibration.py

Or import and call individual stages programmatically:

    from run_calibration import run_full_pipeline, run_area_stage, run_fit_stage
"""

import sys
import os

# ---------------------------------------------------------------------------
# Adjust these paths so that the repo modules and its utils are importable.
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
UTILS_PATH = os.path.join(REPO_ROOT, "utils")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, UTILS_PATH)

import calibration.read_data as reader
import calibration.compute_area as area_mod
import calibration.analyze_sipm as sipm_mod

# ---------------------------------------------------------------------------
# CONFIG – change these values to match your dataset
# ---------------------------------------------------------------------------
CONFIG = dict(
    # ---- SiPM identity ----
    sipm_no      = 400,          # integer used in all data/result filenames
    sipm_str     = "dn",          # label used inside the data filenames (often "1")
    channel      = "C4",         # oscilloscope channel prefix, e.g. "C1", "C2", or None

    # ---- Voltage sweep ----
    voltage_min  = 4,           # lowest bias voltage (integer, V)
    voltage_max  = 4,           # highest bias voltage (inclusive, integer, V)

    # ---- Raw-data reading ----
    file_start   = 0,            # first 4-digit file index (inclusive)
    file_stop    = 10,           # last  4-digit file index (exclusive)
    segment_no   = 1000,         # waveform segments per oscilloscope file
    data_root    = "/Users/user/Random/notebooks/todor_code/data/sipm/",# base directory; subfolders <sipm_no>/<voltage>/ expected
    file = '/Users/user/Random/notebooks/todor_adapt_to_mule/sipm348_diffamp_2us.h5',

    # ---- Spectrum fitting ----
    bin_size     = 4e-12,      # histogram bin size for the charge spectrum (V·s)
    fitting_mode = "dependent",  # "dependent" or "independent" Gaussian fit
    save_plots   = True,        # write PNG files to PLOTS_FOLDER
    debug_plots  = True,        # show matplotlib debug windows during fitting
)
# ---------------------------------------------------------------------------


def run_area_stage(sipm_no, voltage_min, voltage_max,
                   channel=None, sipm_str="1",
                   file_start=0, file_stop=25, segment_no=1000,
                   data_root="/Users/user/Random/notebooks/todor_code/data/sipm/",
                   file=None):
    """
    Stage 1 + 2: read all raw waveforms for every bias voltage of one SiPM,
    then locate the ROI, filter outliers, integrate pulses and save a per-voltage
    .csv file of pulse-charge areas.

    Parameters
    ----------
    sipm_no : int
        SiPM serial number as it appears in the data directory names.
    voltage_min, voltage_max : int
        Inclusive voltage range to process (V).
    channel : str or None
        Oscilloscope channel prefix (e.g. "C2"). Pass None if the filenames have
        no channel prefix.
    sipm_str : str
        String used inside the filename to identify the SiPM (often "1" when a
        single SiPM was recorded, or e.g. "412413" for dual recordings).
    file_start, file_stop : int
        Range [start, stop) of the 4-digit index that is appended to every
        oscilloscope file name.
    segment_no : int
        Number of waveform segments stored in each oscilloscope file.
    data_root : str
        Root directory; raw data are expected at
        ``<data_root>/<sipm_no>/<voltage>V/``.

    Returns
    -------
    None
        Areas are written to CSV files in the DATA_FOLDER configured in
        configuration.py.
    """
    print("=" * 60)
    print(f"Area stage: SiPM {sipm_no}, {voltage_min}V – {voltage_max}V")
    print("=" * 60)

    v = 4.2 # specific overvoltage HARD-CODED
    voltage_str = f"{v}V"
    print(f"\n--- Processing {voltage_str} ---")

    # Build filename and path the same way procedure_areas_save() does
    fname = f"{sipm_str}_00001--"
    if channel is not None:
        fname = f"{channel}--{fname}"
    location = os.path.join(data_root, str(sipm_no), voltage_str) + "/"

    # 1. Read waveforms
    if file is not None:
        all_waveforms = reader.get_waveforms(file)
    else:
        all_waveforms = reader.iterate_large_files(
            file_start, file_stop, fname,
            segment_no=segment_no, loc=location
        )
    print(f"   Loaded {len(all_waveforms)} waveforms.")

    # 2. Locate pulse ROI
    roi_begin, roi_end, peak_loc = area_mod.determine_roi(all_waveforms, plot=True)
    roi = [roi_begin, roi_end]

    # 3. Remove outlier waveforms
    filtered = area_mod.filter_outliers(all_waveforms, peak_loc, roi, plot=True)
    print(f"   {len(filtered)} waveforms retained after filtering.")

    # 4. Integrate and save
    #savename = f"areas_sipm-{sipm_no}_{voltage_str}.csv"
    savename = f"areas-sipm_{sipm_no}-4_2V.csv"
    areas, _, _ = area_mod.find_area(
        filtered, roi,
        no_bins=300, save=True, plot=True,
        savename=savename
    )
    print(f"   Areas saved  →  {savename}  ({len(areas)} entries)")


def run_fit_stage(sipm_no, voltage_min, voltage_max,
                  fitting_mode="dependent",
                  bin_size=2.7e-12,
                  save_plots=False,
                  debug_plots=False):
    """
    Stage 3: read the per-voltage area CSV files produced by :func:`run_area_stage`
    and fit the single-photon charge spectrum for every bias voltage.  Produces a
    gain–voltage curve, calculates the SNR and writes a summary CSV to RESULTS_FOLDER.

    Parameters
    ----------
    sipm_no : int
        SiPM serial number.
    voltage_min, voltage_max : int
        Inclusive voltage range to process (V).
    fitting_mode : {"dependent", "independent"}
        ``"dependent"``  – peak positions are constrained by a common gain G and a
                           single pedestal width (fewer free parameters, preferred).
        ``"independent"`` – each peak is fitted with its own mean and width.
    bin_size : float
        Histogram bin size used to build the charge spectrum (V·s).
    save_plots : bool
        If True, save pretty PNG plots to PLOTS_FOLDER.
    debug_plots : bool
        If True, display debug plots interactively during fitting.

    Returns
    -------
    None
        Results (voltage, gain, SNR, …) are written to
        ``RESULTS_FOLDER/results_sipm-<sipm_no>.csv``.
    """
    print("=" * 60)
    print(f"Fit stage: SiPM {sipm_no}, {voltage_min}V – {voltage_max}V  [{fitting_mode} fit]")
    print("=" * 60)

    if fitting_mode == "dependent":
        fitting_procedure = sipm_mod.procedure_dep_fit
    elif fitting_mode == "independent":
        fitting_procedure = sipm_mod.procedure_indep_fit
    else:
        raise ValueError(f"fitting_mode must be 'dependent' or 'independent', got {fitting_mode!r}")

    sipm_mod.do_all_fits(
        fitting_procedure,
        sipm_no=sipm_no,
        voltage=[voltage_min, voltage_max],
        bin_size=bin_size,
        save=save_plots,
        plot=debug_plots,
    )
    print(f"\nFits complete. Results written to RESULTS_FOLDER/results_sipm-{sipm_no}.csv")


def run_full_pipeline(cfg: dict):
    """
    Convenience wrapper that runs the complete calibration pipeline for a single
    SiPM using the settings in *cfg*.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary; see the CONFIG block at the top of this file
        for all required keys and their meaning.

    Example
    -------
    >>> from run_calibration import run_full_pipeline, CONFIG
    >>> CONFIG["sipm_no"] = 412
    >>> CONFIG["voltage_min"] = 55
    >>> CONFIG["voltage_max"] = 58
    >>> run_full_pipeline(CONFIG)
    """
    run_area_stage(
        sipm_no    = cfg["sipm_no"],
        voltage_min= cfg["voltage_min"],
        voltage_max= cfg["voltage_max"],
        channel    = cfg["channel"],
        sipm_str   = cfg["sipm_str"],
        file_start = cfg["file_start"],
        file_stop  = cfg["file_stop"],
        segment_no = cfg["segment_no"],
        data_root  = cfg["data_root"],
        file = cfg['file']
    )
    # run_fit_stage(
    #     sipm_no     = cfg["sipm_no"],
    #     voltage_min = cfg["voltage_min"],
    #     voltage_max = cfg["voltage_max"],
    #     fitting_mode= cfg["fitting_mode"],
    #     bin_size    = cfg["bin_size"],
    #     save_plots  = cfg["save_plots"],
    #     debug_plots = cfg["debug_plots"],
    # )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_full_pipeline(CONFIG)
