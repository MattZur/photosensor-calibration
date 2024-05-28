# photosensor-calibration
Analysing single photon spectra in visible wavelengths.  Description of code contents is given here.  Fore more info read my interim MPhys report.


# Instructions on code use
In this appendix, detailed instructions are presented on how to use the code for the data analysis. The code is available publicly on GitHub and consists of 4 scripts:
- read data.py,
- compute area.py,
- analyze sipm.py,
- analyze pmt.py,
- configuration.py.
The last of those files contains the file paths for data input and output that should be changed depending on the system used. In the next parts of this appendix, the remaining scripts will be discussed in detail.

## read data.py
A script made for data reading from the format our oscilloscope records data. Multiple data files are saved for every run and after a filename is selected on the oscilloscope the names are saved as `<filename>--<4-digit number of file>`.txt. The function `iterate_large_files()` requires to selection of `start` and `stop` parameters that correspond to the first(inclusive) and last(non-inclusive) 4-digit number of files. In other words, these 2 parameters specify over which files to iterate. After specifying the name of the file in `filename` and the location of the file in loc the function will iterate over all files and return a list of all waveforms read from those files. Every waveform is a 2D NumPy array with the first column being the time in seconds and the second being the amplitude in volts. 

This script also provides a function `make_heatmap()` that will plot all waveforms you select on a heatmap-type of plot(faster than plotting all on top of each other).

## compute area.py
This script contains the determination of pulse beginning and peak, filtering and integration. The function `determine_roi()` takes as an input a list of waveforms in the format explained in previously. It then applies the smoothing filter and returns the index of the beginning, end and peak of the pulse. The beginning and peak are explained in the interim report, while the end is arbitrarily chosen so that the interval is symmetric around the peak. This value is only used as an initial guess in the integration window determination and is not of importance.

The function `filter_outliers()` takes as input all waveforms in the above mentioned format, the index of the peak and the beginning and end of the peak as determined by the previous function. It then iterates all waveforms and rejects the one that have a maximum away from a window of that is symmetric around the maximum and has a width of 75 % the width of the region of interest determined above. 

Finally one needs to apply `find_area()` to perform the second filter and integrate. It iterates all waveforms 3 times. On the first loop, the baseline regions of all waves (flat region before the pulse) are fit with a linear function. The waves that correspond to top and bottom 10 % of slopes from that fit are then rejected. On the second loop, finding the point where each waveform drops below 10 % of its peak value is attempted and the 23 indices are recorded in a list. Flat waveforms are skipped. On the third time, integration via the trapezoidal rule is performed up until the index determined in the previous step. If no index is determined, an average of all others is used. The areas are all put in a list and this list is saved to a ‘.csv’ file at the end. 

For ease of use two more functions are implemented. `procedure_areas_save()` calls the required functions to read all data files for one voltage value of a single SiPM. After that the points of interest are located, filters applied, waveforms integrated and the areas are saved to a file. Similarly, `save_all_areas()` calls `procedure_areas_save()` for all voltages from a single SiPM.

## analyze sipm.py
This script contains the full SiPM analysis. It begins with the implementation of the different models that are tested. Then two identical functions are written to perform the independent and dependent fit respectively. After the initial parameters are provided as explained in the code, these functions will attempt optimization with the MIGRAD algorithm from the MINUIT2 library (here called iminuit). By selecting the value for the `plot` parameter to be `True` a debug plot will be printed, showing the initial guess and the optimized fit. This can help to get the fits working. Similarly to the area calculation, these functions are called from `procedure_indep_fit()` and `procedure_dep_fit()` respectively, where the areas are read from a file, saved at a previous step of the analysis, and a fit is performed. Moreover, a function `do_all_fits()` calls one of the two fitting procedure functions by choice for all voltages from a single SiPM. It performs all fits, plots the gain- voltage curve, calculates the SNR and saves a ‘.txt’ file in the results folder with all the results. It also holds the initial fit parameters that may need to be tweaked a little to get the fits to work. Only this function should be called in most cases. At the end, there is a function `overvoltages_plot()` to extrapolate the gain at a specified overvoltage and plot it.

## analyze pmt.py
This file is completely identical to analyze sipm.py but the models are slightly different, which introduces slight changes at different places in the code.
