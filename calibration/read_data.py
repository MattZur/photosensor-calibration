import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import sys 
import h5py

sys.path.append("/home/todor/University/MPhys project/MPhys_project/utils/")
from calibration.utils.plotting_utils import plot2d
import matplotlib.colors
import matplotlib.cm as colormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from calibration.configuration import PLOTS_FOLDER

event_info_type       = np.dtype([
            ('event_number', np.uint32),
            ('timestamp', np.uint64),
            ('samples', np.uint32),
            ('sampling_period', np.uint64),
            ('channels', np.int32),
            ])

calibration_info_type = np.dtype([
            ('event_number', np.uint32),
            ('channels',     np.uint32),
            ('integrated_Q',  np.float64),
            ('height',       np.float64),
            ])

def rwf_type(samples  :  int) -> np.dtype:
    """
    Generates the data-type for raw waveforms

    Parameters
    ----------

        samples  (int)  :  Number of samples per waveform

    Returns
    -------

        (ndtype)  :  Desired data type for processing


    """
    return np.dtype([
            ('event_number', np.uint32),
            ('channels', np.int32),
            ('rwf', np.float32, (samples,))
        ])

def load_evt_info(file_path, merge = False):
    '''
    Loads in a processed WD .h5 file as pandas DataFrame, extracting event information tables.
    This function allows the processed WD .h5 file to be chunked or unchunked.
    Chunked is the older file format where the h5 structure is of the form /event_information/block$NUM_values.
    Unchunked is of the form /RAW/event_info without any block$NUM_values.

    Parameters
    ----------

    file_path (str)   :  Path to saved data
    merge     (bool)  :  Flag for merging chunked data

    Returns
    -------

    (pd.DataFrame)  :  Dataframe of event information
    '''
    h5_data = []
    with h5py.File(file_path) as f:
        # extract event info
        if list(f.keys())[0] == 'RAW': # case for unchunked data
            evt_info = f.get('RAW/event_info')
            h5_data = evt_info[:]
        else:
            evt_info = f.get('event_information') # case for chunked data
            for i in evt_info.keys():
                q = evt_info.get(str(i))
                for j in q:
                    h5_data.append(j)

    return pd.DataFrame(map(list, h5_data), columns = (event_info_type).names)

def load_rwf_info(file_path  :  str,
                  samples    :  int) -> list:
    '''
    Loads in a processed WD .h5 file as pandas dataframe, extracting raw waveform tables.
    Samples must be provided, and can be found using `load_evt_info()`.
    This function allows the processed WD .h5 file to be chunked or unchunked.
    Chunked is the older file format where the h5 structure is of the form /rwf/block$NUM_values.
    Unchunked is of the form /RAW/rwf without any block$NUM_values.

    Parameters
    ----------

    file_path (str)  :  Path to saved data
    samples   (int)  :  Number of samples in each raw waveform

    Returns
    -------

    (pd.DataFrame)  :  Dataframe of raw waveform information
    '''
    h5_data = []
    with h5py.File(file_path) as f:
        if list(f.keys())[0] == 'RAW':
            rwf_info = f.get('RAW/rwf')
            h5_data = rwf_info[:]
        else:
            rwf_info = f.get('rwf')
            for i in rwf_info.keys():
                q = rwf_info.get(str(i))
                for j in q:
                    h5_data.append(j)

    return pd.DataFrame(map(list, h5_data), columns = (rwf_type(samples)).names)

def get_waveforms(file):
    filename = (file.rsplit('.')[0])
    # Load event + waveform info
    wf_evt = load_evt_info(file)
    samples = int(wf_evt.loc[0].samples)
    sampling_period = (int(wf_evt.loc[0].sampling_period)*(10**-9)) 
    wf_rwf = load_rwf_info(file, samples)
    
    all_waveforms = []
    for i, wf_num in enumerate(range(len(wf_evt))):
        #single_wf = single_wf_manipulation(wf_rwf['rwf'][wf_num], 0.1)
        single_wf = wf_rwf['rwf'][wf_num]
        time = [float(x) * sampling_period for x in range(len(single_wf))]
        all_waveforms.append(np.column_stack((time, single_wf)))
    return all_waveforms

def generate_counter_string(iterator):
    """Generate the counter string in every filename from an integer. (e.g. 00023 from 23)

    Parameters
    ----------
    iterator : int
        the file count 

    Returns
    -------
    string
        The file name ending
    """
    appendix = ""
    if iterator < 10:
        appendix = "0000{}".format(iterator)
    elif iterator < 100:
        appendix = "000{}".format(iterator)
    elif iterator < 1000:
        appendix = "00{}".format(iterator)
    elif iterator < 10000:
        appendix = "0{}".format(iterator)
    else:
        print("Illegal filename exception")
        appendix = "00000"  

    return appendix  



def read_large_file(filename, loc="/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/data/"):
    """Read a single large file

    Parameters
    ----------
    filename : str
        filename
    loc : str, optional
        filepath, by default "/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/data/"

    Returns
    -------
    _type_
        _description_
    """
    all_waveforms = []
    raw_data = np.genfromtxt(loc + filename, skip_header=0, delimiter=',')
    new_waveform = np.array([[-1, -1]])
    for index, entry in enumerate(raw_data):
            #print(entry)
        if entry[0] < raw_data[index - 1, 0]:
            if index == 0:
                new_waveform = np.append(new_waveform, [entry], axis=0)
                continue
            new_waveform = np.delete(new_waveform, 0, axis=0)
            all_waveforms.append(new_waveform)
                #new_waveform = [entry]
            new_waveform = np.array([[-1, -1], entry])
                #print("saving...")
        else:
                

            new_waveform = np.append(new_waveform, [entry], axis=0)
                #new_waveform.append(entry)
            if index == len(raw_data) - 1:
                    #print(new_waveform)
                new_waveform = np.delete(new_waveform, 0, axis=0)
                all_waveforms.append(new_waveform)
                    #new_waveform = []
                new_waveform = np.array([[-1, -1]])    
    
    return all_waveforms



def iterate_large_files(start, stop, filename, segment_no=1000, loc="/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/data/"):
    """Iterate over all files and split all the segments into a single list

    Parameters
    ----------
    start : int
        filename count start
    stop : int
        filename count stop
    filename : string
        the filename to look for
    segment_no : int, optional
        the number of segments in each file, by default 1000
    loc : str, optional
        the location of the file, by default "/home/todor/University/MPhys project/MPhys_project/analyze-lecroy/data/"

    Returns
    -------
    list<numpy.array>
        a list with each entry being an numpy array containing the time in the 0th column and the amplitude in the 1st column
    """

    header = 4 + segment_no
    all_waveforms = []
    print("Reading files...")
    for iterator in tqdm(range(start, stop)):
        file = loc + filename + generate_counter_string(iterator) + ".txt"
        raw_data = np.genfromtxt(file, skip_header=header, delimiter=',')
        #new_waveform = []
        new_waveform = np.array([[-1, -1]])
        for index, entry in enumerate(raw_data):
            #print(entry)
            if entry[0] < raw_data[index - 1, 0]:
                if index == 0:
                    new_waveform = np.append(new_waveform, [entry], axis=0)
                    continue
                new_waveform = np.delete(new_waveform, 0, axis=0)
                all_waveforms.append(new_waveform)
                #new_waveform = [entry]
                new_waveform = np.array([[-1, -1], entry])
                #print("saving...")
            else:
                

                new_waveform = np.append(new_waveform, [entry], axis=0)
                #new_waveform.append(entry)
                if index == len(raw_data) - 1:
                    #print(new_waveform)
                    new_waveform = np.delete(new_waveform, 0, axis=0)
                    all_waveforms.append(new_waveform)
                    #new_waveform = []
                    new_waveform = np.array([[-1, -1]])
        
        #print(all_waveforms)

    return all_waveforms

def make_heatmap(all_waveforms, save=False, savename="initial_data_reading_10x1000waveforms_heatmap.png", plot_title=False, title="Recorded waveforms for a single SiPM at 56V bias"):
    """Make a heatmap of the waveforms (faster than plotting all waveforms)

    Parameters
    ----------
    all_waveforms : list<numpy.array>
        a list of numpy arrays with each array being 1 waveform with timepoints in the 0th column and amplitudes in the 1st column
    save : bool, optional
        save the figure or not, by default False
    savename : str, optional
        name of the file to save, by default "initial_data_reading_10x1000waveforms_heatmap.png"
    """
    time = []
    amplitude = []
    print("Making a heatmap...")
    for index, waveform in tqdm(enumerate(all_waveforms)):
        for index_inner, single_point in enumerate(waveform):
            time.append(single_point[0])
            amplitude.append(single_point[1])

    time = np.array(time)
    time *= 10**9
    amplitude = np.array(amplitude)
    amplitude *= 10**3
        
    
    
    image, x_edges, y_edges = np.histogram2d(time, amplitude, bins=[400, 300])
    image = np.where(image == 0, np.full(np.shape(image), np.nan), image)
    
    fig = plt.figure(figsize=(12, 9))
    axes = fig.add_subplot()

    im = plot2d(image, x_edges, y_edges, axes, norm=matplotlib.colors.LogNorm())
    axes.set_xlabel('time[ns]', fontsize=22)
    axes.set_ylabel("Amplified signal[mV]", fontsize=22)
    axes.tick_params(axis='both', which='major', labelsize=18)
    axes.tick_params(axis='both', which='minor', labelsize=18)
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label("Observed Frequency", fontsize=22)
    cax.tick_params(axis='both', which='both', labelsize=18)
    fig.tight_layout()
    
    
    if plot_title == True:
        axes.set_title(title, fontsize=22)

    if save == True:
        loc = PLOTS_FOLDER
        fig.savefig(loc + savename, dpi=600)

    plt.show()


    
if __name__ == "__main__":
    all_waveforms = iterate_large_files(0, 25, "C1--850V_pmt-0047_1000--", loc="/home/todor/University/MPhys project/Data_PMT/0047/850V/")

    make_heatmap(all_waveforms, False, "pmt-0047_850V_25000waveforms.png", False)

    

