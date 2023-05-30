# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:41:48 2023

@author: eeenr
"""

# Standard Library
import os
from glob import glob
import datetime as dt
#import cmath

# Others
import pandas as pd
import numpy as np
import xarray as xr # Version 2022.6.0 is the only version where this works - maybe there's a 

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib as mpl

# Custom dictionaries should have the same format
pcasp_scattering_inputs = {
    'name'         : 'PCASP',
    'min_diameter' : 0.05,
    'max_diameter' : 8,
    'diameter_res' : 0.001,
    'min_primary'  : 35,
    'max_primary'  : 120,
    'min_secondary': 60,
    'max_secondary': 145,
    'wavelength'   : 0.6328
    }
cdp_scattering_inputs = {
    'name'         : 'CDP',
    'min_diameter' : 1,
    'max_diameter' : 100,
    'diameter_res' : 0.1,
    'min_primary'  : 1.7,
    'max_primary'  : 14,
    'min_secondary': None,
    'max_secondary': None,
    'wavelength'   : 0.658
    }


def kwarg_handling(ax, axes_defaults, user_kwargs, ignore_lims=False):
    """
    ax: matplotlib axes
        Axes to plot on
    axes_defaults: dict
        Dictionary of default kwargs for the plot
    user_kwargs: dict
        User-defined kwarg dictionary for the plot
    ignore_lims: bool (default False)
        Ignore limit kwargs (needed when limits are not appropriate to be pre-chosen)
    """
    if not ignore_lims:
        if 'xlim' in user_kwargs:
            ax.set_xlim(user_kwargs['xlim'])   
        else:
            ax.set_xlim(axes_defaults['xlim'])
        if 'ylim' in user_kwargs:
            ax.set_ylim(user_kwargs['ylim'])   
        else:
            ax.set_ylim(axes_defaults['ylim'])
    if 'xlabel' in user_kwargs:
        ax.set_xlabel(user_kwargs['xlabel'])
    else:
        ax.set_xlabel(axes_defaults['xlabel'])
    if 'ylabel' in user_kwargs:
        ax.set_ylabel(user_kwargs['ylabel'])
    else:
        ax.set_ylabel(axes_defaults['ylabel'])
    for kw in ('xlim', 'ylim', 'xlabel', 'ylabel'):
        user_kwargs.pop(kw, None)
    return ax, user_kwargs

def add_item_to_command(command, item, value):
    try:
        command = command + ' -' + item + ' ' + str(value)
        return command
    except:
        print(item + ' not in input dictionary')
        exit



def generate_scattering_table(ri, folder, mcs_loc, input_dict, silent=False):
    """Generate Mie Scattering data interacting with the MieConScat program.

    Using default values for the PCASP or CDP probe, use Phil Rosenberg's code to generate Mie
    Scattering data for a particular refractive index. You can also specify a 'Custom' instrument
    with a custom dictionary of instrument parameters.

    Parameters
    ----------
    ri: complex number
        Complex number in the standard Python form RE+IMj (e.g. 1.58-0.03j).
    folder: string
        string for the folder location where the files will be output. Should not end in \\
    mcs_loc: str
        location of MieConScat
    custom_input: dict (default None)
        Dictionary of input parameters in the format of the defaults for CDP/PCASP
    silent: bool (default False)
        silence the commands you're passing to the CL (not recommended)

    Returns
    -------
    none
    """
    try:
        command = mcs_loc + ' -wav ' + str(input_dict['wavelength'])
    except:
        print('Wavelength not in input dictionary')
    
    # Build command to run the scattering program.
    command = add_item_to_command(command, 'dmin', input_dict['min_diameter'])
    command = add_item_to_command(command, 'dmax', input_dict['max_diameter'])
    command = add_item_to_command(command, 'dint', input_dict['diameter_res'])
    command = add_item_to_command(command, 'rerimin', ri.real)
    command = add_item_to_command(command, 'imrimin', ri.imag)
    command = add_item_to_command(command, 'ang1min', input_dict['min_primary'])
    command = add_item_to_command(command, 'ang1max', input_dict['max_primary'])
    if input_dict['min_secondary'] != None:
        command = add_item_to_command(command, 'ang2min', input_dict['min_secondary'])
        command = add_item_to_command(command, 'ang2max', input_dict['max_secondary'])
    filename = 'scattering_' + input_dict['name'] + '_' + str(ri)[1:-1] + '.csv'
    command = command + ' ' + folder + '\\' + filename
    if silent == False:
        print(command)
    os.system(command)
    return filename

def create_higher_order_scatter_data(folder,csv_name):
    """Create surface area and particle volume channel cross-section data.

    Parameters
    ----------
    folder: string
        name of the folder containing scattering data
    csv_name: string
        name of the diameter CSV to process
    """

    # Read in data to pandas df and create area and volume columns
    scat_data = pd.read_csv(os.path.join(folder,csv_name), header=5)
    ri_str = list(scat_data.columns)[1]
    scat_data['area'] = np.pi * scat_data['Diameter']**2 # microns^2
    scat_data['vol'] = (np.pi/6) * scat_data['Diameter']**3 # microns^3 (what a lovely unit!)
    area_path = os.path.join(folder,csv_name)[:-4] + '_area.csv'
    vol_path = os.path.join(folder,csv_name)[:-4] + '_volume.csv'

    # Output data to csvs and add headers so they are readable by the CStoDConverter
    scat_data.to_csv(area_path, columns=['area',ri_str],header=['Diameter',ri_str],index=False)
    scat_data.to_csv(vol_path, columns=['vol',ri_str],header=['Diameter',ri_str],index=False)

    scatter_header = open(os.path.join(folder,csv_name),'r').read().split('\n')[:5]
    # Note that this will leave the headers as "Diameter" - this is not correct but needs
    # to be to keep the program working
    for path in [area_path, vol_path]:
        with open(path,'r') as contents:
            save = contents.read()
        with open(path,'w') as contents:
            for line in scatter_header:
                contents.write(line + '\n')
            contents.write(save)
    return
    
def run_CStoD(faam_calibration_file, mie_csvs, cstod_loc, output_folder, silent=False):
    """Run the CStoDConverter program from the the command line.

    Parameters
    ----------
    faam_calibration_file: string
        location of the calibration file provided by FAAM for the given instrument
    mie_csvs: list of strings
        list of CSV filenames
    output_folder: string
        folder where the diameter data is output to
    silent: bool, default False
        if true, do not print commands sent to command line.
    
    """
    for csv in mie_csvs:
        command = cstod_loc + ' ' + faam_calibration_file + ' ' + csv + ' '
        mie_csv_name = os.path.basename(csv)[10:]
        output_csv_name = 'channel_data' + mie_csv_name
        command = command + os.path.join(output_folder, output_csv_name)
        if not silent:
            print(command)
        os.system(command)
    return
        

def get_refractive_indices(folder, instrument_name):
    """Get a list of refractive indices chosen
    
    This assumes that you have diam, area and vol calibrations. RI are stored as complex128 objects.
    While this works with lots of standard Python stuff, for some things you will need to convert to
    strings, a Pandas MultiIndex in [real, im] form or a custom class.

    Parameters
    ----------
    folder: string
        Folder with Mie scattering data
    instrument_name: string
        Name of the instrument
    Returns
    -------
    list of complex128 objects
        Refractive indices as complex numbers
    """
    csv_names = [os.path.basename(p) for p in glob(os.path.join(folder,'*j.csv'))]
    # strip useless data from names and convert to complex number using eval
    start_char = 14+len(instrument_name)
    ref_indices = [eval(n[start_char:-4]) for n in csv_names]
    return ref_indices

def read_channel_data_csv(csv, no_channels=30):
    """Read the channel data csvs produced by CStoDConverter

    Parameters
    ----------
    csv: string
        filename of the CSV to read (output from CStoDConverter)
    no_channels: int (default 30)
        Number of channels 

    """
    
    df = pd.read_csv(csv, skiprows=3, header=None).T
    df.columns = df.iloc[0]
    df.drop(0,inplace=True) # note use of inplace is very powerful and rarely best practice
                            # do not use inplace when developing code
    df.drop(no_channels+1,inplace=True)
    return df    

def produce_df_for_each_ri(unique_ris, channel_data_folder):
    """Produce a dataframe containing channel data for each refractive index.

    Reads in the channel data and produces a dataframe containing linear and logarithmic
    data for each refractive index's channel diameter, area and volume. Data are renamed
    to share their names with those in the FAAM data on CEDA.

    Parameters
    ----------
    unique_ris: list of complex128 objects
        Complex refractive indices investigated
    channel_data_folder: string
        Folder containing channel data (note NOT the Mie Scattering folder)

    Returns
    -------
    dict {complex128: pandas DataFrame}
        dict. where the keys are each ref. index and the values are dataframes with channel data
    """
    ris_and_data = {}
    for ri in unique_ris:
        test_ri_str = str(ri)[1:-1]
        csvs_with_particular_ri = glob(os.path.join(channel_data_folder, '*'+test_ri_str+'*'))
        one_ri_dfs = {
            'diam_df': read_channel_data_csv(csvs_with_particular_ri[0]),
            'area_df': read_channel_data_csv(csvs_with_particular_ri[1]),
            'vol_df': read_channel_data_csv(csvs_with_particular_ri[2])
        }
        names = ['diameter', 'area', 'volume']
        for i, df in enumerate(one_ri_dfs):
            one_ri_dfs[df].rename(columns={
                'Channel Centre': names[i] + '_centre',
                'Channel Centre Errors': names[i] + '_centre_err',    
                'Channel Widths': names[i] + '_width',
                'Channel Width Errors': names[i] + '_width_err',
                'Channel Logarithmic Centre': names[i] + '_log_centre',
                'Channel Logarithmic Centre Errors': names[i] + '_log_centre_err',    
                'Channel Logarithmic Widths': names[i] + '_log_width',
                'Channel Logarithmic Width Errors': names[i] + '_log_width_err'
                }, inplace=True) # see previous note about use of inplace
            one_ri_dfs[df].drop([
                'Lower Cross Section Boundaries',
                'Lower Cross Section Boundary Errors',
                'Upper Cross Section Boundaries',
                'Upper Cross Section Boundary Errors',
                'Width of Cross Section Boundaries',
                'Width of Cross Section Boundary Errors'
                ], axis=1, inplace=True)
            one_ri_dfs[df].index.name = 'bin'
        combined_single_ri_df = pd.concat([one_ri_dfs['diam_df'],one_ri_dfs['area_df'],one_ri_dfs['vol_df']], axis=1) 
        ris_and_data[ri] = combined_single_ri_df
    return ris_and_data

def produce_data_var_dicts(ris_and_data):
    """Rearrange dataframes to produce a data variable dictionary ready for xarray

    Take the dictionary produced by *produce_df_for_each_ri* and rearrange it so that
    each variable (diameter, area, volume/lin and log) has its own dataframe containing
    a column for each refractive index.

    Parameters
    ----------
    ris_and_data: dict
        For format, see docstring of *produce_df_for_each_ri*
    
    Returns
    -------
    dict {string: pandas DataFrame}
        dict where variable names are keys and values are data from each ref index associated
        with the variable contained in a dataframe
    """
    variable_names = list(ris_and_data[list(ris_and_data.keys())[0]].columns)
    all_vars = {}
    for v in variable_names:
        variable = pd.DataFrame()
        for ri in ris_and_data.keys():
            variable[ri] = ris_and_data[ri][v]
        all_vars[v] = variable
    return all_vars

def make_calibration_xarray_object(data_variables, scattering_inputs, atts_file):
    """Create the xarray object containing calibration data for an instrument.

    Takes data variables produced by *produce_data_var_dicts* and creates an xarray object for
    easy access and flexible use of these. Metadata for the dataset is produced from the
    scattering data inputs that are used for running MieConScat (way up near the start of
    the notebook!). Optionally (but recommended), metadata for each of the channel data variables
    can be read in from an attributes file supplied with this notebook. It is not produced
    directly due to the length of this file.

    Parameters
    ----------
    data_variables: dict
        For format, see docstring of *produce_data_var_dicts*
    scattering_inputs: dict
        Dictionary of input parameters in the format of the defaults for CDP/PCASP
    atts_file: string
        String containing file location of metadata for channel data variables. Can be None.

    Returns
    -------
    xarray DataSet
        Dataset containing all data from the calibration of an instrument.
    """
    ris = list(data_variables['diameter_centre'].columns)
    for var in data_variables:
        # Create tuple required for xarray dataset creation
        if atts_file == None:
            data_variables[var] = (["bin","refractive_index"],data_variables[var].to_numpy(dtype=np.float64))
        else:
            # Add data attributes from file if supplied
            atts = eval(open(atts_file).read())
            data_variables[var] = (["bin","refractive_index"],data_variables[var].to_numpy(dtype=np.float64),atts[var])
    
    calibration_data = xr.Dataset(
        data_vars=data_variables,
        coords = dict(
            refractive_index = ris,
            bin = np.arange(1,31,dtype=float)
        ),
        attrs=dict(
            title = 'Channel calibration data for the ' + scattering_inputs['name'] + ' instrument',
            comment = 'Calibration of the size bins of the ' +  scattering_inputs['name'] + \
            '. Data for these in log form ' + \
            'and for higher power bins (area, volume) are contained within. This has been designed to ' + \
            'match the FAAM NetCDF PCASP calibration format as closely as is useful so that functions ' + \
            'can be applied to these with some modifications by the user.',
            references = 'P.D. Rosenberg, A.R. Dean, P.I. Williams, J.R. Dorsey, A. Minikin, M.A. Pickering ' +\
            'and A. Petzold, Particle sizing calibration with refractive index correction for light scattering '+\
            'optical particle counters and impacts upon PCASP and CDP data collected during the Fennec campaign, '+\
            'Atmos. Meas. Tech., 5, 1147-1163, doi:10.5194/amt-5-1147-2012, 2012.',
            instrument_primary_collection_angles = str(scattering_inputs['min_primary']) + '-' +\
                str(scattering_inputs['max_primary']) + ' deg',
            instrument_secondary_collection_angles = str(scattering_inputs['min_secondary']) + '-' +\
                str(scattering_inputs['max_secondary']) + ' deg',
            instrument_wavelength = str(scattering_inputs['wavelength']) + ' um'
        )
    )
    return calibration_data


def produce_calibration_dataset(channel_data_folder, scattering_inputs, atts_file = None):
    """Wrapper function loading in all channel data produced by CStoDConverter.

    Parameters
    ----------
    channel_data_folder: string
        folder containing the channel data
    scattering_inputs: dict
        Dictionary of input parameters in the format of the defaults for CDP/PCASP
    atts_file: string
        String containing file location of metadata for channel data variables. Can be None.
    
    Returns
    -------
    xarray DataSet
        Dataset containing all data from the calibration of an instrument.
    """
    unique_ris = get_refractive_indices(channel_data_folder, scattering_inputs['name'])
    ris_and_data = produce_df_for_each_ri(unique_ris, channel_data_folder)
    dv = produce_data_var_dicts(ris_and_data)
    calibration_data = make_calibration_xarray_object(dv, scattering_inputs, atts_file)
    return calibration_data

def get_data(core_cloud_data_filename, core_data_filename):
    """ Retrieve all data for a specific run from NetCDF files.

    This assumes a file structure specified above.

    Parameters
    ----------
    core_cloud_data_filename: string
        filename of core cloud data for a particular flight
    core_data_filename: string
        filename of core cloud data for a particular flight

    Returns
    -------
    xarray Dataset
        all core data from the flight including cloud phys in one Dataset
    """

    pcasp = xr.open_dataset(core_cloud_data_filename,group='pcasp', engine='netcdf4')
    cdp = xr.open_dataset(core_cloud_data_filename, group='cdp', engine='netcdf4')
    core = xr.open_dataset(core_data_filename, decode_times=False, engine='netcdf4')
    core_cloud = xr.open_dataset(core_cloud_data_filename, engine='netcdf4')

    pcasp['time'] = core['Time'].values
    cdp['time'] = core['Time'].values
    core_cloud['time'] = core['Time'].values
    core = core.rename_dims(Time='time')
    core = core.rename(Time='time')
    data = xr.merge([core,core_cloud,cdp,pcasp],compat="no_conflicts")
    
    return data

def create_single_dimension(da):
    """Take measurement with frequency >1Hz and compress dimensions.
    
    Measurements with frequency N are generated in Nxtime arrays. This makes
    one time dimension with correct interplolation of times.
    
    Parameters
    ----------
    da: xarray DataArray
        Measurement originally from fl_data.
        Will not work with anything from cl_data.
    
    Returns
    -------
    xarray DataArray
        Flattened DataArray
    """
    if len(da.dims) == 2:
        new_da = da.stack(new_dim=("time", da.dims[1]))
        new_da['new_dim'] = new_da['new_dim'][:]['time'] + \
            new_da['new_dim'][:][da.dims[1]]/int(da.dims[1][3:])
        new_da = new_da.reset_index("time", drop=True)
        new_da = new_da.reset_index(da.dims[1], drop=True)
        new_da = new_da.rename(new_dim='time')
    else:
        print('Existing resolution only 1Hz')
        new_da = da
    return new_da

def convert_time_to_spm(time_str):
    """Get seconds past midnight and return it as an integer.
    
    Parameters
    ----------
    time_str: string
        Time in HH:MM:SS format as string
    
    Returns
    -------
    int
        Seconds past midnight
    """
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def time_slice_data(starts, ends, da):
    """Make slice of a measurement array based on time.
    
    Slices measurement so only time during which the filter is exposed is considered.
    One slice will be made with start and end times of clip specified by start and end list.
    If there are no pauses, there should be one time in each list. If there are, e.g.
    two pauses, you will need three times in each of the starts and ends lists.
    
    Parameters
    ----------
    starts: list of strings, ints or floats
        List of clip start times in seconds past midnight or as strings
    ends: list of strings, ints or floats
        List of clip end times in seconds past midnight or as strings
    da: xarray DataArray
        measurement to be sliced
    
    Returns
    -------
    xarray DataArray
        single array with correct time span
    """
    if len(starts) != len(ends):
        raise TypeError("To make a time slice, the number of start times must equal the number of end times")

    slices = []
    for i in range(len(starts)):
        start = convert_time_to_spm(starts[i]) if type(starts[i])==str else starts[i]
        end = convert_time_to_spm(ends[i]) if type(ends[i])==str else ends[i]
        slices.append(slice(start,end))
    sliced_da = xr.concat([da.sel(time=s) for s in slices], dim='time')
    return sliced_da

# Standard temperature and pressure, should be checked if changing standards.
def get_TP_correction(all_data, p_std=1013.25, t_std=273.15):
    """Perform temperature correction on a measurement.
    
    Uses the Boyle's law and Charles' law to adjust concentrations to standard litres.
    Temperature measurements are taken from the deiced temperature to be consistent
    with processing for PCASP and CDP data.
    
    Parameters
    ----------
    all_data: xarray Dataset
        Merged dataset containing core and cloud physics data.
    p_std: float
        Standard pressure in Hectopascals
    t_std: float
        Standard temperature in Kelvin

    Returns
    -------
    xarray DataArray
        array of floats providing the multiplicative correction factor.
    """
    pres = create_single_dimension(all_data.PS_RVSM) # Pressure from aircraft RSVM
    di_temp = create_single_dimension(all_data.TAT_DI_R) # Deiced temperature
    correction_factor = (p_std*di_temp)/(t_std*pres)
    return correction_factor

def get_pcasp_data_for_leg(flight_data, run_start_times, run_end_times):
    """Retrieve particle size distribution data (psd) from the PCASP dataset.

    Provides two time-clipped versions of the psd - one with the temperature and pressure
    correction applied and another without. Note that despite the name, the psd is not a
    ready-to-go dN/dlogD - it is merely dN using a volumetric version of the flow. IMPORTANT:
    The psd comes measured in L-1 and is adjusted to cm-3 here. You should double-check this
    is the convention used with your own datasets.

    Parameters
    ----------
    flight_data
        all core data from the flight including cloud phys in one Dataset
    run_start_times: list of strings, ints or floats
        List of clip start times in seconds past midnight or as strings
    ends: list of strings, ints or floats
        List of clip end times in seconds past midnight or as strings

    Returns
    -------
    xarray DataArray
        particle size distribution data without the temp/pressure adjustment applied
    xarray DataArray
        particle size distribution with the temp/pressure adjustment applied
    xarray DataArray
        flow rate through the instrument, used to calculate the number of actual particle counts
    """
    corr_factor = get_TP_correction(flight_data)
    corr_factor = corr_factor.rolling(time=32, center=True).mean()
    with xr.set_options(keep_attrs=True):
        uncorrected_psd = time_slice_data(run_start_times, run_end_times, flight_data.pcasp_conc_psd/1000)
        corrected_psd = flight_data.pcasp_conc_psd*corr_factor/1000 # factor of 1000 to convert to cm-3
    corrected_psd = time_slice_data(run_start_times, run_end_times, corrected_psd)
    pcasp_flow = time_slice_data(run_start_times, run_end_times, flight_data['pcasp_flow'])
    bin_array = np.arange(1, 31, dtype=float)
    corrected_psd = corrected_psd.rename({'pcasp_bin_centre': 'bin'})
    corrected_psd.coords['bin'] = ('bin', bin_array)
    uncorrected_psd = uncorrected_psd.rename({'pcasp_bin_centre': 'bin'})
    uncorrected_psd.coords['bin'] = ('bin', bin_array)
    return uncorrected_psd, corrected_psd, pcasp_flow

def get_cdp_data_for_leg(flight_data, run_start_times, run_end_times, CDP_sample_area=0.00199):
    """Retrieve particle size distribution data (psd) from the CDP dataset.

    Provides two time-clipped versions of the psd - one with the temperature and pressure
    correction applied and another without. Note that despite the name, the psd is not a
    ready-to-go dN/dlogD - it is merely dN using a volumetric version of the flow. IMPORTANT:
    The psd comes measured in L-1 and is adjusted to cm-3 here. You should double-check this
    is the convention used with your own datasets. 

    Parameters
    ----------
    flight_data
        all core data from the flight including cloud phys in one Dataset
    run_start_times: list of strings, ints or floats
        List of clip start times in seconds past midnight or as strings
    run_start_ends: list of strings, ints or floats
        List of clip end times in seconds past midnight or as strings
    CDP_sample_area: float
        Sampling cross-sectional area of the CDP. In cm^2.

    Returns
    -------
    xarray DataArray
        particle size distribution data without the temp/pressure adjustment applied
    xarray DataArray
        particle size distribution with the temp/pressure adjustment applied
    xarray DataArray
        flow rate through the instrument, used to calculate the number of actual particle counts
    """
    corr_factor = get_TP_correction(flight_data)
    corr_factor = corr_factor.rolling(time=32, center=True).mean()
    with xr.set_options(keep_attrs=True):
        uncorrected_psd = time_slice_data(run_start_times, run_end_times, flight_data.cdp_conc_psd/1000)
        corrected_psd = flight_data.cdp_conc_psd*corr_factor/1000 # factor of 1000 to convert to cm-3
    corrected_psd = time_slice_data(run_start_times, run_end_times, corrected_psd)
    # Generate CDP flow in cm^3 s^-1 by multiplying true air speed (in m/s) by 100 to convert to cm/s
    # and the CDP sampling area (in cm^2)
    true_air_speed = create_single_dimension(flight_data['TAS_RVSM'])
    cdp_flow = time_slice_data(run_start_times, run_end_times,
        true_air_speed*100*CDP_sample_area)
    bin_array = np.arange(1, 31, dtype=float) 
    corrected_psd = corrected_psd.rename({'cdp_bin_centre': 'bin'})
    corrected_psd.coords['bin'] = ('bin', bin_array)
    uncorrected_psd = uncorrected_psd.rename({'cdp_bin_centre': 'bin'})
    uncorrected_psd.coords['bin'] = ('bin', bin_array)
    return uncorrected_psd, corrected_psd, cdp_flow

# Lambda functions for flexible bin merging operations
la_add = lambda x,y: x + y
la_avg = lambda x,y: (x+y)/2
la_quad = lambda x,y: (x**2 + y**2)**(0.5)
la_log_avg = lambda x,y: np.log10((10**x + 10**y)/2)

def bin_merger(da, func):
    """An engine for the merging of bins according to a specificed operation.
    
    Hard-coded to merge bins 5 and 6. TODO: Generalise.

    Parameters
    ----------
    da: xarray DataArray
        DataArray with a bin dimension (and bins 5, 6, 15 and 16 present)
    func: lambda function
        Lambda function combining bin 5 an 6 data.
    
    Returns
    -------
    xarray DataArray
        DataArray with appropriate bins merged.
    """
    bin_5 = da.sel(bin=5)
    bin_6 = da.sel(bin=6)
    bin_15 = da.sel(bin=15)
    bin_16 = da.sel(bin=16)
    bin_5_6 = func(bin_5, bin_6)
    bin_15_16 = func(bin_15, bin_16)
    bin_5_6['bin'] = 5.5
    bin_15_16['bin'] = 15.5
    da = xr.concat([da.sel(bin=slice(0, 4)),
            bin_5_6,
            da.sel(bin=slice(7, 14)),
            bin_15_16,
            da.sel(bin=slice(17, None))], dim='bin')
    return da

def merge_pcasp_bins(corrected_psd, uncorrected_psd, cal_at_ri):
    """Merge the PCASP bins 15 and 16. Uses the engine function bin_merger.

    Parameters
    ----------
    corrected_psd: xarray DataArray
        particle size distribution corrected for T and P
    uncorrected_psd: xarray DataArray
        particle size distribution uncorrected for T and P
    cal_at_ri: xaray Dataset
        calibration data for a particular refractive index
    
    Returns
    -------
    xarray DataArray
        particle size distribution corrected for T and P, with appropriate bins merged.
    xarray DataArray
        particle size distribution uncorrected for T and P, with appropriate bins merged.
    xaray Dataset
        calibration data for a particular refractive index, with appropriate bins merged.
    """
    merged_psds = []
    for psd in (corrected_psd, uncorrected_psd):
        psd = bin_merger(psd, la_avg)
        merged_psds.append(psd)
    corrected_psd, uncorrected_psd = merged_psds
    to_add = ['diameter_log_width']
    to_avg = ['diameter_centre', 'area_centre', 'volume_centre']
    to_log_avg = ['diameter_log_centre']
    to_add_in_quad = ['diameter_log_width_err', 'diameter_log_centre_err', 'diameter_centre_err',
        'area_centre_err', 'volume_centre_err']
    added = {}
    averaged = {}
    log_averaged = {}
    added_in_quad = {}
    for k in to_add:
        added[k] = bin_merger(cal_at_ri[k], la_add)
    for k in to_avg:
        averaged[k] = bin_merger(cal_at_ri[k], la_avg)
    for k in to_log_avg:
        averaged[k] = bin_merger(cal_at_ri[k], la_log_avg)
    for k in to_add_in_quad:
        added_in_quad[k] = bin_merger(cal_at_ri[k], la_quad)
    reduced_cal_ds = xr.Dataset({**added, **averaged, **log_averaged, **added_in_quad})
    return corrected_psd, uncorrected_psd, reduced_cal_ds

def get_psd_x_axis_plotting_data(cal_at_ri):
    """Get data used for plotting (x-axis data) from the calibration data.

    Produces an xarray Dataset with linearised logarithmic channel diameters and their errors, as
    well as area and volume scaled to diameter, and their corresponding errors. Errors are given
    as "lower" and "upper" errors, ready to be used in the ax.errorbar() function.

    Parameters
    ----------
    cal_at_ri: xarray Dataset
        Dataset containing channel calibration data at a specific refractive index
    
    Returns
    -------
    xarray Dataset
        Dataset containing x-axis information including errors for the plotting of psd
    """

    # For plotting dN/dlogD produce log data in linear form.
    lin_log_diameters = 10**cal_at_ri['diameter_log_centre']
    lin_log_diameter_lb = 10**(cal_at_ri['diameter_log_centre']-cal_at_ri['diameter_log_centre_err'])
    lin_log_diameter_ub = 10**(cal_at_ri['diameter_log_centre']+cal_at_ri['diameter_log_centre_err'])
    lin_log_diam_lower_error = lin_log_diameters - lin_log_diameter_lb
    lin_log_diam_upper_error = lin_log_diameter_ub - lin_log_diameters
    

    # For plotting dS/dlogD convert area channel data to diameter
    area_as_diameter = (4*cal_at_ri['area_centre']/np.pi)**(0.5)
    area_as_diameter_lb = (4*(cal_at_ri['area_centre'] - cal_at_ri['area_centre_err'])/np.pi)**(0.5)
    area_as_diameter_ub = (4*(cal_at_ri['area_centre'] + cal_at_ri['area_centre_err'])/np.pi)**(0.5)
    area_as_diam_lower_error = area_as_diameter - area_as_diameter_lb
    area_as_diam_upper_error = area_as_diameter_ub - area_as_diameter

    # For plotting dV/dlogD convert volume channel data to diameter
    vol_as_diameter = (6*cal_at_ri['volume_centre']/np.pi)**(1/3.)
    vol_as_diameter_lb = (6*(cal_at_ri['volume_centre'] - cal_at_ri['volume_centre_err'])/np.pi)**(1/3.)
    vol_as_diameter_ub = (6*(cal_at_ri['volume_centre'] + cal_at_ri['volume_centre_err'])/np.pi)**(1/3.)
    vol_as_diam_lower_error = vol_as_diameter - vol_as_diameter_lb
    vol_as_diam_upper_error = vol_as_diameter_ub - vol_as_diameter

    plotting_data = xr.Dataset({
        'lin_log_diameter': lin_log_diameters,
        'lin_log_diam_lower_error': lin_log_diam_lower_error,
        'lin_log_diam_upper_error': lin_log_diam_upper_error,
        'area_as_diameter': area_as_diameter,
        'area_as_diam_lower_error': area_as_diam_lower_error,
        'area_as_diam_upper_error': area_as_diam_upper_error,
        'vol_as_diameter': vol_as_diameter,
        'vol_as_diam_lower_error': vol_as_diam_lower_error,
        'vol_as_diam_upper_error': vol_as_diam_upper_error,
    })
    return plotting_data

def get_log_psds(cal_at_ri, uncorrected_psd, corrected_psd, instrument_flow, bin_merging=False):
    """Produce logaritmic particle size distribution data for number, size and volume.

    Produces a dataset with all information required to plot a particle size distribution, including
    errors in both the x-axis and y-axis. Should the user choose, this can be saved as a CSV for
    plotting with external packages by performing
    log_psds.to_dataframe().to_csv('csv_name')

    Parameters
    ----------
    cal_at_ri: xarray Dataset
        all calibration data for a particular refractive index
    uncorrected_psd: xarray DataArray
        particle size distribution uncorrected for T and P
    corrected_psd: xarray DataArray
        particle size distribution corrected for T and P
    filter_flow: xarray DataArray
        flow rate through the filter, used to calculate the number of actual particle counts
    bin_merging: bool (default False)
        when true, bins pairs (5 and 6) and (15 and 16) are merged. Use when using the PCASP.

    Returns
    -------
    xarray Dataset
        Dataset containing all info. for the plotting of particle size distributions with errors.
    """
    if bin_merging:
        corrected_psd, uncorrected_psd, cal_at_ri = \
            merge_pcasp_bins(corrected_psd, uncorrected_psd, cal_at_ri)

    dN = corrected_psd
    dS = dN * cal_at_ri['area_centre']
    dV = dN * cal_at_ri['volume_centre']
    dNdlogD = corrected_psd/cal_at_ri['diameter_log_width']
    dSdlogD = dNdlogD * cal_at_ri['area_centre']
    dVdlogD = dNdlogD * cal_at_ri['volume_centre']

    # dN error calculation
    duration = 1
    counts_for_psd = uncorrected_psd/instrument_flow/(duration)
    counting_error = np.sqrt(counts_for_psd.mean(dim='time'))

    # dS and dV errors
    dS_err = ((dS**2) * ((counting_error/corrected_psd)**2 +
       (cal_at_ri['area_centre_err']/cal_at_ri['area_centre'])**2))**(0.5)
    
    dV_err = ((dV**2) * ((counting_error/corrected_psd)**2 +
        (cal_at_ri['volume_centre_err']/cal_at_ri['volume_centre'])**2))**(0.5)


    # log distribution errors
    dNdlogD_sq_err = (dNdlogD**2) * ((counting_error/corrected_psd)**2 +
    (cal_at_ri['diameter_log_width_err']/cal_at_ri['diameter_log_width'])**2)
    dNdlogD_err = dNdlogD_sq_err**(0.5)

    dSdlogD_sq_err = (dSdlogD**2) * ((counting_error/corrected_psd)**2 +
    (cal_at_ri['diameter_log_width_err']/cal_at_ri['diameter_log_width'])**2 +
    (cal_at_ri['area_centre_err']/cal_at_ri['area_centre'])**2)
    dSdlogD_err = dSdlogD_sq_err**(0.5)

    dVdlogD_sq_err = (dVdlogD**2) * ((counting_error/corrected_psd)**2 +
    (cal_at_ri['diameter_log_width_err']/cal_at_ri['diameter_log_width'])**2 +
    (cal_at_ri['volume_centre_err']/cal_at_ri['volume_centre'])**2)
    dVdlogD_err = dVdlogD_sq_err**(0.5)
    
    plotting_data = get_psd_x_axis_plotting_data(cal_at_ri)


    log_psds = xr.Dataset({
        'dN':           dN,
        'dN_err':       counting_error,
        'dS':           dS,
        'dS_err':       dS_err,
        'dV':           dV,
        'dV_err':       dV_err,
        'dNdlogD':      dNdlogD,
        'dNdlogD_err':  dNdlogD_err,
        'dSdlogD':      dSdlogD,
        'dSdlogD_err':  dSdlogD_err,
        'dVdlogD':      dVdlogD,
        'dVdlogD_err':  dVdlogD_err,
        'diameter_log_width': cal_at_ri['diameter_log_width']
        }
    )

    log_psds = xr.merge([log_psds, plotting_data])

    return log_psds

def get_mean_log_psds(cal_at_ri, uncorrected_psd, corrected_psd, instrument_flow, bin_merging=False):
    """Produce logaritmic particle size distribution data for number, size and volume.

    Produces a dataset with all information required to plot a particle size distribution, including
    errors in both the x-axis and y-axis. Should the user choose, this can be saved as a CSV for
    plotting with external packages by performing
    log_psds.to_dataframe().to_csv('csv_name')

    Parameters
    ----------
    cal_at_ri: xarray Dataset
        all calibration data for a particular refractive index
    uncorrected_psd: xarray DataArray
        particle size distribution uncorrected for T and P
    corrected_psd: xarray DataArray
        particle size distribution corrected for T and P
    filter_flow: xarray DataArray
        flow rate through the filter, used to calculate the number of actual particle counts
    bin_merging: bool (default False)
        when true, bins pairs (5 and 6) and (15 and 16) are merged. Use when using the PCASP.

    Returns
    -------
    xarray Dataset
        Dataset containing all info. for the plotting of particle size distributions with errors.
    """
    if bin_merging:
        corrected_psd, uncorrected_psd, cal_at_ri = \
            merge_pcasp_bins(corrected_psd, uncorrected_psd, cal_at_ri)
    filter_leg_mean_psd = corrected_psd.mean(dim='time',skipna=True)

    dN = filter_leg_mean_psd
    dS = dN * cal_at_ri['area_centre']
    dV = dN * cal_at_ri['volume_centre']
    dNdlogD = filter_leg_mean_psd/cal_at_ri['diameter_log_width']
    dSdlogD = dNdlogD * cal_at_ri['area_centre']
    dVdlogD = dNdlogD * cal_at_ri['volume_centre']

    # dN error calculation
    duration = uncorrected_psd['time'][-1]-uncorrected_psd['time'][0]
    counts_for_psd = uncorrected_psd/instrument_flow/(duration)
    counting_error = np.sqrt(counts_for_psd.mean(dim='time'))

    # dS and dV errors
    dS_err = ((dS**2) * ((counting_error/filter_leg_mean_psd)**2 +
       (cal_at_ri['area_centre_err']/cal_at_ri['area_centre'])**2))**(0.5)
    
    dV_err = ((dV**2) * ((counting_error/filter_leg_mean_psd)**2 +
        (cal_at_ri['volume_centre_err']/cal_at_ri['volume_centre'])**2))**(0.5)


    # log distribution errors
    dNdlogD_sq_err = (dNdlogD**2) * ((counting_error/filter_leg_mean_psd)**2 +
    (cal_at_ri['diameter_log_width_err']/cal_at_ri['diameter_log_width'])**2)
    dNdlogD_err = dNdlogD_sq_err**(0.5)

    dSdlogD_sq_err = (dSdlogD**2) * ((counting_error/filter_leg_mean_psd)**2 +
    (cal_at_ri['diameter_log_width_err']/cal_at_ri['diameter_log_width'])**2 +
    (cal_at_ri['area_centre_err']/cal_at_ri['area_centre'])**2)
    dSdlogD_err = dSdlogD_sq_err**(0.5)

    dVdlogD_sq_err = (dVdlogD**2) * ((counting_error/filter_leg_mean_psd)**2 +
    (cal_at_ri['diameter_log_width_err']/cal_at_ri['diameter_log_width'])**2 +
    (cal_at_ri['volume_centre_err']/cal_at_ri['volume_centre'])**2)
    dVdlogD_err = dVdlogD_sq_err**(0.5)
    
    plotting_data = get_psd_x_axis_plotting_data(cal_at_ri)


    log_psds = xr.Dataset({
        'dN':           dN,
        'dN_err':       counting_error,
        'dS':           dS,
        'dS_err':       dS_err,
        'dV':           dV,
        'dV_err':       dV_err,
        'dNdlogD':      dNdlogD,
        'dNdlogD_err':  dNdlogD_err,
        'dSdlogD':      dSdlogD,
        'dSdlogD_err':  dSdlogD_err,
        'dVdlogD':      dVdlogD,
        'dVdlogD_err':  dVdlogD_err,
        'diameter_log_width': cal_at_ri['diameter_log_width']
        }
    )

    log_psds = xr.merge([log_psds, plotting_data])

    return log_psds

def plot_dNdlogD(ax, log_psds, legend=False, **kwargs):
    """Plot logarithmic particle number distribution on an axis.

    Compatible with the matplotlib kwargs feature in this notebook.

    Parameters
    ----------
    ax: matplotlib axes
        axes to be plotted on
    log_psds: xarray Dataset
        containing all particle size distribution data
    legend: bool (default: False)
        Create a legend on the axes
    
    Returns
    -------
    matplotlib axes
        axes with plot on
    """
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(visible=True, which='major', color='gray')
    ax.grid(visible=True, which='minor', color='gray', alpha=0.25)
    ax.xaxis.set_major_formatter(formatter=mpl.ticker.ScalarFormatter())
    default_kwargs = {
        'xlim': [0.1, 50],
        'ylim': [0.01, 1000],
        'xlabel': 'Particle diameter ($\mu$m)',
        'ylabel': r'$\frac{\mathrm{d}N}{\mathrm{d}\,\log{(D)}}$ (cm$^{-3}$)'
    }
    ax, kwargs = kwarg_handling(ax, default_kwargs, kwargs)
    if len(kwargs) == 0:
        ri = str(log_psds['refractive_index'].values)[1:-1]
        ax.errorbar(log_psds['lin_log_diameter'],log_psds['dNdlogD'],
            xerr=[log_psds['lin_log_diam_lower_error'],log_psds['lin_log_diam_upper_error']],
            yerr=log_psds['dNdlogD_err'],
            linewidth=0,marker='o',elinewidth=0.5,ecolor='g',markersize=2,capsize=2,label=ri,
            markeredgecolor='gray',markerfacecolor='g',markeredgewidth=0.5,capthick=0.25)
    else:
        ax.errorbar(log_psds['lin_log_diameter'],log_psds['dNdlogD'],
            xerr=[log_psds['lin_log_diam_lower_error'],log_psds['lin_log_diam_upper_error']],
            yerr=log_psds['dNdlogD_err'], **kwargs)
    if legend:
        ax.legend()
    return ax

def plot_dSdlogD(ax, log_psds, legend=False, **kwargs):
    """Plot logarithmic particle surface area distribution on an axis.

    Compatible with the matplotlib kwargs feature in this notebook.

    Parameters
    ----------
    ax: matplotlib axes
        axes to be plotted on
    log_psds: xarray Dataset
        containing all particle size distribution data
    legend: bool (default: False)
        Create a legend on the axes
    
    Returns
    -------
    matplotlib axes
        axes with plot on
    """
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(visible=True, which='major', color='gray')
    ax.grid(visible=True, which='minor', color='gray', alpha=0.25)
    ax.xaxis.set_major_formatter(formatter=mpl.ticker.ScalarFormatter())
    default_kwargs = {
        'xlim': [0.1, 50],
        'ylim': [0.01, 1000],
        'xlabel': 'Particle diameter ($\mu$m)',
        'ylabel': r'$\frac{\mathrm{d}S}{\mathrm{d}\,\log{(D)}}$ ($\mu$m$^{2}$ cm$^{-3}$)'
    }
    ax, kwargs = kwarg_handling(ax, default_kwargs, kwargs)
    if len(kwargs) == 0:
        ri = str(log_psds['refractive_index'].values)[1:-1]
        ax.errorbar(log_psds['area_as_diameter'],log_psds['dSdlogD'],
                xerr=[log_psds['area_as_diam_lower_error'],log_psds['area_as_diam_upper_error']],
                yerr=log_psds['dSdlogD_err'],
                linewidth=0,marker='o',elinewidth=0.5,ecolor='r',markersize=2,capsize=2,label=ri,
                markeredgecolor='gray',markerfacecolor='r',markeredgewidth=0.5,capthick=0.25)
    else:
        ax.errorbar(log_psds['area_as_diameter'],log_psds['dSdlogD'],
            xerr=[log_psds['area_as_diam_lower_error'],log_psds['area_as_diam_upper_error']],
            yerr=log_psds['dSdlogD_err'], **kwargs)
    if legend:
        ax.legend()
    return ax

def plot_dVdlogD(ax, log_psds, legend=False, **kwargs):
    """Plot logarithmic particle volume distribution on an axis.

    Compatible with the matplotlib kwargs feature in this notebook.

    Parameters
    ----------
    ax: matplotlib axes
        axes to be plotted on
    log_psds: xarray Dataset
        containing all particle size distribution data
    legend: bool (default: False)
        Create a legend on the axes
    
    Returns
    -------
    matplotlib axes
        axes with plot on
    """
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(visible=True, which='major', color='gray')
    ax.grid(visible=True, which='minor', color='gray', alpha=0.25)
    ax.xaxis.set_major_formatter(formatter=mpl.ticker.ScalarFormatter())
    default_kwargs = {
        'xlim': [0.1, 50],
        'ylim': [0.01, 1000],
        'xlabel': 'Particle diameter ($\mu$m)',
        'ylabel': r'$\frac{\mathrm{d}V}{\mathrm{d}\,\log{(D)}}$ ($\mu$m$^{3}$ cm$^{-3}$)'
    }
    ax, kwargs = kwarg_handling(ax, default_kwargs, kwargs)
    if len(kwargs) == 0:
        ri = str(log_psds['refractive_index'].values)[1:-1]
        ax.errorbar(log_psds['vol_as_diameter'],log_psds['dVdlogD'],
            xerr=[log_psds['vol_as_diam_lower_error'],log_psds['vol_as_diam_upper_error']],
            yerr=log_psds['dVdlogD_err'], 
            linewidth=0,marker='o',elinewidth=0.5,ecolor='b',markersize=2,capsize=2,label=ri,
            markeredgecolor='gray',markerfacecolor='b',markeredgewidth=0.5,capthick=0.25)
    else:
        ax.errorbar(log_psds['vol_as_diameter'],log_psds['dVdlogD'],
            xerr=[log_psds['vol_as_diam_lower_error'],log_psds['vol_as_diam_upper_error']],
            yerr=log_psds['dVdlogD_err'], **kwargs)
    if legend:
        ax.legend()
    return ax


def plot_total_number_concentration(ax, corrected_psd, flag=None, **kwargs):
    """Plot total number concentrations for either the PCASP or CDP probe.
    
    Kwargs implemented for concentration axis only
    
    Parameters
    ----------
    axes: list of matplotlib axes
        Must have length 3 - first axes for dN/dD plot, second for dS/dD, third for dV/dD
    corrected_psd: xarray DataArray
        particle size distribution corrected for T and P
    flag: xarray DataArray
        data flag for the plot. 
    
    Returns
    -------
    matplotlib axes
        axes plotted on
    """
    dN = corrected_psd.sum(dim='bin')

    indices_of_gaps = list(np.where(np.diff(dN.time) != 1)[0] + 1)
    plot_indices = [i for s in [[0], indices_of_gaps, [-1]] for i in s]

    x_axis_time_fmt = mdates.DateFormatter('%H:%M')
    time = [dt.datetime.fromtimestamp(val) for val in dN.time.values]


    default_kwargs = {
        'xlabel': 'Time',
        'ylabel': r'Concentration / cm$^{-3}$)'
    }
    ax, kwargs = kwarg_handling(ax, default_kwargs, kwargs, ignore_lims=True)
    ax.yaxis.label.set_color('blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.xaxis.set_major_formatter(x_axis_time_fmt)
    ax.grid()
    if len(kwargs) == 0:
        for i in range(len(plot_indices)-1):
            ax.plot(time[plot_indices[i]:plot_indices[i+1]],
                    dN[plot_indices[i]:plot_indices[i+1]], lw=0.25, color='b')
    else:
        for i in range(len(plot_indices)-1):
            ax.plot(time[plot_indices[i]:plot_indices[i+1]],
                    dN[plot_indices[i]:plot_indices[i+1]], **kwargs)        
    label = True
    vspan = None
    if len(indices_of_gaps) > 0:
        for i in indices_of_gaps:
            if label:
                vspan = ax.axvspan(time[i-1],time[i],color='gray',alpha=0.3, label='Filter shut')
                label = False
            else:
                ax.axvspan(time[i-1],time[i],color='gray',alpha=0.3)
    ax.set_ylim(0, None)
    flag_line = None
    if flag is not None:
        flag_ax = ax.twinx()
        flag_time = [dt.datetime.fromtimestamp(val) for val in flag.time.values]
        flag_ax.set_ylim(0,3)
        flag_ax.set_yticks([0,1,2,3])
        flag_ax.tick_params(axis='y', labelcolor='red')
        flag_ax.set_ylabel('Data flag', color='r')
        indices_of_gaps = list(np.where(np.diff(flag.time) != 1)[0] + 1)
        plot_indices = [i for s in [[0], indices_of_gaps, [-1]] for i in s]
        flag_mask = np.where(flag > 0.5, False, True)
        masked_flag= np.ma.masked_where(flag_mask, flag)
        label = True
        for i in range(len(plot_indices)-1):
            if label == True:
                flag_line, = flag_ax.plot(flag_time[plot_indices[i]:plot_indices[i+1]],
                                masked_flag[plot_indices[i]:plot_indices[i+1]],
                                lw=1.5, color='r', label='Flag')
                label = False
            else:
                flag_ax.plot(flag_time[plot_indices[i]:plot_indices[i+1]],
                    masked_flag[plot_indices[i]:plot_indices[i+1]], lw=1.5, color='r')
        flag_ax.grid(visible=True, which='major', color='r', alpha=0.1)
    
    legend_items = []
    if vspan is not None:
        legend_items.append(vspan)
    if flag_line is not None:
        legend_items.append(flag_line)


    
    ax.legend(legend_items, [l.get_label() for l in legend_items])
    return ax

def mask_humidity_above_threshold(psd, rh_data, rh_threshold):
    """Applies a mask to remove datapoints where relative humidity is above a threshold.

    Parameters
    ----------
    psd: xarray DataArray
        DataArray containing the particle size distribution counts from the FAAM data. Must have
        time dimension, i.e. not yet averaged out.
    rh_data: xarray DataArray
        DataArray containing relative humidity vs time data.
    rh_threshold: float
        Minimum relative humidity to exclude
    """
    rh_mask = rh_data < rh_threshold
    rh_corrected = psd.where(rh_mask,drop=True)
    return rh_corrected

def mask_humidity_above_threshold(psd, rh_data, rh_threshold):
    """Applies a mask to remove datapoints where relative humidity is above a threshold.

    Parameters
    ----------
    psd: xarray DataArray
        DataArray containing the particle size distribution counts from the FAAM data. Must have
        time dimension, i.e. not yet averaged out.
    rh_data: xarray DataArray
        DataArray containing relative humidity vs time data.
    rh_threshold: float
        Minimum relative humidity to exclude
    """
    rh_mask = rh_data < rh_threshold
    rh_corrected = psd.where(rh_mask,drop=True)
    return rh_corrected

def apply_nevzorov_mask(psd, nev_flag):
    """Applies a mask to remove datapoints where the Nevzorov flag is active.

    Parameters
    ----------
    psd: xarray DataArray
        DataArray containing the particle size distribution counts from the FAAM data. Must have
        time dimension, i.e. not yet averaged out.
    nev_flag: xarray DataArray
        DataArray containing nevzorov flag data
    """
    nev_mask = nev_flag > 0.5
    nev_corrected = psd.where(nev_mask,drop=True)
    return nev_corrected

def area_under_gap_over_time(pcasp_psd, cdp_psd, dist_str):
    """Integrate the area between the last PCASP bin and the first CDP bin.

    Integration performed by linear interpolation in logarithmic space.

    Parameters
    ----------
    pcasp_psd: xarray Dataset
        logarithmic particle size distributions for the PCASP
    cdp_psd: xarray Dataset
        logarithmic particle size distribution for the CDP
    dist_str: string
        string specifying the distribution to integrate. Must be dNdlogD, dSdlogD or dVdlogD

    Returns
    -------
    float
        area under the "gap"
    """

    log_max_pcasp_diam = np.log10(pcasp_psd['lin_log_diameter'][-1]) + pcasp_psd['diameter_log_width'][-1]/2
    log_min_cdp_diam = np.log10(cdp_psd['lin_log_diameter'][0]) -  cdp_psd['diameter_log_width'][0]/2
    log_gap_width = log_min_cdp_diam - log_max_pcasp_diam
    a_under_gap = 0.5*(pcasp_psd[dist_str].isel(bin=-1)+cdp_psd[dist_str].isel(bin=0))*log_gap_width
    return a_under_gap

def integrate_distribution_with_errors_over_time(pcasp_psd, cdp_psd):
    """
    Integrate particle size distributions with errors from a PCASP and a CDP instrument.

    Parameters
    ----------
    pcasp_psd : xarray.Dataset
        A dataset containing the particle size distribution (PSD) data from a PCASP instrument.
        It should have the following variables:
        - 'dN' (number concentration, units: cm^-3)
        - 'dS' (surface area concentration, units: cm^-3)
        - 'dV' (volume concentration, units: cm^-3)
        - 'dN_err' (error in number concentration, units: cm^-3)
        - 'dS_err' (error in surface area concentration, units: cm^-3)
        - 'dV_err' (error in volume concentration, units: cm^-3)
        It should also have a 'bin' dimension representing the size bins.

    cdp_psd : xarray.Dataset
        A dataset containing the PSD data from a CDP instrument.
        It should have the same variables and dimensions as `pcasp_psd`.

    Returns
    -------
    dN : float
        The total number concentration (units: cm^-3).
    dS : float
        The total surface area concentration (units: um^2 cm^-3).
    dV : float
        The total volume concentration (units: um^3 cm^-3).
    dN_err : float
        The error in the total number concentration (units: cm^-3).
    dS_err : float
        The error in the total surface area concentration (units: cm^-3).
    dV_err : float
        The error in the total volume concentration (units: cm^-3).

    """
    # Sum the values along the "bin" dimension and convert to floats
    pcasp_dN, pcasp_dS, pcasp_dV = [pcasp_psd.dN.sum(dim='bin'),
                                               pcasp_psd.dS.sum(dim='bin'),
                                               pcasp_psd.dV.sum(dim='bin')]
    pcasp_dN_err = np.sqrt((pcasp_psd.dN_err**2).sum(dim='bin'))
    pcasp_dS_err = np.sqrt((pcasp_psd.dS_err**2).sum(dim='bin'))
    pcasp_dV_err = np.sqrt((pcasp_psd.dV_err**2).sum(dim='bin'))

    cdp_dN, cdp_dS, cdp_dV = [cdp_psd.dN.sum(dim='bin'),
                                         cdp_psd.dS.sum(dim='bin'),
                                         cdp_psd.dV.sum(dim='bin')]
    cdp_dN_err = np.sqrt((cdp_psd.dN_err**2).sum(dim='bin'))
    cdp_dS_err = np.sqrt((cdp_psd.dS_err**2).sum(dim='bin'))
    cdp_dV_err = np.sqrt((cdp_psd.dV_err**2).sum(dim='bin'))

    # Calculate the gap values
    gap_dN, gap_dS, gap_dV = [area_under_gap_over_time(pcasp_psd, cdp_psd, 'dNdlogD'),
                                         area_under_gap_over_time(pcasp_psd, cdp_psd, 'dSdlogD'),
                                         area_under_gap_over_time(pcasp_psd, cdp_psd, 'dVdlogD')]
    
    gap_dN_err = np.sqrt(pcasp_psd['dN_err'][-1]**2 + cdp_psd['dN_err'][0]**2)
    gap_dS_err = np.sqrt(pcasp_psd['dS_err'][-1]**2 + cdp_psd['dS_err'][0]**2)
    gap_dV_err = np.sqrt(pcasp_psd['dV_err'][-1]**2 + cdp_psd['dV_err'][0]**2)

    # Calculate the final values
    dN = pcasp_dN + gap_dN + cdp_dN
    dS = pcasp_dS + gap_dS + cdp_dS
    dV = pcasp_dV + gap_dV + cdp_dV
    dN_err = np.sqrt(pcasp_dN_err**2 + gap_dN_err**2 + cdp_dN_err**2)
    dS_err = np.sqrt(pcasp_dS_err**2 + gap_dS_err**2 + cdp_dS_err**2)
    dV_err = np.sqrt(pcasp_dV_err**2 + gap_dV_err**2 + cdp_dV_err**2)
    return dN, dS, dV, dN_err, dS_err, dV_err

def area_under_gap(pcasp_psd, cdp_psd, dist_str):
    """Integrate the area between the last PCASP bin and the first CDP bin.

    Integration performed by linear interpolation in logarithmic space.

    Parameters
    ----------
    pcasp_psd: xarray Dataset
        logarithmic particle size distributions for the PCASP
    cdp_psd: xarray Dataset
        logarithmic particle size distribution for the CDP
    dist_str: string
        string specifying the distribution to integrate. Must be dNdlogD, dSdlogD or dVdlogD

    Returns
    -------
    float
        area under the "gap"
    """

    log_max_pcasp_diam = np.log10(pcasp_psd['lin_log_diameter'][-1]) + pcasp_psd['diameter_log_width'][-1]/2
    log_min_cdp_diam = np.log10(cdp_psd['lin_log_diameter'][0]) -  cdp_psd['diameter_log_width'][0]/2
    log_gap_width = log_min_cdp_diam - log_max_pcasp_diam
    a_under_gap = 0.5*(pcasp_psd[dist_str][-1]+cdp_psd[dist_str][0])*log_gap_width
    return a_under_gap.values

def integrate_distribution_with_errors(pcasp_psd, cdp_psd):
    """
    Integrate particle size distributions with errors from a PCASP and a CDP instrument.

    Parameters
    ----------
    pcasp_psd : xarray.Dataset
        A dataset containing the particle size distribution (PSD) data from a PCASP instrument.
        It should have the following variables:
        - 'dN' (number concentration, units: cm^-3)
        - 'dS' (surface area concentration, units: cm^-3)
        - 'dV' (volume concentration, units: cm^-3)
        - 'dN_err' (error in number concentration, units: cm^-3)
        - 'dS_err' (error in surface area concentration, units: cm^-3)
        - 'dV_err' (error in volume concentration, units: cm^-3)
        It should also have a 'bin' dimension representing the size bins.

    cdp_psd : xarray.Dataset
        A dataset containing the PSD data from a CDP instrument.
        It should have the same variables and dimensions as `pcasp_psd`.

    Returns
    -------
    dN : float
        The total number concentration (units: cm^-3).
    dS : float
        The total surface area concentration (units: um^2 cm^-3).
    dV : float
        The total volume concentration (units: um^3 cm^-3).
    dN_err : float
        The error in the total number concentration (units: cm^-3).
    dS_err : float
        The error in the total surface area concentration (units: cm^-3).
    dV_err : float
        The error in the total volume concentration (units: cm^-3).

    """
    # Sum the values along the "bin" dimension and convert to floats
    pcasp_dN, pcasp_dS, pcasp_dV = map(float, [pcasp_psd.dN.sum(dim='bin'),
                                               pcasp_psd.dS.sum(dim='bin'),
                                               pcasp_psd.dV.sum(dim='bin')])
    pcasp_dN_err, pcasp_dS_err, pcasp_dV_err = map(lambda x: float(np.sqrt((x**2).sum(dim='bin'))),
                                                   [pcasp_psd.dN_err, pcasp_psd.dS_err, pcasp_psd.dV_err])
    cdp_dN, cdp_dS, cdp_dV = map(float, [cdp_psd.dN.sum(dim='bin'),
                                         cdp_psd.dS.sum(dim='bin'),
                                         cdp_psd.dV.sum(dim='bin')])
    cdp_dN_err, cdp_dS_err, cdp_dV_err = map(lambda x: float(np.sqrt((x**2).sum(dim='bin'))),
                                             [cdp_psd.dN_err, cdp_psd.dS_err, cdp_psd.dV_err])
    # Calculate the gap values
    gap_dN, gap_dS, gap_dV = map(float, [area_under_gap(pcasp_psd, cdp_psd, 'dNdlogD'),
                                         area_under_gap(pcasp_psd, cdp_psd, 'dSdlogD'),
                                         area_under_gap(pcasp_psd, cdp_psd, 'dVdlogD')])
    gap_dN_err, gap_dS_err, gap_dV_err = map(lambda x, y: float(np.sqrt(x[-1]**2 + y[0]**2)),
                                             [pcasp_psd['dN_err'], pcasp_psd['dS_err'], pcasp_psd['dV_err']],
                                             [cdp_psd['dN_err'], cdp_psd['dS_err'], cdp_psd['dV_err']])
    # Calculate the final values
    dN = pcasp_dN + gap_dN + cdp_dN
    dS = pcasp_dS + gap_dS + cdp_dS
    dV = pcasp_dV + gap_dV + cdp_dV
    dN_err = np.sqrt(pcasp_dN_err**2 + gap_dN_err**2 + cdp_dN_err**2)
    dS_err = np.sqrt(pcasp_dS_err**2 + gap_dS_err**2 + cdp_dS_err**2)
    dV_err = np.sqrt(pcasp_dV_err**2 + gap_dV_err**2 + cdp_dV_err**2)
    return dN, dS, dV, dN_err, dS_err, dV_err

def make_array_str(arr):
    """Convert an array to a CSV string"""
    arr_str = ''
    for num in list(arr.values):
        arr_str = arr_str + str(num) +','
    arr_str = arr_str[:-1]
    return arr_str

def create_calibration_CSV(input_fn, instrument, output_fn, time_index=0, group=None):
    """Create CSVs containing calibration data for use with CStoDConverter.
    
    Users will need to investigate the correct time index themselves. For the PCASP, calibrations
    are performed at the start and end of each campaign. The start index = 0, the end index = 1.
    Check the NetCDF file to confirm this. The CDP is calibrated before most flights, and a master
    calibration is created for each campaign to iron out inconsistencies. Typically, you should use
    the master calibration - selection of 'CDP' automatically chooses this here. If you wish to use
    an individual flight calibration, you should choose the group flight_cal but this behaviour
    that I haven't prepared for and will probably break the function! 

    Parameters
    ----------
    input_fn: string
        filename of the input calibration NetCDF file 
    instrument: string
        name of the instrument being calibrated. Must be CDP or PCASP
    output_fn: string
        file for output. Not restricted to CSV but wise to choose so.
    time_index: int (default 0)
        choice of calibration times, see note above.
    group:
        group of NetCDF file to interrogate, see note above.
    """

    if instrument not in ['CDP','PCASP']:
        raise ValueError('Instrument string should be CDP or PCASP')
    if group is None:
        if instrument == 'CDP':
            group = 'master_cal'
        else:
            group = 'bin_cal'
    
    cal = xr.open_dataset(input_fn, group=group)
    min_voltage = cal.ADC_range[time_index, :, 0]
    max_voltage = cal.ADC_range[time_index, :, 1]
    gradient = cal.polynomial_fit_parameters[time_index, :, 1]
    intercept = cal.polynomial_fit_parameters[time_index, :, 0]
    var_grad = cal.polynomial_fit_variance[time_index, :, 1]
    var_int = cal.polynomial_fit_variance[time_index, :, 0]
    covar_int = cal.polynomial_fit_covariance[time_index, :]
    lower_thresholds = cal.ADC_threshold[time_index, :, 0]
    upper_thresholds = cal.ADC_threshold[time_index, :, 1]
    lower_boundaries = cal.scattering_cross_section[time_index, :, 0]
    upper_boundaries = cal.scattering_cross_section[time_index, :, 1]
    lower_boundary_errs = cal.scattering_cross_section_err[time_index, :, 0]
    upper_boundary_errs = cal.scattering_cross_section_err[time_index, :, 1]
    channel_widths = cal.scattering_cross_section_width[time_index]
    channel_widths_errs = cal.scattering_cross_section_width_err[time_index]
    boundary_type = cal.dependent_scattering_cross_section_err[time_index]
    csv_file = open(output_fn,'w')
    csv_file.write('Straight line fits,\n')
    csv_file.write('Min Voltage (A-D Counts),')
    csv_file.write(make_array_str(min_voltage))
    csv_file.write('\nMax Voltage (A-D Counts),')
    csv_file.write(make_array_str(max_voltage))
    csv_file.write('\nGradient (micron^2/A-D Counts),')
    csv_file.write(make_array_str(gradient))
    csv_file.write('\nIntercept (micron^2),')
    csv_file.write(make_array_str(intercept))
    csv_file.write('\nVar(Gradient)(micron^4/A-DCounts),')
    csv_file.write(make_array_str(var_grad))
    csv_file.write('\nVar(Intercept) (micron^4),')
    csv_file.write(make_array_str(var_int))
    csv_file.write('\nCovar(Gradient Intercept) (micron^4/A-DCounts),')
    csv_file.write(make_array_str(covar_int))
    csv_file.write('\n\nLower Thresholds,')
    csv_file.write(make_array_str(lower_thresholds))
    csv_file.write('\nUpper Thresholds,')
    csv_file.write(make_array_str(upper_thresholds))
    csv_file.write('\nLower Boundaries,')
    csv_file.write(make_array_str(lower_boundaries))
    csv_file.write('\nUpper Boundaries,')
    csv_file.write(make_array_str(upper_boundaries))
    csv_file.write('\nLower Boundary Errors,')
    csv_file.write(make_array_str(lower_boundary_errs))
    csv_file.write('\nUpper Boundary Errors,')
    csv_file.write(make_array_str(upper_boundary_errs))
    csv_file.write('\nChannel Widths,')
    csv_file.write(make_array_str(channel_widths))
    csv_file.write('\nChannel Widths Errors,')
    csv_file.write(make_array_str(channel_widths_errs))
    csv_file.write('\nBoundaries Independent/Dependent (0/1),')
    csv_file.write(make_array_str(boundary_type))
    csv_file.write('\n')
    csv_file.close()
    print(output_fn + ' created')
    return

    