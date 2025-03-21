# %%
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm 
import os
import xarray as xr
import dask
from dask.distributed import Client
from tqdm import tqdm
import os
import logging
from downscaling.data.downloader_ensembles import get_dates
from downscaling.data.compute_U100 import get_ws100_reanalysis, get_ws100_raw_ensemble

# %%
def get_reanalysis(variables):
    """
    Process raw reanalysis data for specified variables into daily and weekly means.
    Creates both daily and rolling weekly mean files for the period 1979-2023 (DJF).
    
    Args:
        variables: List of variable names to process
    """
    os.makedirs('data/processed/reanalysis', exist_ok=True)
    for var in variables:
        if os.path.exists(f'data/processed/reanalysis/{var}_DJF1979_2023_rolling_weekly_mean.nc'): continue
        ras = []
        for year1 in tqdm(range(1979,2023),desc=f'Getting reanalysis for {var}...'):
            year1,year2 = year1,year1
            year_range = f'jan{str(year1)[2:]}_jan{str(year2)[2:]}'
            # Basic reading method
            path = f'data/raw/reanalysis/{var}_glob_regrid_{year_range}.nc'
            if var in ['ws100']:
                ras_tmp = xr.open_dataset(path)
            else:
                ras_tmp = xr.open_dataset(path, engine='cfgrib')
            if var in ['10uv']:
                # Calculate wind speed from u and v components
                ras_tmp['ws10'] = np.sqrt(ras_tmp.u10**2 + ras_tmp.v10**2).assign_attrs({
                    'units': 'm/s',
                    'long_name': '10-meter wind speed',
                })
                ras_tmp = ras_tmp.drop_vars(['u10', 'v10'])
            ras.append(ras_tmp)
        ras = xr.concat(ras,dim='time')
        daily_mean = ras.resample(time='1D').mean().astype('float32')
        daily_mean.to_netcdf(f'data/processed/reanalysis/{var}_DJF1979_2023_daily_mean.nc')
        weekly_mean = ras.rolling(time=28, center=True, min_periods=1).mean().astype('float32')
        weekly_mean.to_netcdf(f'data/processed/reanalysis/{var}_DJF1979_2023_rolling_weekly_mean.nc')
    return 

# %%
def get_past_year(given_time, years):
    """
    Calculate times for historical years before a given time
    
    Args:
        given_time: Reference timestamp
        years: Number of years to look back
        
    Returns:
        List of timestamps for previous years
    """
    if years==0:
        return [pd.to_datetime(given_time.values)]

    years_priors = []

    for year in range(years, 0,-1):
        years_prior = pd.to_datetime(given_time.values) - pd.DateOffset(years=year)
        years_priors.append(years_prior)

    return years_priors

def get_past_ds(ds_all, ds_sub, sub_type, pre_years=15):
    """
    Extract climatology data from historical years
    
    Args:
        ds_all: Complete dataset covering all years
        ds_sub: Subset dataset for reference times
        sub_type: Type of subset ('RA' for reanalysis)
        pre_years: Number of years to include in climatology
        
    Returns:
        Dataset with historical data arranged by reference times
    """
    tmps = []
    # All dates
    set2 = set(pd.to_datetime(ds_all.time.data))

    for i_time in tqdm(range(ds_sub.time.size),desc=f'Getting {pre_years} years climatology...'):
        given_time = ds_sub.isel(time=i_time).time
        # given_time = ds_sub.isel(time=-1).time
        given_time.values, 
        past_years = get_past_year(given_time, pre_years)
        
        # Convert list to set for faster comparison
        set1 = set(past_years)

        if pre_years==0:
            hdate = [0]
        else:
            hdate = range(-pre_years,0)
        # Check if all elements in list1 are in list2
        if set1.issubset(set2):
            tmp = ds_all.sel(time=past_years).rename({'time':'hdate'}).assign_coords(hdate=hdate)
            if sub_type=='RA':
                tmp = tmp.expand_dims(initial_time=[given_time.values],step=[0])
            else:
                tmp = tmp.expand_dims(time=[given_time.values])

            tmps.append(tmp)
        else:
            continue

    if sub_type=='RA':
        tmps = xr.combine_nested(tmps,concat_dim='initial_time').expand_dims(number=[0])
    else:
        tmps = xr.combine_nested(tmps,concat_dim='time')
    return tmps


def get_reanalysis_for_training(variables):
    """
    Prepare reanalysis data for training by extracting current values and climatology
    
    Args:
        variables: List of variable names to process
    """
    for var in variables:
        weekly_mean = xr.open_dataset(f'data/processed/reanalysis/{var}_DJF1979_2023_rolling_weekly_mean.nc')
        weekly_mean = weekly_mean.compute()
        if os.path.exists(f'data/processed/reanalysis/{var}_DJF1979_2023_rolling_weekly_mean_training_ra.nc'): continue
        ds_ra = get_past_ds(weekly_mean,weekly_mean,'RA',pre_years=0)
        ds_ra.to_netcdf(f'data/processed/reanalysis/{var}_DJF1979_2023_rolling_weekly_mean_training_ra.nc')
        del ds_ra
        if os.path.exists(f'data/processed/reanalysis/{var}_DJF1979_2023_rolling_weekly_mean_training_cl.nc'): continue
        ds_cl = get_past_ds(weekly_mean,weekly_mean,'RA',pre_years=15)
        ds_cl.to_netcdf(f'data/processed/reanalysis/{var}_DJF1979_2023_rolling_weekly_mean_training_cl.nc')
        del ds_cl

    return 

# %%
def get_refer_bench(daily_mean, ds):
    """
    Create reference and benchmark datasets from reanalysis data
    aligned with forecast valid times
    
    Args:
        daily_mean: Daily mean reanalysis data
        ds: Forecast dataset with time and step dimensions
        
    Returns:
        ra: Reference dataset matched to forecast valid times
        cl: Climatology dataset matched to forecast valid times
    """
    valid_times = ds.valid_time.values.ravel()
    ra = daily_mean.sel(time=valid_times).rename({'time': 'valid_time'})
    ra = ra.assign_coords({
        'time': ('valid_time', np.repeat(ds.time.data, ds.step.size)),
        'step': ('valid_time', np.tile(ds.step.data, ds.time.size))
    })

    # Reorganize data using these indices
    ra = ra.set_index(valid_time=('time', 'step')).unstack('valid_time').squeeze().expand_dims(number=[1])

    # Get climatology
    if ds.time.size > 1:
        # climatology of hindcasts 
        sub_ds = ds.isel(time=range(5,ds.time.size))
        valid_times = sub_ds.valid_time.values.ravel()
        cl = daily_mean.sel(time=valid_times).rename({'time': 'valid_time'})
        cl = cl.assign_coords({
            'time': ('valid_time', np.repeat(sub_ds.time.data, sub_ds.step.size)),
            'step': ('valid_time', np.tile(sub_ds.step.data, sub_ds.time.size))
        })

        # Reorganize data using these indices
        cl = cl.set_index(valid_time=('time', 'step')).unstack('valid_time').squeeze().drop('number').rename({'time':'number'}).expand_dims(time=ds.time)
        cl = cl.assign_coords(number=range(cl.number.size))
    else:
        cl = ra.copy(deep=True)
    return ra, cl


# %%


# %%
def processing(ds):
    """
    Process dataset by:
    1. Computing weekly means
    2. Adjusting coordinate names based on data type
    
    Args:
        ds: Input dataset
        
    Returns:
        Processed dataset with standardized coordinates
    """
    ds = ds.rolling(step=7, center=True).mean().dropna('step').isel(step=np.arange(6)*7)
    if ds.time.size>1:
        initial_time = pd.to_datetime(ds.time.isel(time=[-1])) + pd.DateOffset(years=1)
        ds = ds.rename({'time':'hdate'})
        ds = ds.assign_coords(hdate=range(-20,0)).expand_dims(initial_time=initial_time)
    else:
        ds = ds.rename({'time':'initial_time'})
        ds = ds.expand_dims(hdate=[0])
    return ds


def get_ensembles(variables, year_start, year_end):
    """
    Prepare ensemble forecast and hindcast data for each time period
    
    Args:
        variables: List of variable names to process
        year_start: First year to process
        year_end: Last year to process
    """
    for year_s in tqdm(range(year_start,year_end),desc=f'Getting ensembles...',position=0):
        start=f'{year_s}-12-01'
        end=f'{year_s+1}-03-01'
        dates = get_dates(start=start, end=end,months=['01','02','12'])
        os.makedirs('data/processed/fcasts/raw',exist_ok=1)
        os.makedirs('data/processed/hcasts/raw',exist_ok=1)
        for var in variables:
            daily_mean = xr.open_dataset(f'data/processed/reanalysis/{var}_DJF1979_2023_daily_mean.nc')
            for ensemble_type in ['fcasts','hcasts']:
                if os.path.exists(f'data/processed/{ensemble_type}/raw/dynam_{var}_dates{start}to{end}.nc') and os.path.exists(f'data/processed/{ensemble_type}/raw/refer_{var}_dates{start}to{end}.nc') and os.path.exists(f'data/processed/{ensemble_type}/raw/bench_{var}_dates{start}to{end}.nc') : continue
                ds_all = []
                ra_all = []
                cl_all = []
                for date in tqdm(dates,desc=f'Getting {ensemble_type} for {var} from {start} to {end}...',position=1):
                    if var in ['10uv']:
                        path_u = f'data/raw/{ensemble_type}/10u/{date}_10u.grib'
                        path_v = f'data/raw/{ensemble_type}/10v/{date}_10v.grib'
                        ds_u = xr.open_dataset(path_u,engine='cfgrib').astype('float32')
                        ds_v = xr.open_dataset(path_v,engine='cfgrib').astype('float32')
                        ds = ds_u.copy(deep=True).drop_vars(['u10'])
                        ds['ws10'] = np.sqrt(ds_u.u10**2 + ds_v.v10**2).assign_attrs({
                                    'units': 'm/s',
                                    'long_name': '10-meter wind speed',
                                })
                        del ds_u, ds_v
                    elif var in ['ws100']:
                        path = f'data/raw/{ensemble_type}/{var}/{date}_{var}.nc'
                        ds = xr.open_dataset(path).astype('float32')
                    else:
                        path = f'data/raw/{ensemble_type}/{var}/{date}_{var}.grib'
                        ds = xr.open_dataset(path,engine='cfgrib').astype('float32')
                    try:
                        ra, cl = get_refer_bench(daily_mean, ds)
                    except:
                        raise ValueError(f'Error in {var} {ensemble_type} {date}')
                        
                    ds = processing(ds).astype('float32')
                    ra = processing(ra).astype('float32')
                    cl = processing(cl).astype('float32')
                    ds_all.append(ds)
                    ra_all.append(ra)
                    cl_all.append(cl)
                ds_all = xr.concat(ds_all,dim='initial_time').astype('float32')
                ra_all = xr.concat(ra_all,dim='initial_time').astype('float32')
                cl_all = xr.concat(cl_all,dim='initial_time').astype('float32')
                ds_all.to_netcdf(f'data/processed/{ensemble_type}/raw/dynam_{var}_dates{start}to{end}.nc')
                ra_all.to_netcdf(f'data/processed/{ensemble_type}/raw/refer_{var}_dates{start}to{end}.nc')
                cl_all.to_netcdf(f'data/processed/{ensemble_type}/raw/bench_{var}_dates{start}to{end}.nc')
                del ds_all
                del ra_all
                del cl_all

    data_types = ['dynam','refer','bench']
    for var in variables:
        for ensemble_type in ['fcasts','hcasts']:
            for data_type in data_types:
                ds_all = []
                if os.path.exists(f'data/processed/{ensemble_type}/raw/{data_type}_{var}_dates{year_start}-12-01to{year_end}-03-01.nc'): continue
                for year_s in tqdm(range(year_start,year_end)):
                    start=f'{year_s}-12-01'
                    end=f'{year_s+1}-03-01'
                    ds_all.append(xr.open_dataset(f'data/processed/{ensemble_type}/raw/{data_type}_{var}_dates{start}to{end}.nc'))
                ds_all = xr.concat(ds_all,dim='initial_time').astype('float32')
                ds_all.to_netcdf(f'data/processed/{ensemble_type}/raw/{data_type}_{var}_dates{year_start}-12-01to{year_end}-03-01.nc')
    return

def process_single_period(year_s, ensemble_type, data_type, var):
    """
    Process data for a single time period
    
    Args:
        year_s: Year to process
        ensemble_type: Type of ensemble (fcasts or hcasts)
        data_type: Type of data (dynam, refer, bench)
        var: Variable name
        
    Returns:
        Loaded dataset or None if error occurs
    """
    try:
        start = f'{year_s}-12-01'
        end = f'{year_s+1}-03-01'
        file_path = f'data/processed/{ensemble_type}/raw/{data_type}_{var}_dates{start}to{end}.nc'
        
        # Configure dask to handle large data chunks
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            ds = xr.open_dataset(file_path, chunks={
                'time': 10,
                'latitude': 23,
                'longitude': 60,
            })
        return ds
    except Exception as e:
        logging.error(f"Error processing file for {year_s}: {str(e)}")
        return None

def process_data_batch(year_start, year_end, ensemble_type, data_type, var, chunk_size=3):
    """
    Process data in batches to manage memory usage
    
    Args:
        year_start: First year to process
        year_end: Last year to process
        ensemble_type: Type of ensemble (fcasts or hcasts)
        data_type: Type of data (dynam, refer, bench)
        var: Variable name
        chunk_size: Number of years to process in each batch
        
    Returns:
        Boolean indicating success or failure
    """
    output_file = f'data/processed/{ensemble_type}/raw/{data_type}_{var}_dates{year_start}-12-01to{year_end}-03-01.nc'
    
    if os.path.exists(output_file):
        logging.info(f"Output file {output_file} already exists. Skipping.")
        return True

    try:
        all_datasets = []
        
        for i in range(year_start, year_end, chunk_size):
            chunk_end = min(i + chunk_size, year_end)
            chunk_ds = []
            
            # Process each small batch
            for year_s in tqdm(range(i, chunk_end), desc=f"Processing years {i}-{chunk_end}"):
                ds = process_single_period(year_s, ensemble_type, data_type, var)
                if ds is not None:
                    chunk_ds.append(ds)
            
            if chunk_ds:
                # Merge small batch data
                chunk_merged = xr.concat(chunk_ds, dim='initial_time').astype('float32')
                all_datasets.append(chunk_merged)
                
                # Clean up memory
                del chunk_ds
                dask.distributed.get_client().cancel(chunk_merged)
            
            # Force garbage collection
            import gc
            gc.collect()
        
        # Finally merge all data and write to file
        if all_datasets:
            final_ds = xr.concat(all_datasets, dim='initial_time')
            final_ds.to_netcdf(output_file)
            del final_ds
            gc.collect()
            
        return True
    except Exception as e:
        logging.error(f"Error processing batch: {str(e)}")
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
                logging.info(f"Removed incomplete output file: {output_file}")
            except:
                pass
        return False

def main(variables, year_start, year_end):
    """
    Main function to process all data combinations
    
    Args:
        variables: List of variable names to process
        year_start: First year to process
        year_end: Last year to process
    """
    # Set up dask client
    client = Client(n_workers=4, threads_per_worker=2, memory_limit='16GB')
    logging.info("Dask client initialized")
    
    data_types = ['dynam', 'refer', 'bench']
    successful_processes = 0
    total_processes = len(variables) * 2 * len(data_types)
    
    try:
        for var in variables:
            for ensemble_type in ['fcasts', 'hcasts']:
                for data_type in data_types:
                    logging.info(f"Processing: {var} - {ensemble_type} - {data_type}")
                    
                    success = process_data_batch(
                        year_start=year_start,
                        year_end=year_end,
                        ensemble_type=ensemble_type,
                        data_type=data_type,
                        var=var,
                        chunk_size=4
                    )
                    
                    if success:
                        successful_processes += 1
                    
                    # Clean memory after each processing group
                    import gc
                    gc.collect()
                    
        logging.info(f"Processing completed. Success rate: {successful_processes}/{total_processes}")
    
    finally:
        client.close()

# %%
def get_calibrated_ens(var):
    """
    Create calibrated ensemble forecasts using bias correction
    
    Args:
        var: Variable name to calibrate
        
    Returns:
        Tuple of calibrated forecast and hindcast datasets
    """
    if os.path.exists(f'data/processed/hcasts/calib/dynam_{var}_dates{year_start}-12-01to{year_end}-03-01.nc'): 
        return None, None
    os.makedirs('data/processed/fcasts/calib',exist_ok=True)
    os.makedirs('data/processed/hcasts/calib',exist_ok=True)
    def calibration_fc(x,x_dynam,x_refer):
        """
        Calibrate forecast using mean/standard deviation correction
        
        Args:
            x: Original forecast
            x_dynam: Reference model output (hindcast)
            x_refer: Reference observation data
            
        Returns:
            Calibrated forecast
        """
        xmean_dynam = x_dynam.mean(['number','hdate'])
        xmean_refer = x_refer.mean(['number','hdate'])
        xstdv_dynam = np.sqrt(x_dynam.var(['number','hdate']))
        xstdv_refer = np.sqrt(x_refer.var(['number','hdate']))
        return (x - xmean_dynam)*(xstdv_refer/xstdv_dynam) + xmean_refer

    def calibration_hc(x,x_dynam,x_refer):
        """
        Calibrate hindcasts using leave-one-out approach
        
        Args:
            x: Original hindcast
            x_dynam: Reference model output
            x_refer: Reference observation data
            
        Returns:
            Calibrated hindcast
        """
        cal = []
        for hdate in x.hdate.data:
            xmean_dynam = x_dynam.drop_sel(hdate=hdate).mean(['number','hdate'])
            xmean_refer = x_refer.drop_sel(hdate=hdate).mean(['number','hdate'])
            xstdv_dynam = np.sqrt(x_dynam.drop_sel(hdate=hdate).var(['number','hdate']))
            xstdv_refer = np.sqrt(x_refer.drop_sel(hdate=hdate).var(['number','hdate']))
            cal_tmp = (x.sel(hdate=hdate) - xmean_dynam)*(xstdv_refer/xstdv_dynam) + xmean_refer
            cal.append(cal_tmp)
            del xmean_dynam,xmean_refer,xstdv_dynam,xstdv_refer,cal_tmp
        cal = xr.concat(cal,dim='hdate')
        return cal

    fc_dynam = xr.open_dataset(f'data/processed/fcasts/raw/dynam_{var}_dates{year_start}-12-01to{year_end}-03-01.nc')
    fc_refer = xr.open_dataset(f'data/processed/fcasts/raw/refer_{var}_dates{year_start}-12-01to{year_end}-03-01.nc')
    fc_bench = xr.open_dataset(f'data/processed/fcasts/raw/bench_{var}_dates{year_start}-12-01to{year_end}-03-01.nc')
    hc_dynam = xr.open_dataset(f'data/processed/hcasts/raw/dynam_{var}_dates{year_start}-12-01to{year_end}-03-01.nc')
    hc_refer = xr.open_dataset(f'data/processed/hcasts/raw/refer_{var}_dates{year_start}-12-01to{year_end}-03-01.nc')
    hc_bench = xr.open_dataset(f'data/processed/hcasts/raw/bench_{var}_dates{year_start}-12-01to{year_end}-03-01.nc')
    print('fc_dynam',fc_dynam.dims)
    print('fc_refer',fc_refer.dims)
    print('fc_bench',fc_bench.dims)
    print('hc_dynam',hc_dynam.dims)
    print('hc_refer',hc_refer.dims)
    print('hc_bench',hc_bench.dims)
    if var=='z500':
        fc_dynam = fc_dynam.rename({'gh':'z'})
        hc_dynam = hc_dynam.rename({'gh':'z'})

    # fcasts calibration 
    fc_calib = calibration_fc(fc_dynam,x_dynam=hc_dynam,x_refer=fc_refer).astype('float32')
    # hcasts calibration 
    hc_calib = calibration_hc(hc_dynam,x_dynam=hc_dynam,x_refer=hc_refer).astype('float32')
    fc_calib.to_netcdf(f'data/processed/fcasts/calib/dynam_{var}_dates{year_start}-12-01to{year_end}-03-01.nc')
    hc_calib.to_netcdf(f'data/processed/hcasts/calib/dynam_{var}_dates{year_start}-12-01to{year_end}-03-01.nc')
    print('fc_calib',fc_calib.dims)
    print('hc_calib',hc_calib.dims)

    return fc_calib, hc_calib


if __name__ == '__main__':
    # Define parameters for processing
    year_start = 2015
    year_end = 2022
    variables = ['10uv','z500']
    print(variables)

    # Execute processing functions
    get_reanalysis(variables)
    get_reanalysis_for_training(variables)
    get_ensembles(variables,year_start,year_end)
    main(variables, year_start, year_end)

    for var in tqdm(variables,desc='Getting calibration...'):
        fc_calib, hc_calib = get_calibrated_ens(var=var)

    # Process reanalysis data to calculate 100m wind speed
    get_ws100_reanalysis()
    
    # Process ensemble forecast/hindcast data
    get_ws100_raw_ensemble(2015, 2024)    # Process data from 2015 to 2023

    variables = ['ws100']
    # Execute processing functions
    get_reanalysis(variables)
    get_reanalysis_for_training(variables)
    get_ensembles(variables,year_start,year_end)
    main(variables, year_start, year_end)

    for var in tqdm(variables,desc='Getting calibration...'):
        fc_calib, hc_calib = get_calibrated_ens(var=var)

