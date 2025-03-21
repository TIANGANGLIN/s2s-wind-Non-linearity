import pickle as pkl
# from src.downscaling.data.processing import*
import xarray as xr
import numpy as np
from tqdm import tqdm 
import os
import pandas as pd 

def get_dates(start='1994-01-01', end='2022-01-13', months=['12','01','02'], all_dates=False, all_seasons=False):
    """
    Generate a list of dates within a specific time range and filtering conditions.
    
    Parameters:
    -----------
    start : str
        Start date in format 'YYYY-MM-DD'
    end : str
        End date in format 'YYYY-MM-DD'
    months : list
        List of months to include (as strings '01' to '12')
    all_dates : bool
        If True, returns all dates within range; if False, returns only Mondays and Thursdays
    all_seasons : bool
        If True, returns dates for all months; if False, filters by months parameter
        
    Returns:
    --------
    list
        List of dates meeting the specified criteria
    """
    if all_dates:
        if all_seasons:
            # Get all dates except February 29
            dates = [ele for ele in pd.date_range(start, end).strftime('%Y-%m-%d').tolist() if ele[5:]!='02-29']
        else:
            # Get dates for specified months only, excluding February 29
            dates = [ele for ele in pd.date_range(start, end).strftime('%Y-%m-%d').tolist() if ele[5:7] in months and ele[5:]!='02-29']

        dates.sort()
        return dates
    
    # Get all Mondays and Thursdays in specified months, excluding February 29
    mondays = [ele for ele in pd.date_range(start, end, freq='W-MON').strftime('%Y%m%d').tolist() if ele[4:6] in months and ele[4:]!='0229']
    thusdays = [ele for ele in pd.date_range(start, end, freq='W-THU').strftime('%Y%m%d').tolist() if ele[4:6] in months and ele[4:]!='0229']
    mondays.extend(thusdays)
    del thusdays
    mondays.sort()
    return mondays

def compute_wind_at_100m_v1(u10, v10):
    """
    Calculate wind speed at 100m height using power law profile equation.
    
    Parameters:
    -----------
    u10 : xarray.DataArray
        U component of wind at 10m height
    v10 : xarray.DataArray
        V component of wind at 10m height
        
    Returns:
    --------
    wind_speed_100m : xarray.Dataset
        Wind speed magnitude at 100m height
    """
    # Height parameters
    z1 = 10    # Reference height (m)
    z2 = 100   # Target height (hub height, m)
    
    # Power law exponent (value used in literature)
    alpha = 1/7    # Approximately 0.143
    
    # Calculate power law ratio factor
    power_ratio = (z2/z1)**alpha
    
    # Calculate wind components at 100m height
    u100 = u10 * power_ratio
    v100 = v10 * power_ratio
    
    # Calculate wind speed magnitude at 100m
    wind_speed_100m = np.sqrt(u100**2 + v100**2)
    wind_speed_100m = wind_speed_100m.to_dataset(name='ws100')
    wind_speed_100m = wind_speed_100m.assign_attrs({
        'units': 'm/s',
        'long_name': 'wind speed at 100m',
        'method': 'power law profile with alpha=1/7'
    })
    
    return wind_speed_100m

def compute_wind_at_100m_v2(u10, v10, z0):
    """
    Calculate wind speed at 100m height using logarithmic wind profile equation.
    
    Parameters:
    -----------
    u10 : xarray.DataArray
        U component of wind at 10m height
    v10 : xarray.DataArray
        V component of wind at 10m height
    z0 : xarray.DataArray
        Surface roughness length (m)
        
    Returns:
    --------
    wind_speed_100m : xarray.Dataset
        Wind speed magnitude at 100m height
    """
    # Height parameters
    z1 = 10    # Reference height (m)
    z2 = 100   # Target height (m)
    
    # Set minimum threshold for roughness length to avoid mathematical issues
    min_z0 = 1e-5
    z0 = xr.where(z0 < min_z0, min_z0, z0)
    
    # Calculate wind ratio factor using logarithmic wind profile equation
    # wind_ratio = ln(z2/z0) / ln(z1/z0)
    wind_ratio = np.log(z2 / z0) / np.log(z1 / z0)
    
    # Calculate wind components at 100m height
    u100 = u10 * wind_ratio
    v100 = v10 * wind_ratio
    
    # Calculate wind speed magnitude at 100m
    wind_speed_100m = np.sqrt(u100**2 + v100**2)
    wind_speed_100m = wind_speed_100m.to_dataset(name='ws100')
    wind_speed_100m = wind_speed_100m.assign_attrs({
                            'units': 'm/s',
                            'long_name': 'wind speed at 100m',
                        })
    
    return wind_speed_100m

def get_ws100_reanalysis():
    """
    Process and prepare reanalysis data to calculate 100m wind speed.
    Creates individual yearly files and a concatenated file covering 1979-2023.
    """
    os.makedirs('data/raw/reanalysis', exist_ok=True)
    
    # Process each year of data individually
    for year1 in tqdm(range(1979, 2023), desc=f'Getting reanalysis for ws100...'):
        year2 = year1
        year_range = f'jan{str(year1)[2:]}_jan{str(year2)[2:]}'
        
        # Skip if input file doesn't exist or output already exists
        if not os.path.exists(f'data/raw/reanalysis/fsr_glob_regrid_{year_range}.nc'):
            continue
        if os.path.exists(f'data/raw/reanalysis/ws100_glob_regrid_{year_range}.nc'): 
            continue
            
        ras = []
        # Load 10m wind components
        for var in ['10uv']:
            path = f'data/raw/reanalysis/{var}_glob_regrid_{year_range}.nc'
            ras_tmp = xr.open_dataset(path, engine='cfgrib')
            if 'heightAboveGround' in ras_tmp.dims: 
                ras_tmp = ras_tmp.drop('heightAboveGround')
            ras.append(ras_tmp)
        ras = xr.merge(ras)
        
        # Calculate 100m wind speed using power law profile
        ras = compute_wind_at_100m_v1(ras['u10'], ras['v10'])
        
        # Save yearly file
        ras.to_netcdf(f'data/raw/reanalysis/ws100_glob_regrid_{year_range}.nc')

    # Concatenate all yearly files into a single file covering 1979-2023
    for var in ['ws100']:
        ras = []
        if os.path.exists(f'data/raw/reanalysis/{var}_glob_regrid_jan79_jan23.nc'): 
            continue
            
        for year1 in tqdm(range(1979, 2023), desc=f'Concat reanalysis for {var}...'):
            year2 = year1
            year_range = f'jan{str(year1)[2:]}_jan{str(year2)[2:]}'
            path = f'data/raw/reanalysis/{var}_glob_regrid_{year_range}.nc'
            
            if var in ['ws100']:
                ras_tmp = xr.open_dataset(path,)
            else:
                ras_tmp = xr.open_dataset(path, engine='cfgrib')

            ras.append(ras_tmp)
            del ras_tmp
            
        ras = xr.concat(ras, dim='time')
        ras.to_netcdf(f'data/raw/reanalysis/{var}_glob_regrid_jan79_jan23.nc')

    return 

def get_paired_fsr(fsr, u10):
    """
    Extract and align surface roughness data with wind data timestamps.
    
    Parameters:
    -----------
    fsr : xarray.Dataset
        Surface roughness dataset
    u10 : xarray.Dataset
        10m wind dataset with specific time and step coordinates
        
    Returns:
    --------
    sub_fsr : xarray.Dataset
        Surface roughness data aligned with the wind data timestamps
    """
    # Clean up unnecessary dimensions
    fsr = fsr.drop('valid_time').drop('number').drop('step').drop('surface')
    
    # Extract valid times that exist in both datasets
    valid_times = u10.valid_time.values.ravel()
    valid_times = [ele for ele in valid_times if ele in fsr.time]
    sub_fsr = fsr.sel(time=valid_times).rename({'time': 'valid_time'})

    # Assign coordinates to match the u10 dataset structure
    sub_fsr = sub_fsr.assign_coords({
        'time': ('valid_time', np.repeat(u10.time.data, u10.step.size)),
        'step': ('valid_time', np.tile(u10.step.data, u10.time.size))
    })

    # Reshape to match the forecast data structure
    sub_fsr = sub_fsr.set_index(valid_time=('time', 'step')).unstack('valid_time').squeeze()
    return sub_fsr

def get_ws100_raw_ensemble(year_start, year_end):
    """
    Process forecast and hindcast ensemble data to calculate 100m wind speed.
    
    Parameters:
    -----------
    year_start : int
        Start year for processing
    year_end : int
        End year for processing (exclusive)
    """
    # Load surface roughness data (needed for logarithmic wind profile method)
    path = 'data/raw/reanalysis/fsr_glob_regrid_jan79_jan23.nc'
    ds_fsr = xr.open_dataset(path)
    
    # Process data year by year
    for year_s in tqdm(range(year_start, year_end), desc=f'Getting ensembles...', position=0):
        # Define time range (winter season: December to February)
        start = f'{year_s}-12-01'
        end = f'{year_s+1}-03-01'
        dates = get_dates(start=start, end=end, months=['01','02','12'])
        
        # Create directories for processed data
        os.makedirs('data/processed/fcasts/raw', exist_ok=1)
        os.makedirs('data/processed/hcasts/raw', exist_ok=1)

        # Process both forecast and hindcast data
        for ensemble_type in ['fcasts', 'hcasts']:
            for date in tqdm(dates, desc=f'Getting {ensemble_type} from {start} to {end}...', position=1):
                # Skip if output file already exists
                if os.path.exists(f'data/raw/{ensemble_type}/ws100/{date}_ws100.nc'): 
                    continue
                    
                # Load wind component data
                path_u = f'data/raw/{ensemble_type}/10u/{date}_10u.grib'
                path_v = f'data/raw/{ensemble_type}/10v/{date}_10v.grib'
                if not os.path.exists(path_u): 
                    continue
                    
                ds_u = xr.open_dataset(path_u, engine='cfgrib').astype('float32')
                ds_v = xr.open_dataset(path_v, engine='cfgrib').astype('float32')
                ds_u = ds_u.drop('heightAboveGround')
                ds_v = ds_v.drop('heightAboveGround')
                
                # Option to use surface roughness for logarithmic profile (commented out)
                # ds_sub_fsr = get_paired_fsr(ds_fsr, ds_u)
                # ds = compute_wind_at_100m(ds_u.u10, ds_v.v10, ds_sub_fsr.fsr)
                
                # Calculate 100m wind speed using power law profile
                ds = compute_wind_at_100m_v1(ds_u.u10, ds_v.v10)
                
                # Save output file
                os.makedirs(f'data/raw/{ensemble_type}/ws100/', exist_ok=1)
                ds.to_netcdf(f'data/raw/{ensemble_type}/ws100/{date}_ws100.nc')
    return

if __name__ == '__main__':
    # Process reanalysis data to calculate 100m wind speed
    get_ws100_reanalysis()
    
    # Process ensemble forecast/hindcast data
    get_ws100_raw_ensemble(2015, 2024)    # Process data from 2015 to 2023
