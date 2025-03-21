import numpy as np
from collections import OrderedDict
import cdsapi
from tqdm import tqdm

def ra_downloader(vars,res=2.7):
    if resolution_tim=='1H':
        time_range = [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                    ]
    elif resolution_tim=='6H':
        time_range = ['00:00', '06:00', '12:00','18:00',]
    else:
        raise ValueError('')
    
    if seasons=='all':
        month_range = [str(ele).zfill(2) for ele in np.arange(1,13)]
    elif seasons=='DJF':
        month_range = [str(ele).zfill(2) for ele in [1,2,12]]
    else:
        raise ValueError()

    dict_general = {
        'product_type': 'reanalysis',
            'format': 'netcdf',
            'month': month_range,
            'year': [str(ele) for ele in np.arange(year1,year2+1)],
            'day': [str(ele).zfill(2) for ele in np.arange(1,32)],
            'time': time_range, 
            'format': 'grib',
            'grid': [res, res],
            "area": [80, -120, 20, 40],
            }
    
    c = cdsapi.Client()

    for var in vars:
        save_path = f'./data/raw/reanalysis/{var}_glob_regrid_{year_range}.nc'
        if os.path.exists(save_path): continue
        if var in ['z500','z50','z10','z250','z850']:
            dict_var = {'variable': 'geopotential'}
            dict_var.update({'pressure_level': var.split('z')[-1]})
            dataset_type = 'reanalysis-era5-pressure-levels'
        elif var in ['100uv']:
            dict_var = {'variable': ['100m_u_component_of_wind', '100m_v_component_of_wind',]}
            dataset_type = 'reanalysis-era5-single-levels'
        elif var in ['10uv']:
            dict_var = {'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind',]}
            dataset_type = 'reanalysis-era5-single-levels'
        elif var in ['t2m']:
            dict_var = {'variable': '2m_temperature'}
            dataset_type = 'reanalysis-era5-single-levels'
        elif var in ['sp']:
            dict_var = {'variable': ['surface_pressure',]}
            dataset_type = 'reanalysis-era5-single-levels'
        else:
            dict_var = {'variable': var,}
            dataset_type = 'reanalysis-era5-single-levels'

        dict_var.update(dict_general) # dict_var
        c.retrieve(
            dataset_type,
            dict_var,
            save_path,
            )
    
    return


if __name__ == '__main__':
    import os
    res = 2.7 # 0.9, 2.7
    resolution_tim = '6H' # '1H'
    seasons = 'all' # all, DJF

    for year1 in tqdm(range(1979,2023)):
        year1,year2 = year1,year1
        year_range = f'jan{str(year1)[2:]}_jan{str(year2)[2:]}'
        if seasons!='all':
            year_range = f'{year_range}_{seasons}'

        os.makedirs(f'./data/raw/reanalysis/',exist_ok=1)

        # Reananlysis processor 
        var_filenames = OrderedDict({
                    '10uv':  [f'10u_glob_regrid_{year_range}.nc',f'10v_glob_regrid_{year_range}.nc',],
                    'z500': [f'z500_glob_regrid_{year_range}.nc'],
                    # 't2m': [f't2m_glob_regrid_{year_range}.nc'],
                    # 'sp': [f'sp_glob_regrid_{year_range}.nc'],
                    # '100uv':  [f'100m_u_component_of_wind_glob_regrid_{year_range}.nc',
                    #         f'100m_v_component_of_wind_glob_regrid_{year_range}.nc',],
                    })

        # Download
        ra_downloader(vars=list(var_filenames.keys()),res=res)
