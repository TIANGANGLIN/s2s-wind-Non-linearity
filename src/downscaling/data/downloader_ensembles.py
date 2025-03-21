#!/usr/bin/env python
# Doc: 
# Check param: https://codes.ecmwf.int/grib/param-db/
# Example
# https://apps.ecmwf.int/datasets/data/s2s-realtime-daily-averaged-ecmf/levtype=sfc/type=pf/
import os
from ecmwfapi import ECMWFDataServer
import random
import sys
import copy
import pandas as pd 

def get_dates(start='1994-01-01', end='2022-01-13',months=['12','01','02'],all_dates=False,all_seasons=False):
    if all_dates:
        # dates = [ele for ele in pd.date_range(start, end).strftime('%Y%m%d').tolist()]
        # dates = [ele for ele in pd.date_range(start, end).strftime('%Y%m%d').tolist() if ele[4:6] in months and ele[4:]!='0229']
        if all_seasons:
            # dates = [ele for ele in pd.date_range(start, end).strftime('%Y%m%d').tolist() if ele[4:]!='0229']
            dates = [ele for ele in pd.date_range(start, end).strftime('%Y-%m-%d').tolist() if ele[5:]!='02-29']
        else:
            dates = [ele for ele in pd.date_range(start, end).strftime('%Y-%m-%d').tolist() if ele[5:7] in months and ele[5:]!='02-29']

        dates.sort()
        return dates
    
    mondays = [ele for ele in pd.date_range(start, end, freq='W-MON').strftime('%Y%m%d').tolist() if ele[4:6] in months and ele[4:]!='0229']
    thusdays = [ele for ele in pd.date_range(start, end, freq='W-THU').strftime('%Y%m%d').tolist() if ele[4:6] in months and ele[4:]!='0229']
    mondays.extend(thusdays)
    del thusdays
    mondays.sort()
    return mondays

def get_param(var):
    param_dict = {
                  'gh': '156',
                  '10u': '165',
                  '10v': '166',
                  '100u': '228246',
                  '100v': '228247',
                  '2t': '167',
                  'sp': '134',
                  }
    if var in param_dict.keys():
        return param_dict[var]
    elif var in ['z500']:
        return param_dict['gh']
    elif var=='10uv':
        return param_dict['10si']
    elif var=='100uv':
        return param_dict['100si']
    elif var=='t2m':
        return param_dict['2t']
    else:
        raise ValueError(f'Do not have this {var}')

def get_levtype(var):
    if var in ['z500']:
        levtype = 'pl'
    elif var in ['100uv','100u','100v','10u','10v','10uv','10si','t2m','sp']:
        levtype = 'sfc'
    else:
        raise ValueError(f'Do not have this {var}')
    return levtype

def get_step(var):
    if var in ['t2m','sst']:
        step = "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/864-888/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056/1056-1080/1080-1104"
    else:
        step = "0/24/48/72/96/120/144/168/192/216/240/264/288/312/336/360/384/408/432/456/480/504/528/552/576/600/624/648/672/696/720/744/768/792/816/840/864/888/912/936/960/984/1008/1032/1056/1080/1104"
    return step


if __name__ == '__main__':

    server = ECMWFDataServer()
    fdates = get_dates(start='2015-12-01', end='2022-03-01',months=['01','02','12'])
    random.shuffle(fdates)
    vars = ['z500','10u','10v',]

    random.shuffle(vars)
    for var in vars:
        param = get_param(var)
        dataclass = "s2"
        dataset = "s2s"
        expver = "prod"
        levtype = get_levtype(var)
        model = "glob"
        fc_number = "1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50"
        hc_number = "1/2/3/4/5/6/7/8/9/10"
        origin = "ecmf"
        param = get_param(var)
        step = get_step(var)
        fc_stream = "enfo"
        hc_stream = "enfh"
        time = "00:00:00"
        type = "pf" # https://codes.ecmwf.int/grib/format/mars/type/
        res = "2.7"
        grid = f"{res}/{res}" # "0.9/0.9", "1.5/1.5"

        basic_conf_dict = {
                    "class": dataclass,
                    "dataset": dataset,
                    "expver": expver,
                    "levtype": levtype,
                    "model": model,
                    "origin": origin,
                    "param": param,
                    "step": step,
                    "time": time,
                    "type": type,
                    "grid": grid,
                    'area': [80, -120, 20, 40,], # large_scale = 20,80,-120,40
                    # 'area': [74, -13, 34, 40,], # Europe = 34,73,-13,40
                    }

        cpu_num = 4
        
        os.makedirs(f'./data/raw/fcasts/{var}',exist_ok=1)
        os.makedirs(f'./data/raw/hcasts/{var}',exist_ok=1)
        
        fdates_rest = [fdate for fdate in fdates if not os.path.exists(f'./data/raw/hcasts/{var}/{fdate}_{var}.grib')]
        random.shuffle(fdates_rest)

        print('fdates_rest',fdates_rest)
        # with pymp.Parallel(cpu_num) as p:
        #     for idx in p.range(len(fdates_rest)):
        if 1:
            for idx in range(len(fdates_rest)):
                # try:
                if 1:
                    fdate = fdates_rest[idx]
                    if os.path.exists(f'./data/raw/hcasts/{var}/{fdate}_{var}.grib'):continue

                    # Forecast
                    retrieve_conf_dict = copy.copy(basic_conf_dict)
                    retrieve_conf_dict["date"] = f"{fdate[:4]}-{fdate[4:6]}-{fdate[6:]}"
                    retrieve_conf_dict["target"] = f"./data/raw/fcasts/{var}/{fdate}_{var}.grib"
                    retrieve_conf_dict["number"] = fc_number
                    retrieve_conf_dict["stream"] = fc_stream
                    if var=='z500': retrieve_conf_dict['levelist'] = '500'

                    server.retrieve(retrieve_conf_dict)
                    # os.system(f"grib_to_netcdf ./data/raw/fcasts/{var}/{fdate}_{var}.grib -o ./data/raw/fcasts/{var}/{fdate}_{var}.nc -D NC_SHORT")
                    # os.system(f"rm ./data/raw/fcasts/{var}/{fdate}_{var}.grib")

                    # Hindcast
                    year = int(fdate[:4])
                    hdates = [f'{year+ele}-{fdate[4:6]}-{fdate[6:]}' for ele in range(-20,0)]
                    hdates = '/'.join(hdates)
                    retrieve_conf_dict = copy.copy(basic_conf_dict)
                    retrieve_conf_dict["date"] = f"{fdate[:4]}-{fdate[4:6]}-{fdate[6:]}"
                    retrieve_conf_dict["hdate"] = hdates
                    retrieve_conf_dict["target"] = f"./data/raw/hcasts/{var}/{fdate}_{var}.grib"
                    retrieve_conf_dict["number"] = hc_number
                    retrieve_conf_dict["stream"] = hc_stream

                    if var=='z500': retrieve_conf_dict['levelist'] = '500'
                    server.retrieve(retrieve_conf_dict)
                    # os.system(f"grib_to_netcdf ./data/raw/hcasts/{var}/{fdate}_{var}.grib -o ./data/raw/hcasts/{var}/{fdate}_{var}.nc -D NC_SHORT")
                    # os.system(f"rm ./data/raw/hcasts/{var}/{fdate}_{var}.grib")


