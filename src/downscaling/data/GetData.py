import os
import pickle as pkl
import xarray as xr
from downscaling.data.EastData import processRA,processFC,processHC

def to_season(da):
    for k, v in da.__dict__.items(): 
        if isinstance(v,xr.core.dataarray.DataArray):
            try:
                da.__dict__.update({k:v.sel(N=v.N.dt.month.isin([12,1,2]))})
            except:
                print(v.N)
                raise ValueError()
    return da

def removeFeb29(da):
    for k, v in da.__dict__.items(): 
        if isinstance(v,xr.core.dataarray.DataArray):
            try:
                v_no_feb29 = v.where(~((v.N.dt.month == 2) & (v.N.dt.day == 29)), drop=True)
                da.__dict__.update({k:v_no_feb29})
            except:
                print(v.N)
                raise ValueError()
    return da

def get_full_RA(args,vars):
    """
    dynam: ERA5
    bench: Climatology
    refer: ERA5
    """

    args.base_ra_dir = f'{args.base_dir}/Train_Test_Data/ERA/'
    data_path_final = f'{args.base_ra_dir}/data_DJF_EAD_RA_{"_".join(vars)}'
    os.makedirs(os.path.dirname(data_path_final),exist_ok=1)

    if not os.path.exists(data_path_final):
        raY = processRA(vars,processing=True)

        pkl.dump(raY, open(data_path_final,'wb'))
        print(f'Saved Data in {data_path_final}')
    else:
        raY = pkl.load(open(data_path_final,'rb'))

    if 'ThenSeason' not in args.standardize:
        raY = to_season(raY)
    
    return removeFeb29(raY)

def get_full_FC(args,vars):
    """
    dynam: ECMWF
    bench: Climatology
    refer: ERA5
    """
    args.base_fc_dir = f'{args.base_dir}/Train_Test_Data/ENS/calib_{args.calib_method}'
    data_path_final = f'{args.base_fc_dir}/data_DJF_EAD_FC_{"_".join(vars)}'
    os.makedirs(os.path.dirname(data_path_final),exist_ok=1)
    if not os.path.exists(data_path_final):
        fcY = processFC(vars,processing=True)
        fcY = to_season(fcY)
        pkl.dump(fcY, open(data_path_final,'wb'))
        print(f'Saved Data in {data_path_final}')
    else:
        fcY = pkl.load(open(data_path_final,'rb'))
    return removeFeb29(fcY)

def get_full_HC(args,vars):
    """
    dynam: ECMWF
    bench: Climatology
    refer: ERA5
    """
    args.base_fc_dir = f'{args.base_dir}/Train_Test_Data/ENS/calib_{args.calib_method}'
    data_path_final = f'{args.base_fc_dir}/data_DJF_EAD_HC_{"_".join(vars)}'
    os.makedirs(os.path.dirname(data_path_final),exist_ok=1)
    if not os.path.exists(data_path_final):
        hcY = processHC(vars,processing=True)
        hcY = to_season(hcY)
        pkl.dump(hcY, open(data_path_final,'wb'))
        print(f'Saved Data in {data_path_final}')
    else:
        hcY = pkl.load(open(data_path_final,'rb'))
    return removeFeb29(hcY)


def convertToBCWH(args,da,data_type='X'):
    da = da.unstack()
    da = da.assign_coords(pdf=range(da.pdf.size))
    da = da.stack(sample=['N', 'pdf', 'time']).stack(channel=['varOI'])
    # da = da.assign_coords(number=range(da.number.size))
    # da = da.stack(sample=['initial_time', 'hdate', 'number', 'step']).stack(channel=['variable'])
    da = da.transpose('sample', ...,'channel', 'lat', 'lon')
    return da 
