import sys
sys.path.append('.')
import warnings
if not sys.warnoptions: warnings.simplefilter("ignore")
import pandas as pd
import xarray as xr

class processRA():
    """
    dynam: ERA5
    bench: Climatology
    refer: ERA5
    """
    def __init__(self,vars_list,processing):
        self.vars = vars_list
        self.processing = processing
        self.start='2015-12-01'
        self.end='2022-03-01'
        self.get_raw_data()
        # print(f'    Get raw data used: {time.time()-st:.2f}s')

    def get_raw_data(self):
        self.dynam = []
        self.bench = []
         
        for var in self.vars:
            refer = xr.open_dataset(f'data/processed/reanalysis/{var}_DJF1979_2023_rolling_weekly_mean_training_ra.nc')
            bench = xr.open_dataset(f'data/processed/reanalysis/{var}_DJF1979_2023_rolling_weekly_mean_training_cl.nc')
            
            self.dynam.append(refer)
            self.bench.append(bench)
            del refer, bench

        self.dynam = xr.merge(self.dynam).rename({'initial_time':'N','number':'pdf','step':'time','latitude':'lat','longitude':'lon'})
        self.bench = xr.merge(self.bench).rename({'initial_time':'N','number':'pdf','step':'time','latitude':'lat','longitude':'lon'})
        self.dynam = reassign_hdate_to_N(self.dynam).to_array().rename({'variable':'varOI'}).compute()
        self.bench = reassign_hdate_to_N(self.bench).to_array().rename({'variable':'varOI'}).compute()
        self.refer = self.dynam.copy(deep=True)

class processFC():
    """
    dynam: ECMWF
    bench: Climatology
    refer: ERA5
    """
    def __init__(self,vars_list,processing):
        self.vars = vars_list
        self.processing = processing
        self.start='2015-12-01'
        self.end='2022-03-01'
        self.get_raw_data()

    def get_raw_data(self):
        self.dynam = []
        self.bench = []
        self.refer = []

        for var in self.vars:
            dynam = xr.open_dataset(f'data/processed/fcasts/calib/dynam_{var}_dates{self.start}to{self.end}.nc')
            bench = xr.open_dataset(f'data/processed/fcasts/raw/bench_{var}_dates{self.start}to{self.end}.nc')
            refer = xr.open_dataset(f'data/processed/fcasts/raw/refer_{var}_dates{self.start}to{self.end}.nc')
            
            self.dynam.append(dynam)
            self.bench.append(bench)
            self.refer.append(refer)

            del dynam
            del bench
            del refer

        self.dynam = xr.merge(self.dynam).rename({'initial_time':'N','number':'pdf','step':'time','latitude':'lat','longitude':'lon'})
        self.bench = xr.merge(self.bench).rename({'initial_time':'N','number':'pdf','step':'time','latitude':'lat','longitude':'lon'})
        self.refer = xr.merge(self.refer).rename({'initial_time':'N','number':'pdf','step':'time','latitude':'lat','longitude':'lon'})
        self.dynam = reassign_hdate_to_N(self.dynam).to_array().rename({'variable':'varOI'}).compute()
        self.bench = reassign_hdate_to_N(self.bench).to_array().rename({'variable':'varOI'}).compute()
        self.refer = reassign_hdate_to_N(self.refer).to_array().rename({'variable':'varOI'}).compute()

class processHC(processFC):
    """
    dynam: ECMWF
    bench: Climatology
    refer: ERA5
    """
    def __init__(self,vars_list,processing):
        super().__init__(vars_list,processing)


    def get_raw_data(self):
        self.dynam = []
        self.bench = []
        self.refer = []

        for var in self.vars:
            refer = xr.open_dataset(f'data/processed/hcasts/raw/refer_{var}_dates{self.start}to{self.end}.nc')
            bench = xr.open_dataset(f'data/processed/hcasts/raw/bench_{var}_dates{self.start}to{self.end}.nc')
            dynam = xr.open_dataset(f'data/processed/hcasts/calib/dynam_{var}_dates{self.start}to{self.end}.nc')

            self.dynam.append(dynam)
            self.bench.append(bench)
            self.refer.append(refer)

            del dynam
            del bench
            del refer

        self.dynam = xr.merge(self.dynam).rename({'initial_time':'N','number':'pdf','step':'time','latitude':'lat','longitude':'lon'})
        self.bench = xr.merge(self.bench).rename({'initial_time':'N','number':'pdf','step':'time','latitude':'lat','longitude':'lon'})
        self.refer = xr.merge(self.refer).rename({'initial_time':'N','number':'pdf','step':'time','latitude':'lat','longitude':'lon'})

        self.dynam = reassign_hdate_to_N(self.dynam).to_array().rename({'variable':'varOI'}).compute()
        self.bench = reassign_hdate_to_N(self.bench).to_array().rename({'variable':'varOI'}).compute()
        self.refer = reassign_hdate_to_N(self.refer).to_array().rename({'variable':'varOI'}).compute()

def reassign_hdate_to_N(da):
    da = da.unstack()
    da = da.rename({'N':'N_tmp',}).stack(N=['N_tmp','hdate'])
    da['N'] = [pd.to_datetime(N) + pd.DateOffset(years=hdate) for (N,hdate) in da.N.data]
    da = da.sortby(da.N).drop_duplicates('N')
    return da

if __name__ == '__main__':
    vars = ['10uv']
    hcY = processHC(vars,processing=True)
    raY = processRA(vars,processing=True)
    fcY = processFC(vars,processing=True)