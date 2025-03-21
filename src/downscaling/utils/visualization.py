import xarray as xr
import numpy as np
import time
import pickle as pkl
import os
import matplotlib.pyplot as plt
from math import ceil
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as ticker

def forPaper_curves(das,models,colors,labels,linestyles,title='Median of Bootstrap',save_plt_path='data/results/visualizations'):
    das = das.assign_coords(time=['1','2','3','4','5','6']).rename({'time':'LeadTime'})
    for score in das.score.data:
        plt.figure(dpi=300,figsize=(6.4/2,4.8/2))

        da_tmp = das.isel(LeadTime=range(0,6)).sel(score=score)
        if 'bootstrap' in da_tmp.dims:
            da_tmp = da_tmp.median('bootstrap')
        if 'lat' in da_tmp.dims:
            da_tmp = get_weighted_spatial_mean(da_tmp)
        for model,color,label,linestyle in zip(models,colors,labels,linestyles):
            da_tmp.sel(model=model).plot(color=color,label=label,linestyle=linestyle)

        handles, labels = plt.gca().get_legend_handles_labels()
        legend_handles = [handles[ele] for ele in range(len(models))]
        legend_handles.insert(3,mlines.Line2D([], [], color='none', label=''))
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        plt.grid(True)
        
        if score in 'MSE':
            plt.legend(handles=legend_handles, ncol=ceil(len(models)/3)) # , prop={'size': plt.rcParams['font.size']-2})

        if score == 'MSE':
            plt.ylabel(score + r' $(m^2s^{-2})$')
        elif score in ['CRPS','BIAS']:
            plt.ylabel(score + r' $(ms^{-1})$')
            plt.ylim([0.5,1.2])
        elif score == 'SSR':
            plt.ylabel(score + '(no unit)')
            plt.ylim([0,1.2])
        else:
            plt.ylabel(score)
        plt.xlabel('Lead weeks')
        plt.title('')
        plt.tight_layout()
        os.makedirs(save_plt_path,exist_ok=1)
        plt.savefig(f'{save_plt_path}/curve_{score}_w1_w6')
        plt.show()
    return

def forPaper_diff_maps(da_tmp,benchs,models,new_models,save_plt_path='data/results/visualizations'):
    """
    For diff: 
    benchs=['ECMWF','MLR'], models = ['$\\Delta$MLR-ECMWF','$\\Delta$CNN-ECMWF','$\\Delta$CNN-MLR']
    benchs=['ECMWF',r'$\\widetilde{MLR}$'], models = ['$\\Delta$$\\widetilde{MLR}$-ECMWF','$\\Delta$$\\widetilde{CNN}$-ECMWF', '$\\Delta$$\\widetilde{CNN}$-$\\widetilde{MLR}$']

    """
    da_tmp = da_tmp.assign_coords(time=[f'Week {w}' for w in range(1,7)]).rename({'time':'LeadTime'})

    os.makedirs(save_plt_path,exist_ok=1)
    if new_models is None:
        new_models = models
        
    diff_tmp = xr.concat([get_diff(da_tmp,ele) for ele in benchs],dim='model').sel(model=models).assign_coords(model=new_models)
    
    if 'bootstrap' in diff_tmp.dims:
        diff_tmp = get_non_significant_gp(diff_tmp)

    if diff_tmp.LeadTime.size>1:
        for lt in [[2],[3,4,5]]:
            week_info = "_".join([str(ele+1) for ele in lt])
            vis = Visualize(f'{save_plt_path}/maps_week_{week_info}')

            vis.map_panels(diff_tmp.squeeze().isel(LeadTime=lt),
                    model_type='diff',
                    row='LeadTime',
                    col='model',
                    row_list=None,
                    col_list=None,
                    vrange=[-10,10],
                    ) 
            print(f'Saved in {save_plt_path}/maps_week_{week_info}')
    else:
        vis = Visualize(f'{save_plt_path}/maps_week')

        vis.map_panels(da_tmp.squeeze(),
                model_type='model',
                row='LeadTime',
                col='model',
                row_list=None,
                col_list=None,
                vrange=[0,10],
                ) 
        
        vis.map_panels(diff_tmp.squeeze(),
                model_type='diff',
                row='LeadTime',
                col='model',
                row_list=None,
                col_list=None,
                vrange=[-100,100],
                ) 
        print(f'Saved in {save_plt_path}/maps_week')

#%%
def select_domain(domain):
    if domain=='large_scale':
        lat1, lat2, lon1, lon2 = 20,80,-120,40 # large_scale
    elif domain=='large_scale_latex':
        lat1, lat2, lon1, lon2 = 20,80,-120,42.5 # large_scale
    elif domain=='domain_eastwards':
        lat1, lat2, lon1, lon2 = 20,80,-90,70 # domain_eastwards
    elif domain=='northern_extratropics':
        lat1, lat2, lon1, lon2 = 23.5,66.5,-180,-180 # northern_extratropics
    elif domain=='Europe':
        lat1, lat2, lon1, lon2 = 34,73,-13,40
    elif domain=='Germany':
        lat1, lat2, lon1, lon2 = 47.3,54.6,6.4,14.9
    elif domain=='Spain':
        lat1, lat2, lon1, lon2 = 37.0,43.5,-10.0,3.7
    elif domain=='UK':
        lat1, lat2, lon1, lon2 = 49.0,60.0,-10.0,4.0
    elif domain=='Ireland':
        lat1, lat2, lon1, lon2 = 55.0,56.0,-10.5,-9.5
    elif domain=='France':
        lat1, lat2, lon1, lon2 = 43.0,51.0,-5.5,7.3
    elif domain=='Marseille':
        lat1, lat2, lon1, lon2 = 43.1,43.5,5.2,5.5
    elif domain=='North Sea':
        lat1, lat2, lon1, lon2 = 51.0,61.0,-4.4,12.0
    elif domain=='debug':
        lat1, lat2, lon1, lon2 = 34,35,0,1
    else:
        raise ValueError("Plz select a domain from large_scale, domain_eastwards, northern_extratropics")
    return lat1, lat2, lon1, lon2

def to_Domain(ds,domain):
    lat1, lat2, lon1, lon2 = select_domain(domain)
    dsi = ds.sel(lat=slice(lat2,lat1),lon=slice(lon1, lon2))
    return dsi

def maps(ax,da):
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS,)
    ax.set_extent([da.lon.min().data,da.lon.max().data,da.lat.min().data,da.lat.max().data],ccrs.PlateCarree())
    return

DataTypeDict = {
    'Data_raX_v': '09y RA', 'Data_fcX_v': '05y FC', 'Data_fcX_6year_HCs': '05y HC', 'Data_fcX_20year_HCs': '26y HC', 'Data_fcX_20year_HCs_ENM': '26y HC ENM', 
    'Data_raY_v': '09y RA', 'Data_fcY_v': '05y FC', 'Data_fcY_6year_HCs': '05y HC', 'Data_fcY_20year_HCs': '26y HC', 'Data_fcY_20year_HCs_ENM': '26y HC ENM', 
    'Data_raX_v_pre': '09y RA', 'Data_fcX_v_pre': '05y FC', 'Data_fcX_6year_HCs_pre': '05y HC', 'Data_fcX_20year_HCs_pre': '26y HC', 
    'Data_raY_v_pre': '09y RA', 'Data_fcY_v_pre': '05y FC', 'Data_fcY_6year_HCs_pre': '05y HC', 'Data_fcY_20year_HCs_pre': '26y HC', 
    'Data_raY_t': '09y RA_t',
}

CalibDict = {
    'calib_MVA_tw1': 'MVA_tw1', 
    'calib_no_calib': 'no_calib',

    'calib_MVA_tw1':'MVA_tw1', 
    'calib_MVA_TGL_PDF':'MVA_TGL_PDF', 
    'calib_MVA_TGL_NUM':'MVA_TGL_NUM', 
    'calib_MVA_TGL_PDF_HL':'MVA_TGL_PDF_HL', 
    'calib_MVA_TGL_NUM_HL':'MVA_TGL_NUM_HL', 
    'calib_MVA_NVG_PDF':'MVA_NVG_PDF', 
    'calib_MVA_NVG_NUM':'MVA_NVG_NUM', 
    
    'calib_MVA_TGL_PDF_HL2':'MVA_TGL_PDF_HL2', 
    'calib_MVA_TGL_PDF2':'MVA_TGL_PDF2', 
}

# Pre-processing using how many years' climatology
CL_yearDict = {
    '15y': '15y',
    '35y': '35y',
}

PT_Dict = {
    'DeseasonalityDetrendScalingRefSelfTraining':'PT1', 
    'DeseasonalityDetrendScalingTraining':'PT2',
    'DeseasonalityDetrendScalingVCL':'PT3',
    'ScalingCL':'ScalingCL',
    'ScalingRA':'ScalingRA',
    'Nothing':'Nothing',
}

def rename_da(da):
    map_dict = {
        **{f'{type_}({model})': f'{model}' for type_ in ['Direct', 'Stocha', 'Quantile', 'Diffusion'] for model in ['MLR', 'CNN']},
        **{f'HY({type_}({model}))': f'HY({model})' for type_ in ['Direct', 'Stocha', 'Quantile', 'Diffusion'] for model in ['MLR', 'CNN']}
    }
    da['model'] = [map_dict.get(model, model) for model in da.model.data]
    # da = da.sel(model=da.model.data != 'Climatology')
    return da

def get_diff(da, model_ref):
    da_diff = xr.concat([
        ((da.sel(model=model) - da.sel(model=model_ref)) / da.sel(model=model_ref)).expand_dims(model=[0]).assign_coords(model=[rf'$\Delta${model}-{model_ref}'])
        for model in da.model.data if model != model_ref], dim='model') * 100
    return da_diff

def get_asb_diff(da, model_ref):
    da_diff = xr.concat([
        (da.sel(model=model) - da.sel(model=model_ref)).expand_dims(model=[0]).assign_coords(model=[rf'$\Delta${model}-{model_ref}'])
        for model in da.model.data if model != model_ref], dim='model')
    return da_diff

def get_non_significant_gp(diff):
    """
    Get significant level from bootstrap
    """

    metric_higher_better = ['CRPSS','FCRPSS', 'ACC','TERM2','FTERM2']
    metric_higher_better.extend([f'{nbins} at resolution at {tercile}' for nbins in [3,6,10] for tercile in ['lower','middle','upper','P10','P90']])
    diff_metric_h_b = [ele for ele in diff.score.data if ele in metric_higher_better ]
    diff_metric_l_b = [ele for ele in diff.score.data if ele not in metric_higher_better ]
    diff_h_b = diff.sel(score=diff_metric_h_b)
    diff_l_b = diff.sel(score=diff_metric_l_b)
    # significantly better for higher better metrics
    p_h_b_sig_better = diff_h_b.where(diff_h_b<0).count(dim='bootstrap')/diff_h_b.bootstrap.size # expected fraction < 0.05
    p_h_b_sig_worser = diff_h_b.where(diff_h_b>0).count(dim='bootstrap')/diff_h_b.bootstrap.size
    p_l_b_sig_better = diff_l_b.where(diff_l_b>0).count(dim='bootstrap')/diff_l_b.bootstrap.size
    p_l_b_sig_worser = diff_l_b.where(diff_l_b<0).count(dim='bootstrap')/diff_l_b.bootstrap.size
    p_sig_better = xr.concat([p_h_b_sig_better,p_l_b_sig_better],dim='score')
    p_sig_worser = xr.concat([p_h_b_sig_worser,p_l_b_sig_worser],dim='score')
    
    confi_better_99 = p_sig_better>0.01
    confi_worser_99 = p_sig_worser>0.01
    confi_better_95 = p_sig_better>0.05
    confi_worser_95 = p_sig_worser>0.05
    confi_better_90 = p_sig_better>0.10
    confi_worser_90 = p_sig_worser>0.10

    return diff.mean(dim='bootstrap').assign_coords(stippling_mask_better_99=confi_better_99,stippling_mask_worser_99=confi_worser_99)\
        .assign_coords(stippling_mask_better_95=confi_better_95,stippling_mask_worser_95=confi_worser_95)\
        .assign_coords(stippling_mask_better_90=confi_better_90,stippling_mask_worser_90=confi_worser_90)

#%% Func Config
class Config:
    def __init__(self):
        self.CrossValidation = 'WithNestedCV'
        self.ForecastVersions = [
            'Direct', 
            'Stochastic',
            'StochasticMem0','StochasticEnsM',
            'Stochastic_mu0',
            'Quantile_10Quant',
            'Quantile_10QuantExtreme',
            'Diffusion',
            'VAE',
            ]
        self.calib_methods = [
            'calib_MVA_tw1',
            'calib_MVA_tw1',
            'calib_no_calib',
            ]
        self.FcProcessings = ['CL']
        self.trainingDomain = 'EADToEAD'
        
        self.CL_years = [
            '15y',
            '35y',
        ]
        self.data_types = ['Data_fcY_v', 'Data_fcY_6year_HCs', 'Data_fcY_20year_HCs']

        self.bootstrap = 'None'
        self.scores = ['ACC', 'MSE', 'CRPS', 'TERM1', 'TERM2', 'SSR', 'BIAS']
        self.dir = 'debug_DJFs_v10/DeseasonalityDetrendScalingRefSelfTraining_vars_both_FC_grid_wise//save_plt/z500/'
        self.member_select = False
        self.quantile_select = False
        self.trnType = 'RA'
        self.models = 'ECMWF,ERA5,Climatology,MytorchLR,MySimpleResCNN'
        self.Nquantiles = '10Quant' # 10Quant 10QuantExtreme
        self.n_iters = None
        self.n_keep = None
        self.n_pdf = None
        self.trainPntOrSeq = 'PntReg' # PntLag
        self.scores_list = None
        self.SeqPntWeek = None
        self.get_ssim = False
    
    def get_path(self, FV,calib, CL_year,data_type, i_fold):
        FV_tmp = None
        if FV in ['VAE']:
            FV_tmp = FV
            FV  = f'{FV}_z_{self.z_size}_sig_{self.vae_sigma}'

        if self.bootstrap in ['boot','bootInterval','bootN']:
            data_type = f'score_boot_{self.bootstrap}_{data_type}_{self.n_iters}_Keep{self.n_keep}In{self.n_pdf}'
        else:
            data_type = f'score_{data_type}'

        if self.scores_list is not None:
            data_type = f'{data_type}_{self.scores_list}'

        # Use attributes to construct the path dynamically
        if self.trainingDomain=='EADToEAD':
            path = f'{self.dir}/{self.CrossValidation}/trainingModels_{self.models}/{FV}_{self.trainPntOrSeq}/{calib}_CL_years_{CL_year}_trnType_{self.trnType}/WithNestedCV_Fold_{i_fold}'
        else:
            path = f'{self.dir}/{self.CrossValidation}/trainingModels_{self.models}/{FV}_{self.trainPntOrSeq}/{self.trainingDomain}_{calib}_CL_years_{CL_year}_trnType_{self.trnType}/WithNestedCV_Fold_{i_fold}'

        if self.trainPntOrSeq in ['SeqPnt']:
            path = f'{path}_{self.SeqPntWeek}'

        if FV_tmp in ['VAE']:
            path = f'{path}/z{self.z_size}_sig_{self.vae_sigma}_{self.DropOutList}_{self.dropout_rate}/'

        if self.member_select:
            path = f'{path}/select_member/{data_type}'

        elif self.quantile_select:
            path = f'{path}/select_quantile/{data_type}'
            
        else:
            path = f'{path}/{data_type}'
        
        if self.get_ssim:
            path = f'{path}_SSIM'

        return path

    def get_data_path(self, FV,calib, CL_year,data_type, i_fold):
        data_type = f'{data_type}'
        return f'{self.dir}/{self.CrossValidation}/trainingModels_{self.models}/{FV}_{self.trainPntOrSeq}/{calib}_CL_years_{CL_year}_trnType_{self.trnType}/WithNestedCV_Fold_{i_fold}/{data_type}'


def get_birth_time(file_path):
    # 获取文件的状态信息
    file_stats = os.stat(file_path)

    # 获取文件创建时间
    # 对于不同操作系统，使用不同的方法获取创建时间
    if hasattr(file_stats, 'st_birthtime'):  # macOS
        creation_time = file_stats.st_birthtime
    else:  # Unix-like systems
        creation_time = file_stats.st_ctime

    # 转换为可读格式
    readable_time = time.ctime(creation_time)
    print(f'{readable_time}: {file_path}')
    return 

def get_units(metric,model_type):

    metrics_dict = {'ACC_N_mean':'ACC',
    'CRPSS_N_mean':'CRPSS',
    '5 at brier score at lower':'Brier Score at lower tercile', 
    '5 at reliability at lower':'Reliability at lower tercile',
    '5 at resolution at lower':'Resolution at lower tercile',
    '5 at brier score at middle':'Brier Score at middle tercile', 
    '5 at reliability at middle':'Reliability at middle tercile',
    '5 at resolution at middle':'Resolution at middle tercile',
    '5 at brier score at upper':'Brier Score at upper tercile', 
    '5 at reliability at upper':'Reliability at upper tercile',
    '5 at resolution at upper':'Resolution at upper tercile',
                   }
    
    units_dict = {
        'MSE': rf'$(m/s)^2$',
        'RMSE': '(m/s)',
        'CRPS': '(m/s)',
        'FCRPS': '(m/s)',
    }
    if metric not in list(metrics_dict.keys()):
        metric = metric
    else:
        metric = metrics_dict[metric]

    if model_type=='diff':
        return "%"
    else:
        if metric in list(units_dict.keys()):
            if isinstance(metric,str):
               return units_dict[metric]
            else:
               return units_dict[str(metric)]
               
        else:
           return metric
        
def get_cmap(metric,model_type):
    """
    Get color map for metric
    """
    if model_type!='diff':
        if metric=='Mean Forecasts':
            cmap = plt.cm.GnBu
        elif metric in ['ACC_mean_N','ACC_N_mean','CRPSS_N_mean','CRPSS_mean_N','FCRPSS_N_mean','FCRPSS_mean_N','ACC','CRPSS']:
            cmap = plt.cm.RdBu
        elif metric=='Spread':
            cmap = plt.cm.PuRd
        elif 'resolution' in metric or 'term2' in metric:
            cmap = plt.cm.Blues
        elif 'reliability' in metric or 'brier' in metric or metric=='CRPS' or 'term1' in metric:
            cmap = plt.cm.Blues
            # cmap = cmap.reversed()
        elif metric in ['RRMSE','RMSE']:
            cmap = plt.cm.Reds
            cmap = cmap.reversed()
        else:
            cmap = plt.cm.RdBu
    else:
        cmap = plt.cm.PiYG

    metric_higher_better = ['ACC','CRPSS','CRPSS_N_mean','FCRPSS_N_mean', 'ACC_N_mean','term2','R2']
    metric_higher_better.extend([f'{nbins} at resolution at {tercile}' for nbins in [3,5,6,10] for tercile in ['lower','middle','upper','P10','P90']])
    if not (metric in metric_higher_better or ('resolution' in metric)):
        cmap = cmap.reversed()
    return cmap

def get_cbar_posi(axes,vertical=1):
    if vertical:
        if axes.shape[1]==1:
            # print("shape==1")
            w_subplot = axes.flat[-1].get_position().xmax - axes.flat[-1].get_position().xmin
            width = w_subplot/10
        else:
            width = axes[0,-1].get_position().xmin - axes[0,-2].get_position().xmax

        left = axes.flat[-1].get_position().xmax + width

        height_comm = np.max([ax.get_position().ymax for ax in axes.flat]) \
                    - np.min([ax.get_position().ymin for ax in axes.flat])
        bottom_comm = np.min([ax.get_position().ymin for ax in axes.flat])

        position_comn = [left, bottom_comm, width, height_comm]
    else:
        if axes.shape[1]==1:
            # print("shape==1")
            w_subplot = axes.flat[-1].get_position().ymax - axes.flat[-1].get_position().ymin
            height = np.abs(w_subplot/10)
        else:
            height = axes[-1,0].get_position().ymin - axes[-2,0].get_position().ymax

        left = np.min([ax.get_position().xmin for ax in axes.flat])
        bottom_comm = np.max([ax.get_position().ymax for ax in axes.flat]) + height*1.5 

        width = np.max([ax.get_position().xmax for ax in axes.flat]) \
                    - np.min([ax.get_position().xmin for ax in axes.flat])

        position_comn = [left, bottom_comm, width, height]
    return position_comn

def get_weighted_spatial_mean(da):
    # print(da.lat)
    weights = np.cos(np.deg2rad(da.lat))
    weights = weights / weights.sum()
    da_weighted = da.weighted(weights).mean(['lat','lon'])
    return da_weighted


def get_cb_label(metric,model_type):
    metrics_dict = {'ACC_N_mean':'ACC',
    'CRPSS_N_mean':'CRPSS',
    '5 at brier score at lower':'Brier Score at lower tercile', 
    '5 at reliability at lower':'Reliability at lower tercile',
    '5 at resolution at lower':'Resolution at lower tercile',
    '5 at brier score at middle':'Brier Score at middle tercile', 
    '5 at reliability at middle':'Reliability at middle tercile',
    '5 at resolution at middle':'Resolution at middle tercile',
    '5 at brier score at upper':'Brier Score at upper tercile', 
    '5 at reliability at upper':'Reliability at upper tercile',
    '5 at resolution at upper':'Resolution at upper tercile',
                   }
    
    units_dict = {
        'MSE': r'$(m^2s^{-2})$',
        'RMSE': r'$(ms^{-1})$',
        'CRPS': r'$(ms^{-1})$',
        'FCRPS': r'$(ms^{-1})$',
    }
    if metric not in list(metrics_dict.keys()):
        metric = metric
    else:
        metric = metrics_dict[metric]

    if model_type=='diff':
        return rf"$\Delta_r${metric} (%)"
    else:
        if metric in list(units_dict.keys()):
           return f'{metric} {units_dict[str(metric)]}'
        else:
           return metric

# New dim
def plot_maps(da,metric,row,col,model_type,save_path,vrange=None,maps=True,figsize=None):
    num_panels = da[row].size * da[col].size
    if da.isnull().any():
        print('nan in plot_maps',da.dims,da.shape)
        
    if da[row].size==0 or da[col].size==0: return
    vertical = 1
    if figsize is None:
        if vertical:
            # figsize=(6.4*da[col].size/da[row].size,4.8)
            figsize=(6.4*1.5,4.8*da[row].size/da[col].size*1.5*1.1)
            # figsize=(6.4,4.8*da[row].size/da[col].size*1.1)
        else:
            figsize=(6.4,4.8*da[row].size/da[col].size)

        if num_panels==1:
            figsize=(6.4*1.5/2,4.8*1/2*1.5*1.1)

    if vrange is not None:
        vmin,vmax = vrange
    else:
        if model_type=='diff' and ('resolution' in metric or 'reliability' in metric or 'brier' in metric):
            vmin = -0.02
            vmax = 0.02
        elif 'ACC' in metric:
            if model_type!='diff':
                vmin = -0.4
                vmax = 0.4
            else:
                vmin = -0.16
                vmax = 0.16
        elif 'CRPSS' in metric:
            if model_type!='diff':
                vmin = -0.08
                vmax = 0.08
            else:
                vmin = -0.08
                vmax = 0.08
        elif metric=='MSE' :
            if model_type=='diff':
                vmin, vmax = -5,5
            else:
                vmin, vmax = 0,10
        elif metric=='Residual' :
            if model_type=='diff':
                vmin, vmax = -1,1
            else:
                vmin, vmax = -0.8,0.8

        elif metric=='CRPS' :
            if model_type=='diff':
                vmin, vmax = -0.1,0.1
            else:
                vmin, vmax = 0.5,2

        elif metric=='R2' :
            if model_type=='diff':
                vmin = -1
                vmax = 1
            else:
                vmin = -1
                vmax = 1

        else:
            vmin, vmax = None,None

    # Define subplots
    if 'resolution' in metric or 'reliability' in metric or 'brier' in metric:
        da = da.isel(lat=range(1,da.lat.size-1),lon=range(1,da.lon.size-1))
    if 'domain' in da.dims:
        da = da.isel(domain=0)

    own_checker = 0 # Just for ablation study
    if own_checker:
        g = da.plot(row='model',col='leadt',robust=True,
            cmap = get_cmap(metric,model_type),
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            subplot_kws={"projection": ccrs.PlateCarree()},
           )

        for i_r, row_data in enumerate(da[row].data):
            for i_c, col_data in enumerate(da[col].data):
                ax = g.axes[i_r,i_c]
                ax.coastlines()
                ax.add_feature(cfeature.BORDERS)
                ax.set_extent([da.lon.min().data,da.lon.max().data,da.lat.min().data,da.lat.max().data])
                # 计算mean(['lat','lon'])的值
                mean_value = da.isel({row:i_r,col:i_c}).mean().data
                ax.set_title(rf"$\mu$={mean_value:.2f} {get_units(metric,model_type)}",loc='right',fontsize=plt.rcParams['font.size']-2)

    else:
        # da = da.assign_coords(leadt=[f'Week {ele+1}' for ele in da.leadt.data])
        # da = da.assign_coords(leadt=[f'Week {ele}' for ele in da.leadt.data])
        subplot_kw={'projection': ccrs.PlateCarree()} if maps else {}
        fig, axes = plt.subplots(
            da[row].size,
            da[col].size,
            figsize=figsize,
            dpi=200,
            subplot_kw=subplot_kw,
            sharex=True, 
            sharey=True, 
            )
        # if model_type=='diff':
        #     plt.subplots_adjust(wspace=0.05, hspace=0.05)
        # else:
        #     plt.subplots_adjust(wspace=0.05, hspace=0.15)
        plt.subplots_adjust(wspace=0.05, hspace=0.12)
            
        # print((6.4*da[col].size/da[row].size,4.8))
        # Make axes as np.array
        if da[row].size==1 and da[col].size!=1:
            axes = axes[None,:]
        elif da[row].size!=1 and da[col].size==1:
            axes = axes[:,None]
        elif da[row].size==1 and da[col].size==1:
            axes = np.array(axes)[None,None]
        else:
            pass
        # plot subplots one-by-one
        for i_r, row_data in enumerate(da[row].data):
            for i_c, col_data in enumerate(da[col].data):
                ax = axes[i_r,i_c]

                data_mean = get_weighted_spatial_mean(da.isel({row:i_r,col:i_c})).data
                data_mean = np.mean(data_mean)
                ax.set_title(rf"$\mu$={data_mean:.2f} {get_units(metric,model_type)}",loc='right',fontsize=plt.rcParams['font.size']-2)
                
                # if i_r==0: ax.set_title(col_data, loc='left',fontsize=plt.rcParams['font.size']-2)
                if i_r==da[row].size-1: ax.set_xlabel(col_data)
                if i_c==0: ax.set_ylabel(row_data)

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_extent([da.lon.min(), da.lon.max(), da.lat.min(), da.lat.max()], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE,linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linewidth=0.5)

                cmap = get_cmap(metric,model_type)
                da_ax = da.isel({row:i_r,col:i_c}).plot(
                    ax=ax, 
                    transform=ccrs.PlateCarree(),
                    levels=100,
                    cmap=cmap,
                    # robust=True,
                    add_colorbar=False,
                    add_labels=False,
                    cbar_kwargs={},
                    vmin=vmin,
                    vmax=vmax,
                    )

                # Add different significant levels
                if 'stippling_mask_better_99' in da.coords and model_type=='diff':
                    for i_sg,(dim,marker) in enumerate(zip(['stippling_mask_better_99','stippling_mask_better_95','stippling_mask_better_90'],['+','2','|'])):
                        no_sig_99 = np.logical_and(da.stippling_mask_better_99,da.stippling_mask_worser_99).isel({row:i_r,col:i_c})
                        no_sig_95 = np.logical_and(da.stippling_mask_better_95,da.stippling_mask_worser_95).isel({row:i_r,col:i_c})
                        no_sig_90 = np.logical_and(da.stippling_mask_better_90,da.stippling_mask_worser_90).isel({row:i_r,col:i_c})
                        no_sig = [  np.logical_not(no_sig_99),   # signifcantly better at 99%
                                    np.logical_xor(no_sig_99,no_sig_95), # signifcantly better at 95%
                                    np.logical_xor(no_sig_95,no_sig_90), # signifcantly better at 90%
                                    ]

                        lons, lats = np.meshgrid(da.lon, da.lat)  # 获取经纬度坐标
                        # stipple_indices = np.argwhere(np.logical_and(da[dim].isel({row:i_r,col:i_c}).values==False,da[dim.replace('stippling_mask_better','stippling_mask_worser')].isel({row:i_r,col:i_c}).values==False))
                        # stipple_indices = np.argwhere(np.logical_or(da[dim].isel({row:i_r,col:i_c}).values==True,da[dim.replace('stippling_mask_better','stippling_mask_worser')].isel({row:i_r,col:i_c}).values==True))
                        
                        stipple_indices = np.argwhere(no_sig[i_sg].data)

                        stipple_lons = lons[tuple(stipple_indices.T)]
                        stipple_lats = lats[tuple(stipple_indices.T)]
                        ax.scatter(stipple_lons, stipple_lats, transform=ccrs.PlateCarree(), marker=marker,s=30,color='grey',zorder=300,linewidths=0.5)

                # Add different significant levels
                if 'alpha_001' in da.coords and model_type=='diff':
                    for i_sg,(alpha,marker) in enumerate(zip(['alpha_001','alpha_005','alpha_010'],['+','2','|'])):
                        sig = da[alpha].isel({row:i_r,col:i_c})

                        lons, lats = np.meshgrid(sig.lon, sig.lat)  # 获取经纬度坐标
                        stipple_indices = np.argwhere(sig.data)

                        stipple_lons = lons[tuple(stipple_indices.T)]
                        stipple_lats = lats[tuple(stipple_indices.T)]
                        ax.scatter(stipple_lons, stipple_lats, transform=ccrs.PlateCarree(), marker=marker,s=30,color='grey',zorder=300,linewidths=0.5)
                    
        # Add the common colorbar for A and B and C
        position_comn = get_cbar_posi(axes,vertical)
        # print('position_comn=',position_comn)
        cbar_ax_common = fig.add_axes(position_comn)
        if model_type=='diff':
            FormatStrFormatter = '%d'
        else:
            FormatStrFormatter = '%.2f'

        if vertical:
            cbar_common = fig.colorbar(da_ax, cax=cbar_ax_common, orientation="vertical",label=get_cb_label(metric,model_type))
            cbar_common.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(FormatStrFormatter))
            cbar_common.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        else:
            cbar_common = fig.colorbar(da_ax, cax=cbar_ax_common, orientation="horizontal",label=get_cb_label(metric,model_type))
            cbar_common.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(FormatStrFormatter))
            cbar_common.ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        
    # Save figures
    title_info = [str(row_data) for row_data in da[row].data]
    # title_info.extend([col_data for col_data in da[col].data])
    title_info = ''.join(title_info)
    title_info = title_info.replace(' ','').replace('$\Delta_r$','diff')


    # fig.tight_layout()
    os.makedirs(save_path,exist_ok=1)
    # save_name = get_cb_label(metric,'model').replace('\n',' ').replace(' ','_')
    save_name = f'{model_type}{metric}'
    if model_type!='diff':
        save_name = f'{title_info}_{save_name}'
    plt.savefig(f"{save_path}/{model_type}_{save_name}.png",bbox_inches='tight')
    return

class Visualize():
    def __init__(self,save_path):
        os.makedirs(save_path,exist_ok=1)
        self.save_path = save_path

    def cuv_plot(self,scores, hue='model'):
        if 'lat' in scores.coords:
            scores = scores.mean(['lat','lon'])
        scores = scores.squeeze()
        
        for score in scores.score.data:
            plt.figure()
            for model in scores[hue].data:
                if 'HY' not in model:
                    ax = scores.sel({'score':score,hue:model}).plot(label=model)
                if f'HY({model})' in scores[hue].data:
                    plt.plot(scores.Dataset.data,scores.sel(score=score,model=f'HY({model})'), label=f'HY({model})',linestyle='dashed', color=ax[0].get_color())

            plt.grid(True)
            plt.title('')
            plt.xlabel('Lead weeks')
            plt.ylabel(score)
            plt.legend()
            plt.savefig(f'{self.save_path}/vis_cuv_{score}.png')

        return
    
    def map_panels(self,scores,model_type,row,col,row_list=None,col_list=None,vrange=None,figsize=None):
        # model_type: model, diff
        if row_list is None:
            row_list = scores[row].data

        if col_list is None:
            col_list = scores[col].data

        figsize=(6.4/2*scores[col].size,4.8/2*scores[row].size)
        if isinstance(scores.score.data,list):

            for score in scores.score.data:
                plt.figure(dpi=300)
                plot_maps(scores.sel(score=score).sel({row:row_list,col:col_list}),
                        score,
                        row=row,
                        col=col,
                        model_type=model_type,
                        save_path=f'{self.save_path}/vis_map_{model_type}',
                        vrange=vrange,
                        maps=True,
                        figsize=figsize,
                        #   figsize=(6.4,4.8/scores[col].size*scores[row].size),

                        )
        else:
            score = scores.score.data
            plt.figure(dpi=300)
            plot_maps(scores.sel({row:row_list,col:col_list}),
                    score,
                    row=row,
                    col=col,
                    model_type=model_type,
                    save_path=f'{self.save_path}/vis_map_{model_type}',
                    vrange=vrange,
                    maps=True,
                    figsize=figsize,
                    #   figsize=(6.4,4.8/scores[col].size*scores[row].size),

                    )
        return

