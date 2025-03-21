import numpy as np
import xarray as xr
from tqdm import tqdm
from typing import List, Union
import xarray as xr

XArray = Union[xr.Dataset, xr.DataArray]
Dim = Union[List[str], str]

from joblib import Parallel, delayed

def numpy_crps_optimized_parallel(FC, RA, chunk_size=10, n_jobs=-1):
    """Parallel optimized chunked version using joblib"""
    M = FC.shape[-1]
    term1 = np.mean(np.abs(FC - RA[...,None]), axis=-1)
    term2 = np.zeros(FC.shape[:-1])
    
    # 这个函数将被并行处理
    def process_chunk(i, j):
        end_i = min(i + chunk_size, M)
        end_j = min(j + chunk_size, M)
        chunk_i = FC[..., i:end_i]
        chunk_j = FC[..., j:end_j]
        diff = np.abs(chunk_i[..., :, None] - chunk_j[..., None, :])
        return np.sum(diff, axis=(-1, -2))
    
    # 创建所有块对的索引
    chunk_indices = [(i, j) for i in range(0, M, chunk_size) for j in range(0, M, chunk_size)]
    
    # 并行处理所有块对
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(i, j) for i, j in chunk_indices
    )
    
    # 汇总结果
    for idx, result in enumerate(results):
        term2 += result
    
    crps_term2 = term2 / (2 * M * M)
    fcrps_term2 = term2 / (2 * M * (M - 1))
    
    crps = term1 - crps_term2
    fcrps = term1 - fcrps_term2
    
    return crps, fcrps, term1, crps_term2, fcrps_term2

def _probabilistic_broadcast(
    observations: XArray, forecasts: XArray, member_dim: str = "member"
) -> XArray:
    """Broadcast dimension except for member_dim in forecasts."""
    observations = observations.broadcast_like(
        forecasts.isel({member_dim: 0}, drop=True)
    )
    forecasts = forecasts.broadcast_like(observations)
    return observations, forecasts

def numpy_crps_optimized(FC, RA, chunk_size=10):
    """Optimized chunked version"""
    M = FC.shape[-1]
    term1 = np.mean(np.abs(FC - RA[...,None]), axis=-1)
    term2 = np.zeros(FC.shape[:-1])
    
    for i in tqdm(range(0, M, chunk_size),desc=f'       Computeing CRPS...',leave=False,position=2):
        end_i = min(i + chunk_size, M)
        chunk_i = FC[..., i:end_i]
        for j in range(0, M, chunk_size):
            end_j = min(j + chunk_size, M)
            chunk_j = FC[..., j:end_j]
            diff = np.abs(chunk_i[..., :, None] - chunk_j[..., None, :])
            term2 += np.sum(diff, axis=(-1, -2))
    
    crps_term2 = term2 / (2 * M * M)
    fcrps_term2 = term2 / (2 * M * (M - 1))
    
    crps = term1 - crps_term2
    fcrps = term1 - fcrps_term2
    
    return crps, fcrps, term1, crps_term2, fcrps_term2

def mean_crps_xskillscore(
    observations: XArray,
    climatology: XArray,
    forecasts: XArray,
    member_weights: XArray = None,
    member_dim: str = "member",
    dim: Dim = None,
    keep_attrs: bool = False,
    **kwargs
) -> XArray:
    
    observations, forecasts = _probabilistic_broadcast(
        observations, forecasts, member_dim=member_dim
    )
    forecasts = forecasts.transpose(member_dim,*observations.dims)
    crps, fcrps, term1, crps_term2, fcrps_term2 = xr.apply_ufunc(
        # numpy_crps_optimized,
        numpy_crps_optimized_parallel,
        forecasts,
        observations,
        input_core_dims=[[member_dim], []],
        output_core_dims=[[], [], [], [], []],  
        kwargs=kwargs,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )
    res = xr.concat([
        crps.expand_dims(score=['CRPS']), 
        fcrps.expand_dims(score=['FCRPS']), 
        term1.expand_dims(score=['TERM1']), 
        crps_term2.expand_dims(score=['TERM2']), 
        fcrps_term2.expand_dims(score=['FTERM2']),
    ],dim='score')
    return res.mean(dim,keep_attrs=keep_attrs)

def numpy_mse(FC,RA):
    return (FC.mean(-1) - RA)**2


def mean_bias_xskillscore(
    obs: XArray,
    cls: XArray,
    fcs: XArray,
    member_weights: XArray = None,
    member_dim: str = "member",
    lat_lon_dims: list = ['lat','lon'],
    dim: Dim = 'N',
    keep_attrs: bool = False,
    **kwargs
) -> XArray:

    obs, fcs = _probabilistic_broadcast(obs, fcs, member_dim=member_dim)
    bias = (obs - fcs.mean(member_dim)).mean(dim,keep_attrs=keep_attrs)
    bias = bias.expand_dims(score=['BIAS'])
    return bias

def mean_mse_xskillscore(
    observations: XArray,
    climatology: XArray,
    forecasts: XArray,
    member_weights: XArray = None,
    member_dim: str = "member",
    dim: Dim = None,
    keep_attrs: bool = False,
    **kwargs
) -> XArray:
    
    observations, forecasts = _probabilistic_broadcast(
        observations, forecasts, member_dim=member_dim
    )
    forecasts = forecasts.transpose(member_dim,*observations.dims)
    res = xr.apply_ufunc(
        numpy_mse,
        forecasts,
        observations,
        input_core_dims=[[member_dim], []],
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    ).expand_dims(score=['MSE'])
    
    return res.mean(dim, keep_attrs=keep_attrs)

def mean_ssr_xskillscore(
    observations: XArray,
    climatology: XArray,
    forecasts: XArray,
    member_weights: XArray = None,
    member_dim: str = "member",
    dim: Dim = None,
    keep_attrs: bool = False,
    **kwargs
) -> XArray:
    
    observations, forecasts = _probabilistic_broadcast(observations, forecasts, member_dim=member_dim)

    mse = mean_mse_xskillscore(observations,climatology,forecasts,member_weights,member_dim,dim,keep_attrs,kwargs=kwargs,).isel(score=0,drop=True)
    rmse = np.sqrt(mse)
    spread = np.sqrt(forecasts.var(member_dim).mean(dim))
    ssr = spread/rmse
    return ssr.expand_dims(score=['SSR'])

def get_scores(fc,ob,cl,funcs=None):
    res = []
    if funcs is None:
            funcs = [
                    mean_crps_xskillscore,
                    mean_mse_xskillscore,
                    mean_ssr_xskillscore,
                    ]
    for func in funcs:
        
        res.append(func(ob,cl,fc,dim='N',member_dim='pdf'))

    # [print(ele.score.data, ele.dims,ele.shape) for ele in res]
    res = xr.concat(res,dim='score')

    return res

def get_data_scores(data, n_iter=1, get_spectrum=False, funcs=None, dim='N', member_dim='pdf', n_jobs=-1):
    from sklearn.utils import resample
    from joblib import Parallel, delayed
    import numpy as np
    import xarray as xr
    
    random_seed = 42
    res = []
    ob = data['ERA5'].isel(pdf=0, drop=True).isel(model=0, drop=True)
    cl = data['Climatology'].isel(model=0, drop=True)
    print(f'ob: {ob.dims} {ob.shape}')
    print(f'cl: {cl.dims} {cl.shape}')
    boot = True if n_iter > 1 else False
    boots_order = []
    
    # 定义一个处理单个模型的函数
    def process_model(model, fc, i_N, boot_idx=None):
        try:
            res_tmp = get_scores(fc.isel(N=i_N), ob.isel(N=i_N), cl.isel(N=i_N), 
                                funcs=funcs)
            if boot:
                res_tmp = res_tmp.expand_dims(boot=[boot_idx])
            return model, res_tmp
        except Exception as e:
            print(f"处理模型 {model} 时出错: {e}")
            return model, None
    
    for i_boot in tqdm(range(n_iter), desc='Bootstrapping...', position=0, leave=True):

        if boot:
            print(f'i_boot={i_boot}')
            current_seed = random_seed + i_boot
            i_N = resample(range(ob.N.size), replace=True, random_state=current_seed)
            boots_order.append(i_N)
            leave = True
        else:
            i_N = range(ob.N.size)
            leave = False
        
        # 使用joblib并行处理所有模型
        model_items = [(model, fc) for model, fc in data.items()]
        
        # 并行处理模型
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_model)(model, fc, i_N, i_boot if boot else None)
            for model, fc in tqdm(model_items, desc=' Computing score...', position=1, leave=False)
        )
        
        # 收集有效结果
        boot_results = []
        for model, result in results:
            if result is not None:
                boot_results.append(result)
        
        if boot:
            # 合并当前bootstrap迭代的结果
            if boot_results:
                combined_boot = xr.combine_by_coords(boot_results)
                res.append(combined_boot)
        else:
            # 非bootstrap情况，直接将结果添加到列表
            res.extend(boot_results)
    
    # 合并所有结果
    if boot:
        if res:
            final_result = xr.combine_by_coords(res)
            return final_result, boots_order
    else:
        if res:
            final_result = xr.combine_by_coords(res)
            return final_result, boots_order
    
    return None, boots_order