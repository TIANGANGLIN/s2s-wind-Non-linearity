import pickle as pkl
from tqdm import tqdm
import numpy as np
import os
from downscaling.utils.metrics import get_data_scores,mean_crps_xskillscore,mean_mse_xskillscore,mean_ssr_xskillscore,mean_bias_xskillscore

def select_domain(domain):
    if domain=='large_scale':
        lat1, lat2, lon1, lon2 = 20,80,-120,40 # large_scale
    elif domain=='Europe':
        lat1, lat2, lon1, lon2 = 34,73,-13,40
    else:
        raise ValueError("")
    return lat1, lat2, lon1, lon2

def to_Domain(ds,domain):
    lat1, lat2, lon1, lon2 = select_domain(domain)
    dsi = ds.sel(lat=slice(lat2,lat1),lon=slice(lon1, lon2))
    return dsi

def load_reference_data(ref_ben_path, domain, dims_order):
    """Load and process reference data (ERA5, ECMWF, Climatology)"""
    Y_class = pkl.load(open(ref_ben_path, 'rb'))
    
    if 'y_initial_class' in ref_ben_path:
        reference_data = {
            'ERA5': Y_class.refer,
            'Climatology': Y_class.bench
        }
    else:
        # Create dictionary of reference datasets
        reference_data = {
            'ECMWF': Y_class.dynam,
            'ERA5': Y_class.refer,
            'Climatology': Y_class.bench
        }
    
    # Process each reference dataset
    processed_data = {}
    for name, data in reference_data.items():
        processed = data.unstack()\
                    .expand_dims(model=[name])
        processed = to_Domain(processed,domain)
        processed = processed.transpose(*dims_order)
        processed_data[name] = processed
    
    del Y_class
    return processed_data

def process_model_data(model_paths, domain, dims_order):
    """Load and process model prediction data"""
    processed_data = {}
    
    for model_name, model_path in tqdm(model_paths.items()):
        if not os.path.exists(model_path):
            print(f'Not exists {model_path}')
            continue
        else:
            print(f'Loading {model_path}')
            
        # try:
        if 1:
            # Load and initial processing
            da = pkl.load(open(model_path, 'rb'))
            da = (da.unstack()
                 .expand_dims(model=[model_name])
                 .pipe(lambda x: to_Domain(x, domain)))
            # da = da.isel(N=range(50))
            
            # Handle special dimensions
            if 'noise_step' in da.dims:
                da = da.isel(noise_step=-1)
                
            if 'quantile' in da.dims:
                da = da.rename({'pdf': 'pdf_tmp'})\
                     .stack(pdf=['pdf_tmp', 'quantile'])
                da = da.assign_coords(pdf=range(da.pdf.size))
            
            processed_data[model_name] = da.transpose(*dims_order)
            
        # except Exception as e:
        #     raise ValueError(f'Error processing {model_path}: {str(e)}')
    
    return processed_data

def find_common_indices(datasets):
    """Find common N indices across all datasets"""
    N_lists = [ds.N.data for ds in datasets.values()]
    N_intersection = N_lists[0]
    
    for N in N_lists[1:]:
        N_intersection = np.intersect1d(N_intersection, N)
        
    return N_intersection

def get_data_dict(ref_ben_path, model_paths, domain='Europe'):
    """
    Main function to load and process all data.
    
    Args:
        ref_ben_path (str): Path to reference benchmark data
        model_paths (dict): Dictionary of model paths
        domain (str): Domain name ('Europe' or 'large_scale')
    
    Returns:
        dict: Dictionary containing all processed datasets
    """
    for model_name, model_path in tqdm(model_paths.items()):
        if not os.path.exists(model_path):
            print(f'Not exists {model_path}')

    # Get dimension order from reference data
    temp_data = pkl.load(open(ref_ben_path, 'rb'))
    dims_order = temp_data.dynam.unstack().expand_dims(model=['place']).dims
    del temp_data
    
    # Load and process all data
    data = load_reference_data(ref_ben_path, domain, dims_order)
    model_data = process_model_data(model_paths, domain, dims_order)
    data.update(model_data)
    
    # Find common indices and align all datasets
    N_common = find_common_indices(data)
    print(f'Common N indices: {len(N_common)}, from {N_common[0]} to {N_common[-1]}')
    
    # Align all datasets to common indices
    for model_name, dataset in data.items():
        data[model_name] = dataset.sel(N=N_common).transpose(*dims_order)
        print(f' {model_name}: {data[model_name].dims}, {data[model_name].shape}')
    
    return data

def process_bootstrap_data(data, n_pdf=10):
    """Process bootstrap data with quantiles"""
    for model, da in data.items():
        if da.pdf.size <= n_pdf:
            continue
        tmp = da.quantile([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95], dim='pdf')
        data[model] = tmp.rename({'quantile': 'pdf'})
        print(f'{model}: {data[model].dims} {data[model].shape}')
    return data

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Results & Data storage
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--dir', type=str, default='data/results/WithOptuna')
    args = parser.parse_args()

    dir = args.dir

    n_iter = 1000
    # n_iter = 1
    ref_type = {}
    ref_type['in_tnvl'] = 'y_initial_class'
    ref_type['in_test'] = 'y_initial_class'
    ref_type['fc_test'] = 'y_fc_class'
    ref_type['hc_test'] = 'y_hc_class'
    # evalType = 'hc_test'
    for evalType in [
        # 'in_tnvl','in_test',
        'hc_test',
        # 'fc_test',
                     ]:
        ref_ben_path=f'{dir}/{ref_type[evalType]}'
        # for fold in [2,1,0]:
        for fold in [args.fold]:
            save_path = f'{dir}/skills/Pixel_score_fold{fold}_n_iter_{n_iter}_{evalType}.pkl'
            os.makedirs(os.path.dirname(save_path),exist_ok=1)
            # if os.path.exists(save_path):continue
            model_paths = {
                'MLR': f'{dir}/MLR/initial_model/reconstruct/initial_on_TrnOnRA_reconstruct_EvalFull_{evalType}_fold_{fold}.pth',
                'MLR_stocha': f'{dir}/MLR/initial_model/stochastic_outputs/initial_on_TrnOnRA_reconstruct_EvalFull_{evalType}_fold_{fold}.pth',
                'CNN': f'{dir}/CNN/initial_model/reconstruct/initial_on_TrnOnRA_reconstruct_EvalFull_{evalType}_fold_{fold}.pth', 
                'CNN_stocha': f'{dir}/CNN/initial_model/stochastic_outputs/initial_on_TrnOnRA_reconstruct_EvalFull_{evalType}_fold_{fold}.pth',
            }
            
            data_dict = get_data_dict(ref_ben_path, model_paths, domain='Europe')
            if n_iter>1:
                data_dict = process_bootstrap_data(data_dict,n_pdf=10)
            Pixel_score = get_data_scores(data_dict,n_iter=n_iter,funcs=[mean_bias_xskillscore,mean_crps_xskillscore,mean_mse_xskillscore,mean_ssr_xskillscore,])
            pkl.dump(Pixel_score,open(save_path,'wb'))

