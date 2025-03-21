import time
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import pytorch_lightning as pl
import torch
import random
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import KFold
import numpy as np
import os
import logging
import pickle as pkl
from itertools import chain
from downscaling.data.GetData import convertToBCWH
import copy
from tqdm import tqdm
import xarray as xr
from pytorch_lightning.callbacks.progress import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.tuner import Tuner
from src.scripts.processor import DataProcessor
from src.scripts.evaluation.evalute import to_Domain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
####################################################################################################
def seed_everything(seed: int):
    """
    Set random seeds for reproducibility across all random number generators.

    Args:
        seed (int): The seed value to use for all random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For CUDA devices
    torch.backends.cudnn.deterministic = True  # Force deterministic algorithms
    torch.backends.cudnn.benchmark = False  # Disable cuDNN auto-tuner

def object_mse(y_pred_invr, y_val_invr):
    """
    Calculate Mean Squared Error between prediction and ground truth.
    Takes mean across PDF dimension first, then computes squared error.

    Args:
        y_pred_invr: Predicted values (inverse transformed)
        y_val_invr: Ground truth values (inverse transformed)

    Returns:
        float: Mean Squared Error value
    """
    trial_mse = (y_pred_invr.mean('pdf') - y_val_invr.mean('pdf'))**2
    trial_mse = trial_mse.mean().data
    return trial_mse

def st_year_to_DJF(st_years, len_year):
    """
    Convert start years to winter season (Dec-Jan-Feb) date slices.

    Args:
        st_years: List of starting years
        len_year: Length in years to include in each slice

    Returns:
        list: List of slice objects with date ranges
    """
    return [slice(f'{year}-12-01', f'{year+len_year}-03-01') for year in st_years]

def get_stochasticity(da, stocha_size):
    """
    Expand the dataset with stochastic samples by adding a stochasticity dimension.

    Args:
        da: Input xarray DataArray
        stocha_size: Number of stochastic samples to generate

    Returns:
        xarray.DataArray: Expanded array with stochastic dimension
    """
    pdf_size = da.pdf.size
    da = da.assign_coords(pdf=range(pdf_size))
    da = da.rename({'pdf':'pdf_tmp'}).expand_dims(stochasticity=range(stocha_size))
    da = da.stack(pdf=['pdf_tmp','stochasticity'])
    da = da.assign_coords(pdf=range(pdf_size * stocha_size))
    return da

def get_N(da, slice_list):
    """
    Extract time indices (N) from a DataArray based on a list of time slices.

    Args:
        da: Input xarray DataArray
        slice_list: List of time slices to extract

    Returns:
        list: Combined list of all time indices
    """
    trn_N = [da.unstack().sel(N=ele).N.data for ele in slice_list]
    trn_N = list(chain(*trn_N))
    return trn_N

def dailyMean(args, X, data_type):
    """
    Process data to extract daily means and convert to batch, channel, width, height format.
    Optionally filters for specific hours of the day.

    Args:
        args: Configuration arguments
        X: Input xarray DataArray
        data_type: Type of data ('X' or 'y')

    Returns:
        xarray.DataArray: Processed data in BCWH format
    """
    X = X.unstack()
    if args.trainDailyMean and args.trn_type=='RA':
        X = X.sel(N=X.N.dt.hour.isin([0]))
    X = convertToBCWH(args, X, data_type)
    return X



class NoisyDataset(Dataset):
    """
    PyTorch Dataset that adds random noise to input features at each access.

    Attributes:
        X: Input features
        y: Target values
        noise_level: Standard deviation of noise to add
    """
    def __init__(self, X, y, noise_level):
        self.X = X
        self.y = y
        self.noise_level = noise_level

    def __len__(self):
        return len(self.X)

    def add_random_noise(self, data, noise_level):
        """
        Add Gaussian random noise to input data.
        
        Args:
            data: Input tensor
            noise_level: Standard deviation of noise to add
        
        Returns:
            torch.Tensor: Data with added noise
        """
        noise = torch.randn_like(data) * noise_level
        return data + noise

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        x_noisy = self.add_random_noise(x, self.noise_level)
        return x_noisy, y

class LitDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for handling train and validation data loaders.
    
    Attributes:
        batch_size: Batch size for data loaders
        train_dataset: Training dataset
        val_dataset: Validation dataset
    """
    def __init__(self, batch_size, train_dataset, val_dataset):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        del train_dataset, val_dataset

    def train_dataloader(self):
        """Returns the training data loader"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """Returns the validation data loader"""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

def get_dir(args, model_name):
    """
    Create and return directory paths for storing model data.
    Creates necessary subdirectories for models, parameters, and results.
    
    Args:
        args: Configuration arguments
        model_name: Name of the model
    
    Returns:
        str: Base directory path
    """
    dir = f'{args.base_dir}/{model_name}'
    os.makedirs(f'{dir}/initial_model/para_search/', exist_ok=1)
    os.makedirs(f'{dir}/initial_model/reconstruct/', exist_ok=1)
    os.makedirs(f'{dir}/initial_model/model/', exist_ok=1)
    os.makedirs(f'{dir}/initial_model/stochastic_param/', exist_ok=1)
    os.makedirs(f'{dir}/initial_model/stochastic_outputs/', exist_ok=1)
    return dir

def create_optuna_study(args, n_full_trials, study_name, storage, n_startup_trials, n_warmup_steps, interval_steps):
    """
    Create or load an Optuna hyperparameter optimization study.
    
    Args:
        args: Configuration arguments
        n_full_trials: Total number of trials to run
        study_name: Name of the study
        storage: Storage URL for the study database
        n_startup_trials: Number of trials before pruning starts
        n_warmup_steps: Number of steps before pruning starts in each trial
        interval_steps: Interval between pruning evaluations
    
    Returns:
        tuple: (study object, number of remaining trials)
    """
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=n_warmup_steps,
        interval_steps=interval_steps,
    )
    study = optuna.create_study(direction='minimize', 
                            sampler=optuna.samplers.TPESampler(seed=42), 
                            study_name=study_name,
                            storage=storage,
                            pruner=pruner, 
                            load_if_exists=True)
    trial_not_fail = [ele for ele in study.trials if ele.state not in [TrialState.FAIL, TrialState.RUNNING]]
    n_trials = n_full_trials - len(trial_not_fail)
    logger.info(f"Still have {n_trials} trials.")

    return study, n_trials


def pl_trainer(args, X_tnvl, y_tnvl, X_test, y_test, 
               max_batch_size, max_epochs, 
               best_model, model_filename, filename='best_model',
               trial=None):
    """
    Initialize and run a PyTorch Lightning trainer with the provided model and data.
    
    This function sets up PyTorch Lightning training with early stopping, checkpointing,
    and optional batch size auto-tuning.
    
    Args:
        args: Configuration arguments
        X_tnvl: Training input data
        y_tnvl: Training target data
        X_test: Validation input data
        y_test: Validation target data
        max_batch_size: Batch size for training, if None will be auto-tuned
        max_epochs: Maximum number of training epochs
        best_model: Model instance to train
        model_filename: Path to save the trained model
        filename: Base filename for model checkpoints
        trial: Optional Optuna trial object for hyperparameter tuning
        
    Returns:
        tuple: (trained model, batch size used, trainer instance)
    """
    # Display batch size information
    print(rf"""
          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                max_batch_size = {max_batch_size}
          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          """)
    
    # Validate model filename
    if model_filename is None:
        raise ValueError('model_filename is None')
    
    # Create directory for model if it doesn't exist
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    
    # Create datasets with and without noise
    train_dataset = NoisyDataset(torch.FloatTensor(X_tnvl.data), torch.FloatTensor(y_tnvl.data), noise_level=args.noise_level)
    test_dataset = TensorDataset(torch.FloatTensor(X_test.data), torch.FloatTensor(y_test.data))
    
    # Set up training callbacks
    callbacks = [
        RichProgressBar(),
        pl.callbacks.EarlyStopping(monitor="val_total_loss", mode="min", patience=25),
        ModelCheckpoint(
            dirpath=os.path.dirname(model_filename),
            monitor="val_total_loss",  # Metric to monitor
            mode="min",  # We want to minimize loss
            save_top_k=1,  # Save only the best model
            save_last=False,
            filename=filename,  # Filename format
            auto_insert_metric_name=False,  # Don't insert metric name automatically
        ),
    ]
    
    # Add pruning callback for Optuna if trial is provided
    if trial is not None:
        callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_total_loss"))
    
    # Create data module with appropriate batch size
    datamodule = LitDataModule(batch_size=32 if max_batch_size is None else max_batch_size,
                    train_dataset=train_dataset, val_dataset=test_dataset)

    # Configure and initialize the trainer
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=True,  # Ensure checkpointing is enabled
        default_root_dir=os.path.dirname(model_filename),
        # Auto batch size scaling is commented out (V3)
        # auto_scale_batch_size='power' if max_batch_size is None else None,
    )
    
    # If batch size is not specified, tune it automatically
    if max_batch_size is None:
        # use Lightning's tuner to find optimal batch size
        tuner = Tuner(trainer)
        datamodule.batch_size = tuner.scale_batch_size(
            best_model, 
            datamodule=datamodule,
            mode='binsearch'  # Binary search for optimal batch size
        )
        print(f"Tuned batch size: {datamodule.batch_size}")

    # Train the model
    trainer.fit(best_model, datamodule=datamodule)
    
    # Return the trained model, batch size used, and trainer instance
    return best_model, max_batch_size, trainer


class ModelTrainer:
    """
    Main class for training and evaluating machine learning models for wind speed prediction.
    Implements nested cross-validation, hyperparameter optimization, and model evaluation.

    Attributes:
        args: Configuration arguments
        model_class: PyTorch model class to train
        model_params: Default parameters for model initialization
        model_name: Name of the model class
        dir: Directory for saving model data
        debug: Debug mode flag
        device: Computing device (CUDA, MPS, or CPU)
        outer_splits: Number of outer CV folds
        inner_splits: Number of inner CV folds
        outer_len: Length in years for outer folds
        inner_len: Length in years for inner folds
        st_years_outer: Starting years for outer folds
        trn_DJFs_folds: Training data winter period folds
        max_batch_size: Maximum batch size for training
        best_lr: Best learning rate found during optimization
        best_wd: Best weight decay found during optimization
    """
    def __init__(self, args, model_class, model_params):
        """
        Initialize the ModelTrainer with configuration and model specifications.
        
        Args:
            args: Configuration arguments
            initial_mean_std: Initial normalization parameters
            fintune_mean_std: Fine-tuning normalization parameters
            model_class: PyTorch model class to train
            model_params: Default parameters for model initialization
        """
        self.args = args
        self.model_class = model_class
        self.model_params = model_params
        self.model_name = model_class.__name__
        self.dir = get_dir(args, model_class.__name__)
        self.debug = self.args.debug
        args_MLR = copy.deepcopy(args)
        
        # Set appropriate pretrained model directory based on model type
        if 'MLR' in self.model_name:
            self.pretrainDir = get_dir(args_MLR, 'MLR')
        else:
            self.pretrainDir = get_dir(args_MLR, 'CNN')

        # Select the best available device for computation
        self.device = torch.device('cuda' if torch.cuda.is_available() 
                            else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() 
                            else 'cpu')

        # Configure nested cross-validation parameters
        self.outer_splits = 3
        self.inner_splits = 6
        self.outer_len = 9
        self.inner_len = 3

        # Define starting years for outer folds (1994-2021 with step size outer_len)
        self.st_years_outer = np.array([year for year in range(1994, 2021, self.outer_len)])

        # Initialize cross-validation folds for winter periods
        self.trn_DJFs_folds = self.get_trn_DJFs()
        self.max_batch_size = None
        self.best_lr = None
        self.best_wd = None
   
    def get_trn_DJFs(self):
        """
        Generate training data folds for winter periods using nested cross-validation.
        
        Returns:
            list: Training winter period ranges for each fold
        """
        trn_DJFs = []
        kf = KFold(n_splits=self.outer_splits)
        
        # Iterate through outer folds
        for i_fold, (trn_val_idx, tst_idx) in enumerate(kf.split(self.st_years_outer)):
            # Get winter period (Dec-Jan-Feb) date ranges for outer fold
            trn_DJFs_outer = st_year_to_DJF(self.st_years_outer[trn_val_idx], self.outer_len)
            trn_DJFs.append(trn_DJFs_outer)
            
        #    # Generate child years for inner folds
        #    st_years_child = np.array([list(year + self.inner_len * np.arange(self.outer_len//self.inner_len)) 
        #                              for year in self.st_years_outer[trn_val_idx]]).flatten()

        #    # Create inner folds
        #    kf_child = KFold(n_splits=self.inner_splits)
        #    for i_fold_child, (trn_idx, val_idx) in enumerate(kf_child.split(st_years_child)):
        #        # Get training and validation dates for inner fold
        #        trn_DJFs_inner = st_year_to_DJF(st_years_child[trn_idx], self.inner_len)
        #        val_DJFs_inner = st_year_to_DJF(st_years_child[val_idx], self.inner_len)
        #        # The following lines are commented out in the original code
        #        # trn_N = get_N(da, trn_DJFs_inner) 
        #        # val_N = get_N(da, val_DJFs_inner) 
        #        pass 

        return trn_DJFs

    def save_model_and_params(self, model_state_dict, params, filename):
        """
        Save model state dictionary and parameters to file.
        
        Args:
            model_state_dict: Model state dictionary
            params: Model parameters
            filename: Path to save the model
        """
        torch.save({
            'model_state_dict': model_state_dict,
            'params': params
        }, filename)
        logger.info(f"Saved model and parameters to {filename}")

    def load_model_and_params(self, filename):
        """
        Load model state dictionary and parameters from file.
        
        Args:
            filename: Path to the saved model
            
        Returns:
            tuple: (model state dictionary, parameters) or (None, None) if file doesn't exist
        """
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=torch.device(self.device))
            params = checkpoint['params']
            params['args'] = self.args
            model = self.model_class(**params)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model and parameters from {filename}")
            return model.state_dict(), params
        return None, None

    def objective(self, trial, X_tnvl, y_tnvl, X_normalize, y_normalize, st_years_inner, 
                is_fine_tuning=False, initial_model_state_dict=None, model_filename=None,
                pretrained_weights=None, full_data=False, top_params=None):
        """
        Objective function for hyperparameter optimization using Optuna.
        
        Args:
            trial: Optuna trial object
            X_tnvl: Training input data
            y_tnvl: Training target data
            X_normalize: Input data normalizer
            y_normalize: Target data normalizer
            st_years_inner: Starting years for inner folds
            is_fine_tuning: Whether performing fine-tuning
            initial_model_state_dict: Initial model state for fine-tuning
            model_filename: Path to save the model
            pretrained_weights: Path to pretrained weights
            full_data: Whether to use full dataset
            top_params: List of top parameter sets from previous optimization
            
        Returns:
            float: Mean MSE across validation folds
        """
        inner_kf = KFold(n_splits=self.inner_splits, shuffle=False)

        # Either use parameters from top trials or sample new ones
        if full_data:
            param_index = trial.number % len(top_params)
            search_params = top_params[param_index]
            seed_everything(search_params['seed'])
        else:
            # Sample random seed and hyperparameters
            seed = trial.suggest_int("seed", 1, 1000)
            seed_everything(seed)
            search_params = {
                'lr': trial.suggest_loguniform('lr', 1e-6, 1e-1),
                'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-1),
            }
        
        epochs_object = self.args.epochs
        params = {**search_params, **self.model_params}
        params['pretrained_weights'] = pretrained_weights
        if 'seed' in params.keys():
            del params['seed']
        val_scores = {}
        val_scores['MSE'] = []

        for inner_fold, (train_index, val_index) in enumerate(inner_kf.split(st_years_inner)):
            if inner_fold != 0: continue
            
            # Split data into train and validation sets
            X_train = self.sub_data('tnvl', X_tnvl, None, st_years_inner[train_index], self.inner_len, normalization=False)
            y_train = self.sub_data('tnvl', y_tnvl, None, st_years_inner[train_index], self.inner_len, normalization=False)
            X_val = self.sub_data('test', X_tnvl, None, st_years_inner[val_index], self.inner_len, normalization=False)
            y_val = self.sub_data('test', y_tnvl, None, st_years_inner[val_index], self.inner_len, normalization=False)
            
            # Subsample data based on whether using full dataset
            if full_data:
                sub_size = 1
            else:
                sub_size = 2
                
            # Sample training data at regular intervals
            i_subsample = [int(ele) for ele in np.linspace(0, X_train.sample.size-1, X_train.sample.size//sub_size)]
            X_train = X_train.isel(sample=i_subsample)
            y_train = y_train.isel(sample=i_subsample)
            
            # Sample validation data at regular intervals
            i_subsample = [int(ele) for ele in np.linspace(0, X_val.sample.size-1, X_val.sample.size//sub_size)]
            X_val = X_val.isel(sample=i_subsample)
            y_val = y_val.isel(sample=i_subsample)

            # Initialize model with selected parameters
            best_model = self.model_class(**params)
            if is_fine_tuning and initial_model_state_dict is not None:
                best_model.load_state_dict(initial_model_state_dict)

            # Train the model
            best_model, self.max_batch_size, trainer = pl_trainer(
                self.args, X_train, y_train, X_val, y_val, 
                self.max_batch_size, epochs_object, 
                best_model, model_filename, filename='best_model_object',
                trial=trial,
            )

            # Store best parameters
            if full_data:
                best_params = {**search_params, **self.model_params}
                trial.set_user_attr("original_trial_param", search_params)
            else:
                best_params = {**trial.params, **self.model_params}

            if 'seed' in best_params.keys():
                del best_params['seed']

            # Evaluate model on validation set
            X_val_norm = self.perturbed_data(X_val)
            y_val_norm = self.perturbed_data(y_val)
            X_val_norm = convertToBCWH(self.args, X_val_norm)
            y_val_norm = convertToBCWH(self.args, y_val_norm)
            y_val_invr = y_normalize.inverse(y_val, 'tnvl')
            y_val_invr = convertToBCWH(self.args, y_val_invr)

            # Get model predictions and calculate MSE
            y_pred_invr, _ = self.get_prediction('tnvl', best_model.state_dict(), best_params, X_val_norm, y_val_norm, y_normalize)
            y_pred_invr = to_Domain(y_pred_invr.unstack(), 'Europe')
            y_val_invr = to_Domain(y_val_invr.unstack(), 'Europe')
            mse_fold = object_mse(y_pred_invr, y_val_invr)
            val_scores['MSE'].append(mse_fold)
            print("mse_fold", mse_fold)


        mse_avrg = np.mean(val_scores['MSE'])
        print("mse_avrg", mse_avrg)
        # Set trial attributes and print results
        trial.set_user_attr("mse", float(mse_avrg))
        return mse_avrg

    def sub_data(self, index_name, full_data, normalize_func, sub_st_years, fold_len, normalization=True):
        """
        Extract a subset of data for a specific time period.

        Args:
            index_name: Name of the subset ('tnvl' or 'test')
            full_data: Full dataset
            normalize_func: Normalization function
            sub_st_years: Starting years for the subset
            fold_len: Length of each fold in years
            normalization: Whether to normalize the data
            
        Returns:
            xarray.DataArray: Subset of data
        """
        # print(f'before sub_data: {full_data.unstack().dims}, {full_data.unstack().shape}')

        # Get date ranges and time indices
        DJFs_range = st_year_to_DJF(sub_st_years, fold_len)
        N = get_N(full_data, DJFs_range) 
        if len(N) == 0:
            return None
            
        # Extract and reshape data
        sub_data = convertToBCWH(self.args, full_data.unstack().sel(N=N))
        if normalization:
            sub_data = normalize_func.transform(sub_data, index_name)
        # print(f'After sub_data: {sub_data.unstack().dims}, {sub_data.unstack().shape}')

        return sub_data

    def perturbed_data(self, da, judge=True):
        """
        Add stochastic perturbations to data if needed for certain model types.

        Args:
            da: Input data array
            judge: Whether to check model type before adding perturbation
            
        Returns:
            xarray.DataArray: Perturbed data
        """
        # Add stochastic perturbations based on model type
        if judge:
            pass 
        else:
            stocha_size = self.args.stocha_size if not self.args.debug else 4
            da = get_stochasticity(da.unstack(), stocha_size)
            da = convertToBCWH(self.args, da)
        return da 
   
    def ypred_to_xr(self, y_pred, sub_y_test, prediction=True):
        """
        Convert numpy prediction array to xarray format with appropriate dimensions.

        Args:
            y_pred: Prediction array
            sub_y_test: Reference test data with correct dimensions
            prediction: Whether this is a prediction (vs. ground truth)
            
        Returns:
            xarray.DataArray: Formatted predictions
        """
        Y_pred = xr.zeros_like(sub_y_test)
        # Assign prediction data to xarray structure
        Y_pred.data = y_pred
        return Y_pred
       
    def stochastic_param(self, Y_trn_res, param_filename):
        """
        Calculate stochastic perturbation parameters from residuals.

        Args:
            Y_trn_res: Residuals (difference between predictions and ground truth)
            param_filename: Path to save the parameters
            
        Returns:
            tuple: (mean, variance) of residuals
        """
        # Extract feature and sample dimensions
        feature_dims = [ele for ele in Y_trn_res.unstack('channel').dims if ele not in ['sample']]
        sample_dims = [ele for ele in Y_trn_res.unstack().dims if ele not in feature_dims]

        # Reshape and calculate statistics
        Y_trn_res = Y_trn_res.unstack().stack(sample=sample_dims).stack(feature=feature_dims).transpose('sample', 'feature')

        # Calculate mean of residuals
        mean_trn_res = Y_trn_res.mean('sample')

        # Calculate variance of residuals
        vari_trn_res = Y_trn_res.var('sample')

        # Save parameters to file
        pkl.dump([mean_trn_res, vari_trn_res], open(param_filename, 'wb'))
        return mean_trn_res, vari_trn_res

    def stochastic_outputs(self, Y_pred, mean_trn_res, vari_trn_res):
        """
        Add stochastic perturbations to model outputs using saved parameters.

        Args:
            Y_pred: Model predictions
            mean_trn_res: Mean of residuals
            vari_trn_res: Variance of residuals
            
        Returns:
            xarray.DataArray: Perturbed predictions
        """
        # Extract feature and sample dimensions
        feature_dims = mean_trn_res.unstack().dims
        sample_dims = [ele for ele in Y_pred.unstack().dims if ele not in feature_dims]

        # Reshape data for perturbation
        Y_pred = self.perturbed_data(Y_pred, judge=False).unstack()
        Y_pred = Y_pred.stack(sample=sample_dims).stack(feature=feature_dims).transpose('sample', 'feature')

        print('feature_dims', feature_dims)
        print('mean_trn_res', mean_trn_res.shape)
        print('vari_trn_res', vari_trn_res.shape)
        print('Y_pred', Y_pred.unstack().dims, Y_pred.unstack().shape)
        print('Y_pred', Y_pred.dims, Y_pred.shape)

        st0 = time.time()

        # Generate noise in batches to save memory
        batch_size = 1000
        if Y_pred.shape[0] > batch_size:
            num_batches = Y_pred.shape[0] // batch_size
            val_noise = np.zeros_like(Y_pred)

            # Process data in batches
            for i in tqdm(range(num_batches), leave=False, desc='Stochasting...'):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                val_noise[start_idx:end_idx] = np.random.normal(
                    mean_trn_res, 
                    np.sqrt(vari_trn_res), 
                    (batch_size, Y_pred.shape[1])
                )

            # Handle remaining samples
            if Y_pred.shape[0] % batch_size != 0:
                start_idx = num_batches * batch_size
                val_noise[start_idx:] = np.random.normal(
                    mean_trn_res,
                    np.sqrt(vari_trn_res),
                    (Y_pred.shape[0] - start_idx, Y_pred.shape[1])
                )
        else:
            # Generate noise for all samples at once if batch size is sufficient
            val_noise = np.random.normal(
                mean_trn_res.astype(np.float32), 
                np.sqrt(vari_trn_res).astype(np.float32), 
                Y_pred.shape
            ).astype(np.float32)
            
        st1 = time.time()

        # Add noise to predictions
        Y_pred = Y_pred + val_noise
        st2 = time.time()

        # Reshape back to original format
        Y_pred = Y_pred.unstack()
        st3 = time.time()

        # Print timing information
        print(f'st1-st0:{st1-st0:.2f}s')
        print(f'st2-st1:{st2-st1:.2f}s')
        print(f'st3-st2:{st3-st2:.2f}s')
        print('Y_pred', Y_pred.dims, Y_pred.shape)

        return Y_pred
    
    def train_initial_model(self, X_initial, y_initial, X_initial_class, y_initial_class, initial_name):
        """
        Train the initial model using nested cross-validation and hyperparameter optimization.

        Args:
            X_initial: Initial input data
            y_initial: Initial target data
            X_initial_class: Class for input data
            y_initial_class: Class for target data
            initial_name: Name of the initial model
            
        Returns:
            tuple: (models, stochastic parameters, normalizers) for each fold
        """
        outer_kf = KFold(n_splits=self.outer_splits, shuffle=False)
        models = {}
        stocha_params = {}
        Xynormalizes = {}
        n_trials = 100

        # Set up storage directory
        if self.args.trainDailyMean and self.args.trn_type=='RA':
            storage_dir = self.dir.replace('_trainDailyMean_0', '_trainDailyMean_1')
            os.makedirs(f'{storage_dir}/initial_model/para_search', exist_ok=1)
        else:
            storage_dir = self.dir

        # Reduce number of trials in debug mode
        n_full_trials = n_trials if not self.debug else 3

        # Get normalization parameters for full dataset
        X_normalize_full, y_normalize_full, normalization = self.get_full_statistical_mean_std(
            X_initial, y_initial, X_initial_class, y_initial_class
        )

        # Iterate through outer folds
        for outer_fold, (tnvl_index, test_index) in enumerate(outer_kf.split(self.st_years_outer)):
            # Skip folds not requested
            if outer_fold != self.args.outer_fold and not self.args.test_Full: 
                continue
                
            # Set up file paths
            model_filename = f'{self.dir}/initial_model/model/initial_model_on_{initial_name}_outer_fold_{outer_fold}.pth'
            pretrained_weights = f'{self.pretrainDir}/initial_model/model/initial_model_on_{initial_name}_outer_fold_{outer_fold}.pth'
            stochastic_param_filename = f'{self.dir}/initial_model/stochastic_param/initial_model_on_{initial_name}_outer_fold_{outer_fold}.pth'

            # Check for existing model
            existing_model_state_dict, existing_params = self.load_model_and_params(model_filename)

            # Prepare and normalize data for this fold
            X_tnvl, X_test, X_normalize_fold = self.get_sub_normalized_data(
                'X', X_initial, X_initial_class, X_normalize_full, normalization, tnvl_index, test_index
            )
            y_tnvl, y_test, y_normalize_fold = self.get_sub_normalized_data(
                'y', y_initial, y_initial_class, y_normalize_full, normalization, tnvl_index, test_index
            )
            
            # Print data dimensions for debugging
            print(f'=================train_initial_model fold {outer_fold}=============================')
            print(f'X_tnvl: {X_tnvl.dims}, {X_tnvl.shape}, {X_tnvl.unstack().dims}, {X_tnvl.unstack().shape}')
            print(f'y_tnvl: {y_tnvl.dims}, {y_tnvl.shape}, {y_tnvl.unstack().dims}, {y_tnvl.unstack().shape}')
            print(f'X_test: {X_test.dims}, {X_test.shape}, {X_test.unstack().dims}, {X_test.unstack().shape}')
            print(f'y_test: {y_test.dims}, {y_test.shape}, {y_test.unstack().dims}, {y_test.unstack().shape}')

            # Create default stochastic parameters for specific model types
            if not ('Diffusion' in self.model_name or 'Stocha' in self.model_name or 'VAE' in self.model_name or 'Quantile' in self.model_name):
                pass
            else:
                mean_trn_res, vari_trn_res = 0, 1
                pkl.dump([mean_trn_res, vari_trn_res], open(stochastic_param_filename, 'wb'))

            # Use existing model if available and not in optimization or debug mode
            if (existing_model_state_dict is not None and 
                os.path.exists(stochastic_param_filename) and 
                not self.args.useOptuna and 
                not self.args.debug):
                    
                logger.info(f"Found existing initial model for fold {outer_fold}. Skipping parameter search.")
                models[outer_fold] = (existing_model_state_dict, existing_params)
                self.best_lr = existing_params['lr']
                self.best_wd = existing_params['weight_decay']
                mean_trn_res, vari_trn_res = pkl.load(open(stochastic_param_filename, 'rb'))
                stocha_params[outer_fold] = (mean_trn_res, vari_trn_res)
                Xynormalizes[outer_fold] = (X_normalize_fold, y_normalize_fold)
                continue

            # Prepare inner fold years for hyperparameter optimization
            st_years_inner = np.array([
                list(year + self.inner_len * np.arange(self.outer_len//self.inner_len)) 
                for year in self.st_years_outer[tnvl_index]
            ]).flatten()
            
            # Create Optuna study for hyperparameter optimization
            if self.args.useOptuna: 
                study, n_trials = create_optuna_study(
                    self.args,
                    n_full_trials=n_full_trials,
                    study_name=f'outer_fold', 
                    storage=f'sqlite:///{storage_dir}/initial_model/para_search/on_{initial_name}_fold{outer_fold}_example.db', 
                    n_startup_trials=10,
                    n_warmup_steps=80,
                    interval_steps=40,
                )


            # Check again for existing model (with additional conditions)
            if existing_model_state_dict is not None and os.path.exists(stochastic_param_filename) and (n_trials<=0 or not self.args.useOptuna) and not self.args.debug:
                logger.info(f"Found existing initial model for fold {outer_fold}. Skipping parameter search.")
                models[outer_fold] = (existing_model_state_dict, existing_params)
                self.best_lr = existing_params['lr']
                self.best_wd = existing_params['weight_decay']
                mean_trn_res, vari_trn_res = pkl.load(open(stochastic_param_filename,'rb'))
                stocha_params[outer_fold] = (mean_trn_res, vari_trn_res)
                Xynormalizes[outer_fold] = (X_normalize_fold, y_normalize_fold)
                continue
            else:
                # Print debug information about model loading conditions
                print('===============================================================')
                print(f'existing_model_state_dict is not None: {existing_model_state_dict is not None}')
                print(f'os.path.exists(stochastic_param_filename): {os.path.exists(stochastic_param_filename)}')
                print(f"'(n_trials<=0 or not self.args.useOptuna): {(n_trials<=0 or not self.args.useOptuna)}")
                print('===============================================================')

            # Perform hyperparameter optimization with Optuna if requested
            if self.args.useOptuna:
                if n_trials>0:
                    # Run optimization for stage 1
                    study.optimize(lambda trial: self.objective(
                        trial, X_tnvl, y_tnvl, X_normalize_fold, y_normalize_fold, st_years_inner,
                        model_filename=model_filename, pretrained_weights=pretrained_weights), 
                        n_trials=n_trials,
                        callbacks=[MaxTrialsCallback(n_full_trials, states=(TrialState.COMPLETE, TrialState.PRUNED))]
                    )
                
                # Create a second optimization stage with top parameters from stage 1
                study_stage2, n_trials_stage2 = create_optuna_study(
                    self.args,
                    n_full_trials=10,
                    study_name=f'outer_fold', 
                    storage=f'sqlite:///{storage_dir}/initial_model/para_search/on_{initial_name}_fold{outer_fold}_example_stage2.db', 
                    n_startup_trials=10,
                    n_warmup_steps=80,
                    interval_steps=40,
                )
                
                # Get top trials from stage 1
                trial_COMPLETE = [ele for ele in study.trials if ele.state in [TrialState.COMPLETE]]
                top_trials = sorted(trial_COMPLETE, key=lambda t: t.value if t.value is not None else float('inf'))[:10]
                top_params = [t.params for t in top_trials if t.params]  # Ensure parameters are not empty
                
                if n_trials_stage2>0:
                    # Run optimization for stage 2 using top parameters from stage 1
                    study_stage2.optimize(lambda trial: self.objective(
                        trial, X_tnvl, y_tnvl, X_normalize_fold, y_normalize_fold, st_years_inner,
                        model_filename=model_filename, pretrained_weights=pretrained_weights,
                        full_data=True, top_params=top_params), 
                        n_trials=n_trials_stage2,
                        callbacks=[MaxTrialsCallback(n_full_trials, states=(TrialState.COMPLETE, TrialState.PRUNED))]
                    )

                # Extract best parameters from stage 2
                best_params = study_stage2.best_trial.user_attrs['original_trial_param']
                self.best_lr = best_params['lr']
                self.best_wd = best_params['weight_decay']
                print(f'best_values:')
                print(study.best_value)
            else:
                # Use default hyperparameters if not using Optuna
                self.max_batch_size = None

                # Set default learning rate and weight decay based on model type
                if 'MLR' in self.model_name:
                    self.best_lr = 1e-2
                    self.best_wd = 1e-5
                else:
                    self.best_lr = 1e-4
                    self.best_wd = 1e-5

                best_params = {}
                best_params['lr'] = self.best_lr
                best_params['weight_decay'] = self.best_wd

            # Combine best hyperparameters with model-specific parameters
            best_params = {**best_params, **self.model_params}
            best_params['pretrained_weights'] = pretrained_weights
            if 'seed' in best_params.keys():
                del best_params['seed']
                
            # Initialize model with best parameters
            best_model = self.model_class(**best_params)
            
            # Train model with best parameters
            best_model, self.max_batch_size, trainer = pl_trainer(
                self.args, X_tnvl, y_tnvl, X_test, y_test, 
                self.max_batch_size, self.args.epochs, 
                best_model, model_filename, 
            )

            # Load best checkpoint
            best_model_path = trainer.checkpoint_callback.best_model_path
            best_model = best_model.load_from_checkpoint(best_model_path, **best_params)

            # Save model state and parameters
            self.save_model_and_params(best_model.state_dict(), best_params, model_filename)
            best_model_loss = trainer.checkpoint_callback.best_model_score.cpu().numpy()
            pkl.dump(best_model_loss, open(f'{model_filename}_best_loss', 'wb'))

            # Calculate stochastic perturbation parameters based on model type
            if not ('Diffusion' in self.model_name or 'Stocha' in self.model_name or 'VAE' in self.model_name or 'Quantile' in self.model_name):
                # For regular models, calculate residuals between predictions and ground truth
                y_pred_invr, _ = self.get_prediction('tnvl', best_model.state_dict(), best_params, X_tnvl, y_tnvl, y_normalize_fold)
                y_truth = self.sub_data('tnvl', y_initial, None, self.st_years_outer[tnvl_index], self.outer_len, normalization=False)
                mean_trn_res, vari_trn_res = self.stochastic_param(Y_trn_res=y_truth-y_pred_invr, param_filename=stochastic_param_filename)
                del y_pred_invr
            else:
                # For stochastic models, use default values
                mean_trn_res, vari_trn_res = 0, 1
                pkl.dump([mean_trn_res, vari_trn_res], open(stochastic_param_filename, 'wb'))

            # Store model, parameters, and normalizers for this fold
            models[outer_fold] = (best_model.state_dict(), best_params)
            stocha_params[outer_fold] = (mean_trn_res, vari_trn_res)
            Xynormalizes[outer_fold] = (X_normalize_fold, y_normalize_fold)
            
        return models, stocha_params, Xynormalizes

    def evaluate_model(self, model_state_dict, params, X_test, y_test):
        """
        Evaluate a trained model on test data.
        
        Args:
            model_state_dict: Model state dictionary
            params: Model parameters
            X_test: Test input data
            y_test: Test target data
            
        Returns:
            numpy.ndarray: Model predictions on test data
        """
        # Add args to parameters and initialize model
        params['args'] = self.args
        model = self.model_class(**params)
        disable = False  # Flag for disabling progress bar

        # Load model state and set to evaluation mode
        model.load_state_dict(model_state_dict)
        model.eval()
        
        # Create test dataset and dataloader
        test_dataset = TensorDataset(torch.FloatTensor(X_test.data), torch.FloatTensor(y_test.data))
        if self.max_batch_size is None:
            self.max_batch_size = 256 * 2 * 3
        test_loader = DataLoader(test_dataset, batch_size=self.max_batch_size)
        
        # Move model to appropriate device
        model.to(device=self.device)
        model.eval()

        # Run inference on test data
        y_pred = []
        mses = []
        with torch.no_grad():
            for batch in tqdm(test_loader, disable=disable):
                x, y = batch
                _, preds = model(x.to(device=self.device))
                y = y.to(device=self.device)
                
                # Store predictions and calculate batch MSE
                y_pred.extend(preds.cpu().numpy())
                mse = ((preds.cpu().numpy()-y.cpu().numpy())**2).mean()
                mses.append(mse)
                del preds  # Free memory

        # Convert predictions to numpy array and calculate overall MSE
        y_pred = np.array(y_pred)
        mses = np.mean(mses)
        print(f'MSE: {mses:.4f}')
        return y_pred
    
    def get_full_statistical_mean_std(self, X_initial, y_initial, X_initial_class, y_initial_class):
        """
        Calculate normalization parameters for the full dataset.
        
        Args:
            X_initial: Input data
            y_initial: Target data
            X_initial_class: Class for input data
            y_initial_class: Class for target data
            
        Returns:
            tuple: (X normalizer, y normalizer, normalization flag)
        """
        # Initialize data processors
        X_normalize = DataProcessor(self.args, self.model_name)
        y_normalize = DataProcessor(self.args, self.model_name)
        
        # Configure normalization based on processor type
        if self.args.fcProcessor in ['RA']:
            # Use training set statistics for normalization
            X_normalize.fit(X_initial, X_initial)
            y_normalize.fit(y_initial, y_initial)
            normalization = True
        elif self.args.fcProcessor in ['CL']:
            # Use climatology as reference for normalization
            X_normalize.fit(X_initial, X_initial_class.bench)
            y_normalize.fit(y_initial, y_initial_class.bench)
            normalization = True
        else:
            # No normalization
            X_normalize = None
            y_normalize = None
            normalization = False
            
        return X_normalize, y_normalize, normalization

    def get_sub_normalized_data(self, data_type, initial_data, initial_class, 
                                normalize_func, normalization,
                                tnvl_index, test_index, initialnormalize_func=True):
        """
        Extract and normalize a subset of data for training or testing.
        
        Args:
            data_type: Type of data ('X' or 'y')
            initial_data: Full dataset
            initial_class: Class for the data
            normalize_func: Normalization function
            normalization: Whether to normalize
            tnvl_index: Indices for training/validation data
            test_index: Indices for test data
            initialnormalize_func: Whether to initialize a new normalizer
            
        Returns:
            list: [normalized training data, normalized test data, normalizer]
        """
        # Extract training/validation data
        tnvl_data = self.sub_data('tnvl', initial_data, None, self.st_years_outer[tnvl_index], self.outer_len, normalization=False)
        if tnvl_data is not None:
            # Get benchmark data and convert formats
            tnvl_bench_data = self.sub_data('tnvl', initial_class.bench, None, self.st_years_outer[tnvl_index], self.outer_len, normalization=False)
            tnvl_data = dailyMean(self.args, tnvl_data, data_type)
            tnvl_bench_data = dailyMean(self.args, tnvl_bench_data, data_type)
            
            # Configure normalizer based on processor type
            if self.args.fcProcessor in ['subRA'] and initialnormalize_func:
                # Create new normalizer using subset statistics
                normalize_func = DataProcessor(self.args, self.model_name)
                normalize_func.fit(tnvl_data, tnvl_data)
            elif self.args.fcProcessor in ['subCL'] and initialnormalize_func:
                # Create new normalizer using subset climatology
                normalize_func = DataProcessor(self.args, self.model_name)
                normalize_func.fit(tnvl_data, tnvl_bench_data)
            elif self.args.fcProcessor in ['RA', 'CL']:
                # Use existing normalizer
                pass
            elif not initialnormalize_func:
                # Skip normalization initialization
                pass
            else:
                raise ValueError(f'Error in {self.args.fcProcessor}')
    
            # Apply normalization
            tnvl_data = normalize_func.transform(tnvl_data, tnvl_bench_data, 'tnvl')

        # Extract test data
        test_data = self.sub_data('test', initial_data, None, self.st_years_outer[test_index], self.outer_len, normalization=False)
        if test_data is not None:
            # Get benchmark data and convert formats
            test_bench_data = self.sub_data('test', initial_class.bench, None, self.st_years_outer[test_index], self.outer_len, normalization=False)
            test_data = dailyMean(self.args, test_data, data_type)
            test_bench_data = dailyMean(self.args, test_bench_data, data_type)
            
            # Apply normalization
            test_data = normalize_func.transform(test_data, test_bench_data, 'test')

        return [tnvl_data, test_data, normalize_func]

    def get_prediction(self, index_name, model_state_dict, params, sub_X, sub_y, y_normalize, p_filename_sub_y_norm=None):
        """
        Generate model predictions on a dataset.
        
        Args:
            index_name: Name of the data subset ('tnvl' or 'test')
            model_state_dict: Model state dictionary
            params: Model parameters
            sub_X: Input data
            sub_y: Target data
            y_normalize: Target data normalizer
            p_filename_sub_y_norm: Optional filename to save normalized predictions
            
        Returns:
            tuple: (inverse transformed predictions, normalized predictions)
        """
        # Run model evaluation to get predictions
        y_pred_norm = self.evaluate_model(model_state_dict, params, sub_X, sub_y)
        
        # Convert predictions to xarray format
        y_pred_norm = self.ypred_to_xr(y_pred_norm, sub_y)
        
        # Inverse transform predictions to original scale
        y_pred_invr = y_normalize.inverse(y_pred_norm, index_name)
        
        return y_pred_invr, y_pred_norm
    
    def align_N(self, X, Y, use_dailyMean=True):
        """
        Align input and target data by time indices and apply daily mean if requested.
        
        Args:
            X: Input data
            Y: Target data
            use_dailyMean: Whether to extract daily means
            
        Returns:
            tuple: (aligned input data, aligned target data)
        """
        # Find common time indices between X and Y
        N_tmp = list(np.intersect1d(X.unstack().N.data, Y.unstack().N.data))
        
        # Select common indices and convert format
        X = convertToBCWH(self.args, X.unstack().sel(N=N_tmp), 'X')
        Y = convertToBCWH(self.args, Y.unstack().sel(N=N_tmp), 'y')
        
        # Apply daily mean processing if requested
        if use_dailyMean:
            X = dailyMean(self.args, X, 'X')
            Y = dailyMean(self.args, Y, 'y')
            
        return X, Y

    def run_initial_train_experiment(self, X_initial, y_initial, 
                                     X_initial_class, y_initial_class,
                                     initial_name, test_datasets):
        """
        Run complete experiment: train initial model and evaluate on test datasets.
        
        Args:
            X_initial: Initial input data
            y_initial: Initial target data
            X_initial_class: Class for input data
            y_initial_class: Class for target data
            initial_name: Name of the initial model
            test_datasets: List of test datasets to evaluate on
            
        This method trains the model and evaluates it on multiple test datasets.
        """
        # Define output path
        p_filename = f'{self.dir}/initial_model/reconstruct/initial_on_TrnOnRA_reconstruct_Eval_in_test_fold_0.pth'

        # Align input and target data
        X_initial, y_initial = self.align_N(X_initial, y_initial)

        # Train initial model
        initial_models, stocha_params, Xynormalizes = self.train_initial_model(
            X_initial, y_initial, X_initial_class, y_initial_class, initial_name
        )
        print(f"Initial training completed on {initial_name}.")

        # Evaluate model on each test dataset
        for i, (X_test, y_test, X_test_class, y_test_class, data_name) in enumerate(test_datasets):
            p_filename = f'{self.dir}/initial_model/reconstruct/initial_on_{initial_name}_reconstruct_{data_name}_test_fold_0.pth'

            # Align test data
            X_test, y_test = self.align_N(X_test, y_test, use_dailyMean=self.args.trainDailyMean)
            
            # Create cross-validation folds
            outer_kf = KFold(n_splits=self.outer_splits, shuffle=False)

            # Evaluate on each fold
            for outer_fold, (tnvl_index, test_index) in enumerate(outer_kf.split(self.st_years_outer)):
                # Get model, parameters, and normalizers for this fold
                model_state_dict, params = initial_models[outer_fold]
                mean_trn_res, vari_trn_res = stocha_params[outer_fold]
                X_normalize_fold, y_normalize_fold = Xynormalizes[outer_fold]
                
                # Normalize test data
                sub_X_norm_tnvl, sub_X_norm_test, _ = self.get_sub_normalized_data(
                    'X', X_test, X_test_class, X_normalize_fold, True, tnvl_index, test_index, initialnormalize_func=False
                )
                sub_y_norm_tnvl, sub_y_norm_test, _ = self.get_sub_normalized_data(
                    'y', y_test, y_test_class, y_normalize_fold, True, tnvl_index, test_index, initialnormalize_func=False
                )
                
                # Process and evaluate on training and test subsets
                for (sub_X_norm, sub_y_norm), index, index_name in zip(
                    [(sub_X_norm_tnvl, sub_y_norm_tnvl), (sub_X_norm_test, sub_y_norm_test)],
                    [tnvl_index, test_index],
                    ['tnvl', 'test']
                ):
                    # Skip if no data available
                    if sub_X_norm is None: 
                        continue
                        
                    # Define output paths
                    p_filename = f'{self.dir}/initial_model/reconstruct/initial_on_{initial_name}_reconstruct_{data_name}_{index_name}_fold_{outer_fold}.pth'
                    p_filename_stocha = f'{self.dir}/initial_model/stochastic_outputs/initial_on_{initial_name}_reconstruct_{data_name}_{index_name}_fold_{outer_fold}.pth'
                    
                    # Skip if predictions already exist
                    if os.path.exists(p_filename) and os.path.exists(p_filename_stocha): 
                        continue

                    # Add stochastic perturbations to input and target data
                    sub_X_norm = self.perturbed_data(sub_X_norm)
                    sub_y_norm = self.perturbed_data(sub_y_norm)

                    # Convert to BCWH format
                    sub_X_norm = convertToBCWH(self.args, sub_X_norm)
                    sub_y_norm = convertToBCWH(self.args, sub_y_norm)

                    # Inverse transform target data
                    sub_y_invr = y_normalize_fold.inverse(sub_y_norm, index_name)
                    sub_y_invr = convertToBCWH(self.args, sub_y_invr)
                    
                    # Generate model predictions
                    sub_p_invr, sub_p_norm = self.get_prediction(
                        index_name, model_state_dict, params, sub_X_norm, sub_y_norm, y_normalize_fold, p_filename
                    )
                    
                    # Save predictions
                    pkl.dump(sub_p_invr, open(p_filename, 'wb'))

                    # Add stochastic perturbations to test predictions
                    if index_name == 'test':
                        p_filename = f'{self.dir}/initial_model/stochastic_outputs/initial_on_{initial_name}_reconstruct_{data_name}_{index_name}_fold_{outer_fold}.pth'
                        sub_p_invr = self.stochastic_outputs(sub_p_invr, mean_trn_res, vari_trn_res)
                        pkl.dump(sub_p_invr.unstack(), open(p_filename, 'wb'))
                        del sub_p_invr
                        
                    print(f"Predictions saved for {data_name} - {index_name}, fold {outer_fold}")