import numpy as np
import pickle as pkl
import os
from scripts.trainer import ModelTrainer
from downscaling.data.GetData import get_full_RA, get_full_FC, get_full_HC, convertToBCWH
from downscaling.models.mlr import MLR, get_MLR_params
from downscaling.models.cnn import CNN
import copy 

def get_ten_samples(sample_size):
    # Generate 100 equally spaced numbers from 0 to sample_size
    numbers = np.linspace(0, sample_size, 101)[:100]
    # Convert results to integer list
    return [int(x) for x in numbers]

def update_easy_args(args, pipeline=False):
    # Set parameters for debug mode
    if args.debug:
        args.epochs = 2
        args.model_Size = 2
        args.model_level = 1
    else:
        args.epochs = 600

    # Define model parameters based on input predictors and predictands
    args.n_channels = len(args.predors.split('_'))
    args.n_classes_dims = [len(args.prednds.split('_'))]
    
    # Set directory paths based on debug mode and Optuna usage
    if args.debug:
        args.base_dir = f'{args.base_dir}/debug'
    if not args.useOptuna:
        args.base_dir = f'{args.base_dir}/NoOptuna'
    else:
        args.base_dir = f'{args.base_dir}/WithOptuna'

    return args

def update_model_args(args, model, pipeline=False):
    # Calculate total number of output classes
    args.n_classes = np.prod(args.n_classes_dims)
    return args
    
def get_data(args, trn_type, predictors, predictand):
    # Load reanalysis data (RA) for training
    X_initial_class = get_full_RA(args, predictors)
    y_initial_class = get_full_RA(args, predictand)
    
    # Load forecast data (FC) for evaluation
    X_fc_class = get_full_FC(args, predictors)
    y_fc_class = get_full_FC(args, predictand)
    
    # Load hindcast data (HC) for evaluation
    X_hc_class = get_full_HC(args, predictors)
    y_hc_class = get_full_HC(args, predictand)
    
    # Save reference datasets if they don't exist
    if not os.path.exists(f'{args.base_dir}/y_initial_class'):
        os.makedirs(f'{args.base_dir}/', exist_ok=1)
        pkl.dump(y_initial_class, open(f'{args.base_dir}/y_initial_class', 'wb'))
        pkl.dump(y_fc_class, open(f'{args.base_dir}/y_fc_class', 'wb'))
        pkl.dump(y_hc_class, open(f'{args.base_dir}/y_hc_class', 'wb'))

    # Convert data to BCWH format (Batch, Channel, Width, Height)
    X_initial = convertToBCWH(args, X_initial_class.dynam, 'X')
    y_initial = convertToBCWH(args, y_initial_class.dynam, 'y')
    initial_name = 'TrnOnRA'
    
    # Process forecast data
    X_fc = convertToBCWH(args, X_fc_class.dynam, 'X')
    y_fc = convertToBCWH(args, y_fc_class.refer.isel(pdf=0).expand_dims(pdf=X_fc_class.dynam.pdf.data).assign_coords(pdf=X_fc_class.dynam.pdf.data), 'y')
    
    # Process hindcast data
    X_hc = convertToBCWH(args, X_hc_class.dynam, 'X')
    y_hc = convertToBCWH(args, y_hc_class.refer.isel(pdf=0).expand_dims(pdf=X_hc_class.dynam.pdf.data).assign_coords(pdf=X_hc_class.dynam.pdf.data), 'y')
    
    return X_initial, y_initial, X_initial_class, y_initial_class, initial_name, \
            X_fc, y_fc, X_fc_class, y_fc_class, \
            X_hc, y_hc, X_hc_class, y_hc_class

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Results & Data storage parameters
    parser.add_argument('--save_path', type=str, default='/home/tganglin/data/hourly_numpy_test_Ganglin/raw/large_scale/non_calib/')
    parser.add_argument('--base_dir', type=str, default='data/results/')
    parser.add_argument('--debug', type=int, default=0)

    # Preprocessing parameters
    parser.add_argument('--trn_type', type=str, default='RA')
    parser.add_argument('--predors', type=str, default='z500')
    parser.add_argument('--prednds', type=str, default='ws100')
    parser.add_argument('--trainingDomain', type=str, default='EADToEAD') 
    parser.add_argument('--trainDailyMean', type=int, default=0)
    parser.add_argument('--trainPntOrSeq', type=str, default='PntReg')
    parser.add_argument('--calib_method', type=str, default='MVA_TGL_PDF_HL2')
    parser.add_argument('--CL_years', type=str, default='15y')
    parser.add_argument('--processor', type=int, default=2)
    parser.add_argument('--standardize', type=str, default='Norm0')
    parser.add_argument('--fcProcessor', type=str, default='subCL')

    parser.add_argument('--noise_level', type=float, default=0.1)

    # Training settings
    parser.add_argument('--useOptuna', type=int, default=1)  # Whether to use Optuna for hyperparameter optimization
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--model_Size', type=int, default=8)
    parser.add_argument('--model_level', type=int, default=4)
    parser.add_argument('--outConvType', type=str, default='OutConvSimple')
    parser.add_argument('--test_Full', type=int, default=1)
    parser.add_argument('--with_skips', type=str, default='1_1_1_1')
    parser.add_argument('--outer_fold', type=int, default=0)
    
    # Model selection
    parser.add_argument('--trainingModels', type=str, default='MLR')

    # Stochasticity parameters
    parser.add_argument('--stocha_size', type=int, default=20)

    # Parse arguments and update them
    args = parser.parse_args()
    args = update_easy_args(args)
    print('=========================')

    # Split predictor and predictand variables
    predictors = args.predors.split('_')
    predictand = args.prednds.split('_')
    
    # Load and prepare datasets
    X_initial, y_initial, X_initial_class, y_initial_class, initial_name, \
    X_fc, y_fc, X_fc_class, y_fc_class, \
    X_hc, y_hc, X_hc_class, y_hc_class = \
    get_data(args, args.trn_type, predictors, predictand)
    
    # Set number of training examples
    args.num_train_imgs = y_initial.sample.size
    
    # Print data dimensions for debugging
    print('X_in', X_initial.dims, X_initial.shape)
    print('y_in', y_initial.dims, y_initial.shape)
    print('X_fc', X_fc.dims, X_fc.shape)
    print('y_fc', y_fc.dims, y_fc.shape)
    print('X_hc', X_hc.dims, X_hc.shape)
    print('y_hc', y_hc.dims, y_hc.shape)

    # Define available models
    all_models_class = [MLR, CNN]
    args.trainingModels = args.trainingModels.split(',')
    
    # Filter models based on command line arguments
    trainingModels = [ele for ele in all_models_class if ele.__name__ in args.trainingModels]
    
    # Prepare test datasets
    test_datasets = [
            (X_initial, y_initial, X_initial_class, y_initial_class, 'Eval_in' if not args.test_Full else 'EvalFull_in'),
            (X_fc, y_fc, X_fc_class, y_fc_class, 'Eval_fc' if not args.test_Full else 'EvalFull_fc'),
            (X_hc, y_hc, X_hc_class, y_hc_class, 'Eval_hc' if not args.test_Full else 'EvalFull_hc'),
        ]
    
    finetune_datasets = []
    
    # Train each selected model
    for cur_model in trainingModels: 
        # Copy arguments for current model
        model_args = copy.copy(args)
        model_args = update_model_args(model_args, cur_model)
        
        # Initialize appropriate trainer based on model type
        if cur_model in [MLR]:
            # Initialize trainer for MLR model
            trainer = ModelTrainer(model_args, cur_model, get_MLR_params(args, X_initial, y_initial))
        else:
            # Initialize trainer for CNN model
            trainer = ModelTrainer(model_args, cur_model, {'args': model_args})

        # Run initial training experiment and evaluation
        trainer.run_initial_train_experiment(X_initial, y_initial, X_initial_class, y_initial_class, initial_name, test_datasets)
