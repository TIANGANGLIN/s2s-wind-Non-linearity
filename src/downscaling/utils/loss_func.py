import torch
import torch.nn.functional as F

# %% Define criterion selection function 

def criterion_selection(loss_name):
    if loss_name=='MSE':
        criterion = my_mse
    else:
        raise ValueError("Error: No this loss function!")
    return criterion 

def my_mse(preds, trgts, model=None,cl=None,add_info=None):
    return {'total_loss':((preds-trgts)**2).mean()}
