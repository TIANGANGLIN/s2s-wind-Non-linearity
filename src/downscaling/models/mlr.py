import pytorch_lightning as pl
import numpy as np
import torch.nn as nn
from torch import optim
from downscaling.utils.loss_func import criterion_selection

class LinearWithReshape(nn.Module):
    def __init__(self, input_size, output_size, target_size):
        super(LinearWithReshape, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.target_size = target_size

    def forward(self, x):
        x_shape = x.shape
        if len(x_shape)==5:
            batch_size, pdf, channel, height, width = x.size()
            x = x.reshape(batch_size * pdf * channel, height * width)
        else:
            batch_size, channel, height, width = x.size()
            x = x.reshape(batch_size, channel * height * width)

        x = self.linear(x)
        if len(x_shape)==5:
            if len(self.target_size)==5:
                x = x.reshape(batch_size, pdf, *self.target_size[1:])
            else:
                x = x.reshape(batch_size, pdf, *self.target_size[1:])
        else:
            x = x.reshape(batch_size, *self.target_size)
        return x

class MLR(pl.LightningModule):
    def __init__(self, args, input_size, output_size, target_size, weight_decay, lr, fine_tune=False,pretrained_weights=None):
        super(MLR, self).__init__()

        self.linear = LinearWithReshape(input_size, output_size,target_size)
        self.weight_decay = weight_decay
        self.lr = lr
        self.fine_tune = fine_tune
        self.loss_func = criterion_selection('MSE')

        self.add_info = None

    def forward(self, x, t=None):
        x = x.float()
        out = self.linear(x)
        return None, out

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, outputs = self(x)
        loss = self.loss_func(outputs, y, model=self,add_info=self.add_info)
        for metric, value in loss.items():
            self.log(f'train_{metric}', value, on_step=False, on_epoch=True, prog_bar=True)
        return loss['total_loss']

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, outputs = self(x)
        loss = self.loss_func(outputs, y, model=self,add_info=self.add_info)
        for metric, value in loss.items():
            self.log(f'val_{metric}', value, on_step=False, on_epoch=True, prog_bar=True)
        return loss['total_loss']

    def test_step(self, batch, batch_idx):
        x, y = batch
        _, outputs = self(x)
        loss = self.loss_func(outputs, y, model=self,add_info=self.add_info)
        for metric, value in loss.items():
            self.log(f'test_{metric}', value, on_step=False, on_epoch=True, prog_bar=True)
        return loss['total_loss']


    def configure_optimizers(self):
        if self.fine_tune:
            optimizer =  optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer =  optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {"optimizer": optimizer,}

def get_MLR_params(args, X_array, Y_array):
    return {
        "args":args,
        "input_size": np.prod(X_array.shape[1:]),
        "output_size": np.prod(Y_array.shape[1:]),
        "target_size": Y_array.shape[1:],
        }
