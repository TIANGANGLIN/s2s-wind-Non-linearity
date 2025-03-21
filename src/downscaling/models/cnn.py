import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from downscaling.utils.loss_func import criterion_selection
import pytorch_lightning as pl

class Flatten(nn.Module):
    """A simple flatten layer that reshapes the input tensor to 2D"""
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class ChannelAttention(nn.Module):
    """
    Implements Channel Attention mechanism that helps the model focus on important channels.
    Uses both average and max pooling followed by MLP to generate attention weights.
    
    Args:
        input_channels: Number of input channels
        reduction_ratio: Ratio to reduce channels in the MLP (default: 16)
    """
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #  https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
        #  uses Convolutions instead of Linear
        if input_channels // reduction_ratio==0:
            mid_channels = 1
        else:
            mid_channels = input_channels // reduction_ratio

        # MLP to process pooled features
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, input_channels)
        )

    def forward(self, x):
        # Take the input and apply average and max pooling
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


class SpatialAttention(nn.Module):
    """
    Implements Spatial Attention mechanism that helps focus on important spatial locations.
    Uses both average and max pooling across channels to generate attention maps.
    
    Args:
        kernel_size: Size of convolution kernel (must be 3 or 7)
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        # Generate channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        # Generate and apply spatial attention
        out = self.conv(out)
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale

class DownDS(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, padding, kernels_per_layer=1, dropout=False, dropout_rate=0.5):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer, kernel_size=kernel_size, padding=padding, dropout=dropout, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpDS(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, padding,bilinear=True, kernels_per_layer=1, with_skip=1, dropout=False, dropout_rate=0.5):
        super().__init__()
        self.with_skip = with_skip
        if not with_skip: in_channels = in_channels//2
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvDS(in_channels, out_channels, kernel_size=kernel_size, padding=padding, mid_channels=in_channels // 2, kernels_per_layer=kernels_per_layer, dropout=dropout, dropout_rate=dropout_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDS(in_channels, out_channels, kernel_size=kernel_size, padding=padding, kernels_per_layer=kernels_per_layer, dropout=dropout, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        if self.with_skip:
            x = torch.cat([x2, x1], dim=1)
        else:
            # print("No skip connection")
            x = x1
        return self.conv(x)


class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=1):
        super(DepthwiseSeparableConv, self).__init__()
        # In Tensorflow DepthwiseConv2D has depth_multiplier instead of kernels_per_layer
        self.depthwise = nn.Conv2d(in_channels, in_channels * kernels_per_layer, kernel_size=kernel_size, padding=padding,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class DoubleConvDS(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size, padding, mid_channels=None, kernels_per_layer=1, dropout=False, dropout_rate=0.5):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=kernel_size, kernels_per_layer=kernels_per_layer, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=kernel_size, kernels_per_layer=kernels_per_layer, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout else nn.Identity()
        if dropout:
            print('using dropout')

    def forward(self, x):
        x = self.double_conv(x)
        x = self.dropout(x)
        return x
    
class BaseRAPL(pl.LightningModule):
    def __init__(self,args,weight_decay,lr,fine_tune=False):
        super().__init__()
        self.args = args
        self.weight_decay = weight_decay
        self.lr = lr
        self.configure_criterion()

    def forward(self, x, y=None):
        return self.model(x, y)

    def configure_optimizers(self):
        optimizer =  optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {"optimizer": optimizer,}

    def configure_criterion(self):
        self.loss_func = criterion_selection('MSE')
        return self.loss_func
    
    def epoch(self, batch, batch_idx):
        """
        Epoch details
        """
        x, y = batch
        x, y = x.float(), y.float()
        del batch
        add_info, p = self(x)
        loss = self.loss_func(p, y, model=self,add_info=add_info)

        return loss, p

    def training_step(self, batch, batch_idx):
        loss, _ = self.epoch(batch,batch_idx)
        for metric, value in loss.items():
            self.log(f'train_{metric}', value, on_step=False, on_epoch=True, prog_bar=True)
        return loss['total_loss']

    def validation_step(self, batch, batch_idx):
        loss, p = self.epoch(batch,batch_idx)
        for metric, value in loss.items():
            self.log(f'val_{metric}', value, on_step=False, on_epoch=True, prog_bar=True)
        return loss['total_loss']

    def test_step(self, batch, batch_idx):
        loss, p = self.epoch(batch,batch_idx)
        for metric, value in loss.items():
            self.log(f'test_{metric}', value, on_step=False, on_epoch=True, prog_bar=True)
        return loss['total_loss']


class OutConvSimple(nn.Module):
    def __init__(self, args, in_channels, out_channels,full_reso=1):
        super(OutConvSimple, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # Initialize a learnable bias correction term
        self.bias_correction = nn.Parameter(torch.zeros(1, out_channels, 23, 60))

    def forward(self, x):
        x = self.conv(x)
        # return x
        # Add the bias correction term (broadcast across batch and spatial dimensions)
        return x + self.bias_correction
    
def vae_linear(in_size, out_size, relu=True):
    return nn.Sequential(
        nn.Linear(in_size, out_size),
        nn.LeakyReLU(),
    ) if relu else nn.Linear(in_size, out_size)


#########################################################################################
# Modules for Diffusion SmaAtUnet
#########################################################################################
class DownDSDiffuse(DownDS):
    def __init__(self, in_channels, out_channels, time_emb_dim, kernel_size, padding, kernels_per_layer=1, dropout=False, dropout_rate=0.5):
        super().__init__(in_channels, out_channels, kernel_size, padding, kernels_per_layer, dropout, dropout_rate)
        
        self.emb_layer = nn.Sequential(
            nn.Linear(
                time_emb_dim,
                out_channels
            ),
            nn.ReLU(),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None]
        return x + emb

class UpDSDiffuse(UpDS):
    def __init__(self, in_channels, out_channels, time_emb_dim, kernel_size, padding,bilinear=True, kernels_per_layer=1, with_skip=1, dropout=False, dropout_rate=0.5):
        super().__init__(in_channels, out_channels, kernel_size, padding, bilinear, kernels_per_layer, with_skip, dropout, dropout_rate)
        
        self.with_skip = with_skip
        self.emb_layer = nn.Sequential(
            nn.Linear(
                time_emb_dim,
                out_channels
            ),
            nn.ReLU(),
        )

    def forward(self, x1, x2, t):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        if self.with_skip:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None]
        return x + emb
    
#########################################################################################
# Direct SmaAtUnet
#########################################################################################
class CNN(BaseRAPL):
    # 3D conv -> 2D conv + 1
    def __init__(self,args,weight_decay,lr,fine_tune=False,pretrained_weights=None):
        super().__init__(args,weight_decay,lr,fine_tune)
        self.args = args
        
        self.kernels_per_layer = 2
        self.bilinear=True
        self.with_skips = [np.int16(ele) for ele in args.with_skips.replace(' ','').split('_')]
        self.kernel_size = 3
        self.padding = 1
        reduction_ratio=16
        n = args.model_Size
        factor_l1 = 2 if args.model_level==1 and self.bilinear else 1
        factor_l2 = 2 if args.model_level==2 and self.bilinear else 1
        factor_l3 = 2 if args.model_level==3 and self.bilinear else 1
        factor_l4 = 2 if self.bilinear else 1

        if args.debug and 'num_train_imgs' not in args.__dict__.keys():
            args.num_train_imgs = 100

        self.DropOutList = [False] * 8
        self.dropout_rate = None

        cbam_channels = [n, 2*n//factor_l1, 4*n//factor_l2, 8*n//factor_l3, 16*n//factor_l4]
        cbam_channels = [2 * ele for ele in cbam_channels]
        self.dn_in_channels = cbam_channels[:4]
        self.dn_ot_channels = cbam_channels[1:]
        self.up_in_channels = [2*ele for ele in self.dn_in_channels[::-1]]
        self.up_ot_channels = [ele//factor_l4 if i!=len(self.dn_in_channels)-1 else ele for i,ele in enumerate(self.dn_in_channels[::-1]) ]

        # Initial convolution
        self.inc = DoubleConvDS(args.n_channels, 2*n, kernel_size=self.kernel_size, padding=self.padding, kernels_per_layer=self.kernels_per_layer)

        # CBAM layers
        self.cbams = nn.ModuleList([CBAM(c, reduction_ratio=reduction_ratio) for c in cbam_channels])
        
        # Down layers
        self.downs = nn.ModuleList([DownDS(c_in, c_ot, 
                                           kernels_per_layer=self.kernels_per_layer,
                                           kernel_size=self.kernel_size, 
                                           padding=self.padding, 
                                           dropout=dropout_dn, 
                                           dropout_rate=self.dropout_rate) \
                                    for c_in, c_ot, dropout_dn in zip(self.dn_in_channels,self.dn_ot_channels, self.DropOutList[:4])])
        
        # Up layers
        self.ups = nn.ModuleList([UpDS(c_in, c_ot, 
                                       bilinear=self.bilinear, 
                                       kernels_per_layer=self.kernels_per_layer,
                                       with_skip=with_skip, 
                                       kernel_size=self.kernel_size, 
                                       padding=self.padding, 
                                       dropout=dropout_up, 
                                       dropout_rate=self.dropout_rate) \
                                    for c_in, c_ot, dropout_up, with_skip in zip(self.up_in_channels,self.up_ot_channels, self.DropOutList[4:], self.with_skips)])

        # OutConvolution layer
        self.outc = OutConvSimple(args,2*n, args.n_classes)

    def encoder(self,x):
        x = self.inc(x)
        res = [x]
        for func in self.downs[:self.args.model_level]:
            x = func(x)
            res.append(x)
        return res

    def encoder_Att(self, res):
        res_Att = []
        for func in self.cbams[:len(res)]:
            x = func(res.pop(0))
            res_Att.append(x)
        return res_Att

    def decoder(self,res_Att):
        x = res_Att.pop()        
        for func in self.ups[4-self.args.model_level:]:
            x = func(x,res_Att.pop())

        x = self.outc(x)
        return x

    def direct(self,x):
        # UNet
        res = self.encoder(x)
        res = self.encoder_Att(res)
        x = self.decoder(res)

         # Reshape data for Quantile
        batch, n_class, lat, lon = x.size()
        x = x.view(batch, *self.args.n_classes_dims, lat, lon)      
        return x
    
    def forward(self, x, y=None):
        x = self.direct(x)
        add_info = x.mean(), x.std()
       
        return add_info, x
