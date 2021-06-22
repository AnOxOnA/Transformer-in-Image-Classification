import copy
import torch
import torchvision
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader
from torch.nn.modules.container import ModuleList
from torchvision.transforms import ToTensor

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

class ConvBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride = 1, padding = 0):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride = stride, padding = padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(False)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x 
        

class CTC_block(nn.Module):
    def __init__(self, num_channel, kernel_size, num_head, num_layers, dropout, pool):
        super(CTC_block, self).__init__()
        self.conv1 = ConvBNReLU(num_channel, num_channel, kernel_size, padding=(kernel_size-1)//2)
        self.mhsa = MHSA(num_head, num_channel, num_channel, num_layers, dropout)
        self.conv2 = ConvBNReLU(num_channel, num_channel, kernel_size, padding=(kernel_size-1)//2)
        self.pool = pool
    
    def forward(self, x):
        tmp = x
        x = self.conv1(x)
        x = self.mhsa(x)
        x = self.conv2(x)
        return self.pool(x+tmp)
    
class CTC(nn.Module):
    def __init__(self, CTC_block, num_blocks, norm = None):
        super(CTC, self).__init__()
        self.layers = _get_clones(CTC_block, num_blocks)
        self.num_blocks = num_blocks
        self.norm = norm
    
    def forward(self, x):
        for mod in self.layers:
            x = mod(x)

        if self.norm is not None:
            x = self.norm(x)
        return x

class MHSA(nn.Module):
    def __init__(self, num_head, num_channel, dim_hid, num_layers, dropout):
        super(MHSA, self).__init__()
        self.num_head = num_head
        self.encoder_layers = nn.TransformerEncoderLayer(num_channel, num_head, dim_hid, dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers)
        
    def forward(self, x):
        batch, num_channel, height, width = x.shape
        x = rearrange(x, 'b c h w -> (h w) b c')
        x = self.encoder(x)
        x = rearrange(x, '(h w) b c -> b c h w', h = height)
        return x

# The baseline model with transformer structure.
class NetA(nn.Module):
    def __init__(self, num_class, size, num_channel, num_head, num_layers, num_blocks, dropout):
        super(NetA, self).__init__()
        self.pool = nn.AvgPool2d((3,3), stride = 1, padding = 1)
        self.CTC_block = CTC_block(num_channel, 3, num_head, num_layers, dropout, self.pool)
        self.CTC = CTC(self.CTC_block, num_blocks)
        if size == 8:
            self.backbone = nn.Sequential(
                ConvBNReLU(3, num_channel, 5),
                ConvBNReLU(num_channel, num_channel, 5),
                nn.MaxPool2d((3,3)),
                self.CTC,
                Rearrange('b c h w -> b (c h w)')
            )
            dim_mlp = num_channel * 64
            self.mlp = nn.Sequential(
                nn.LayerNorm(dim_mlp),
                nn.Linear(dim_mlp, num_class)
            )
        else:
            self.backbone = nn.Sequential(
                ConvBNReLU(3, num_channel, 3, padding = 1),
                ConvBNReLU(num_channel, num_channel, 3, padding = 1),
                nn.MaxPool2d((3,3), stride = 2, padding = 1),
                self.CTC,
                Rearrange('b c h w -> b (c h w)')
            )
            dim_mlp = num_channel * 256
            self.mlp = nn.Sequential(
                nn.LayerNorm(dim_mlp),
                nn.Linear(dim_mlp, num_class)
            )
            
    def forward(self, input_images):
        x = self.backbone(input_images)
        return self.mlp(x)

class NormLinear(nn.Module):
    def __init__(self, size, dropout):
        super(NormLinear, self).__init__()
        self.trans = nn.Parameter(torch.randn(size, size))
        self.bias = nn.Parameter(torch.randn(size))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        tmp = x
        alpha = torch.softmax(self.trans, dim = 0)
        x = torch.matmul(x, alpha)+self.bias
        return tmp + self.dropout(x)
    
class TransLinear(nn.Module):
    def __init__(self, dim_model, dim_hid, height, width, num_layers, dropout):
        super(TransLinear, self).__init__()
        size = height * width
        self.normlinear = NormLinear(size, dropout) 
        self.linears = _get_clones(self.normlinear, num_layers)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.lin1 = nn.Linear(dim_model, dim_hid)
        self.lin2 = nn.Linear(dim_hid, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.rearrange = Rearrange('b (h w) c -> b c h w', h = height)
    
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b c (h w)')
        for mod in self.linears:
            x = torch.relu(mod(x))
        x = rearrange(x, 'b c n -> b n c')
        x = self.norm1(x)
        tmp2 = x
        x = self.lin2(self.dropout1(torch.relu(self.lin1(x))))
        x = self.dropout2(x) + tmp2
        x = self.norm2 (x)
        return self.rearrange(x)

class CLC_block(nn.Module):
    def __init__(self, num_channel, kernel_size, height, width, num_layers, dropout):
        super(CLC_block, self).__init__()
        self.conv1 = ConvBNReLU(num_channel, num_channel, kernel_size, padding=(kernel_size-1)//2)
        self.trans = TransLinear(num_channel, num_channel, height, width, num_layers, dropout)
        self.conv2 = ConvBNReLU(num_channel, num_channel, kernel_size, padding=(kernel_size-1)//2)
    
    def forward(self, x):
        tmp = x
        x = self.conv1(x)
        x = self.trans(x)
        x = self.conv2(x)
        return x+tmp

class CLC(nn.Module):
    def __init__(self, CLC_block, num_blocks, norm = None):
        super(CLC, self).__init__()
        self.layers = _get_clones(CLC_block, num_blocks)
        self.num_blocks = num_blocks
        self.norm = norm
    
    def forward(self, x):
        for mod in self.layers:
            x = mod(x)

        if self.norm is not None:
            x = self.norm(x)
        return x

# The comparison model which shares the same number of parameters with the baseline model.(without transformer structure)  
class NetB(nn.Module):
    def __init__(self, size, num_class, num_channel, num_blocks, dropout):
        super(NetB, self).__init__()
        self.pool = nn.AvgPool2d((3,3), stride = 1, padding = 1)
        self.CLC_block = CLC_block(num_channel, 3, size, size, 0, dropout)
        self.CLC = CLC(self.CLC_block, num_blocks)
        if size == 8:
            self.backbone = nn.Sequential(
                ConvBNReLU(3, num_channel, 5),
                ConvBNReLU(num_channel, num_channel, 5),
                nn.MaxPool2d((3,3)),
                self.CLC,
                Rearrange('b c h w -> b (c h w)')
            )
            dim_mlp = num_channel * 64
            self.mlp = nn.Sequential(
                nn.LayerNorm(dim_mlp),
                nn.Linear(dim_mlp, num_class)
            )
        else:
            self.backbone = nn.Sequential(
                ConvBNReLU(3, num_channel, 3, padding = 1),
                ConvBNReLU(num_channel, num_channel, 3, padding = 1),
                nn.MaxPool2d((3,3), stride = 2, padding = 1),
                self.CLC,
                Rearrange('b c h w -> b (c h w)')
            )
            dim_mlp = num_channel * 256
            self.mlp = nn.Sequential(
                nn.LayerNorm(dim_mlp),
                nn.Linear(dim_mlp, num_class)
            )
            
    def forward(self, input_images):
        x = self.backbone(input_images)
        return self.mlp(x)

# The comparison model which shares the same FLOPs with the baseline model.(without transformer structure) 
class NetC(nn.Module):
    def __init__(self, size, num_class, num_channel, num_blocks, dropout):
        super(NetC, self).__init__()
        self.pool = nn.AvgPool2d((3,3), stride = 1, padding = 1)
        self.CLC_block = CLC_block(num_channel, 3, size, size, 3, dropout)
        self.CLC = CLC(self.CLC_block, num_blocks)
        if size == 8:
            self.backbone = nn.Sequential(
                ConvBNReLU(3, num_channel, 5),
                ConvBNReLU(num_channel, num_channel, 5),
                nn.MaxPool2d((3,3)),
                self.CLC,
                Rearrange('b c h w -> b (c h w)')
            )
            dim_mlp = num_channel * 64
            self.mlp = nn.Sequential(
                nn.LayerNorm(dim_mlp),
                nn.Linear(dim_mlp, num_class)
            )
        else:
            self.backbone = nn.Sequential(
                ConvBNReLU(3, num_channel, 3, padding = 1),
                ConvBNReLU(num_channel, num_channel, 3, padding = 1),
                nn.MaxPool2d((3,3), stride = 2, padding = 1),
                self.CLC,
                Rearrange('b c h w -> b (c h w)')
            )
            dim_mlp = num_channel * 256
            self.mlp = nn.Sequential(
                nn.LayerNorm(dim_mlp),
                nn.Linear(dim_mlp, num_class)
            )
            
    def forward(self, input_images):
        x = self.backbone(input_images)
        return self.mlp(x)