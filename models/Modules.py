import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def lens_to_mask_1d(shape, lens):
    #input: shape: (N, C, T), lens: float tensor (N,)
    #output: mask: (N, 1, T), lens_: int tensor (N, 1)
    N, _, T = shape
    lens_dot = torch.round(lens * T).view(N,1,1)
    mask = torch.arange(T, dtype=torch.int, device=lens.device).expand(N,1,T)<lens_dot
    return mask, lens_dot
    
class TDNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, groups=1, bias=True, activation=nn.ReLU(inplace=True), norm_layer=nn.BatchNorm1d):
        super().__init__()
        assert padding != 'same' or stride==1
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.activation = activation
        self.norm = norm_layer(out_channels)

    def forward(self, x):
        return self.norm(self.activation(self.conv(x)))


class BatchNorm1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        if len(x.shape)>=3:
            return self.norm(x.transpose(-2, -1)).transpose(-2, -1)
        else:
            return self.norm(x)


def get_posemb(dim, max_len):
    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term2 = torch.pow(torch.tensor(10000.0),torch.arange(0, dim, 2).float()/dim)
    div_term1 = torch.pow(torch.tensor(10000.0),torch.arange(1, dim, 2).float()/dim)
    pe[:, 0::2] = torch.sin(position * div_term2)
    pe[:, 1::2] = torch.cos(position * div_term1)
    pe = pe.unsqueeze(0) # (1,max_len,dim)
    return pe

def get_posemb2(dim, max_len):
    pe = torch.zeros(dim, max_len)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(0)
    div_term2 = torch.pow(torch.tensor(10000.0),torch.arange(0, dim, 2).float()/dim)
    div_term1 = torch.pow(torch.tensor(10000.0),torch.arange(1, dim, 2).float()/dim)
    pe[0::2, :] = torch.sin(position * div_term2.unsqueeze(1))
    pe[1::2, :] = torch.cos(position * div_term1.unsqueeze(1))
    pe = pe.unsqueeze(0) # (1,dim,max_len)
    return pe


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)