# --------------------------------------------------------
# reference: SpeechBrain https://github.com/speechbrain/speechbrain

# PCF-NATransformer: https://arxiv.org/abs/2405.12031
# Written by Nian Li
# email: 1173417216@qq.com
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Modules import TDNNBlock, lens_to_mask_1d


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, in_channels, attention_channels=128, global_context=True, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(in_channels * 3, attention_channels, 1, 1, norm_layer=norm_layer)
        else:
            self.tdnn = TDNNBlock(in_channels, attention_channels, 1, 1, norm_layer=norm_layer)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(in_channels=attention_channels, out_channels=in_channels, kernel_size=1, padding='same')

    def forward(self, x, lens=None):#x: [N, C*3, T]
        if self.global_context:
            if lens is None:
                mean = torch.mean(x, dim=-1, keepdim=True)
                std = torch.sqrt(torch.var(x, dim=-1, keepdim=True).clamp(self.eps))
                #std = torch.std(x, dim=-1)
            else:
                mask, lens_dot = lens_to_mask_1d(x.shape, lens)
                mean = (x*mask).sum(dim=-1, keepdim=True)/lens_dot
                std = torch.sqrt(
                    (
                        ((x-mean).pow(2) * mask).sum(dim=-1, keepdim=True)/lens_dot
                    ).clamp(self.eps)
                )
            mean = mean.expand(-1, -1, x.shape[-1])
            std = std.expand(-1, -1, x.shape[-1])
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x
        
        attn = self.conv(self.tanh(self.tdnn(attn)))

        if lens is not None:
            attn = attn.masked_fill(mask == 0, -torch.inf)

        attn = F.softmax(attn, dim=-1)
        mean = (attn * x).sum(dim=-1, keepdim=True)
        std = torch.sqrt(
            (attn * (x-mean).pow(2)).sum(dim=-1, keepdim=True).clamp(self.eps)
        )
        pooled_stats = torch.cat((mean, std), dim=1).squeeze(-1)
        return pooled_stats

class StatisticsPooling(nn.Module):
    def __init__(self,):
        super().__init__()
        self.eps = 1e-12
        
    def forward(self, x, lens=None):
        if lens is None:
            mean = torch.mean(x, dim=-1, keepdim=True)
            std = torch.sqrt(torch.var(x, dim=-1, keepdim=True).clamp(self.eps))
            #std = torch.std(x, dim=-1)
        else:
            mask, lens_dot = lens_to_mask_1d(x.shape, lens)
            mean = (x*mask).sum(dim=-1, keepdim=True)/lens_dot
            std = torch.sqrt(
                (
                    ((x-mean).pow(2) * mask).sum(dim=-1, keepdim=True)/lens_dot
                ).clamp(self.eps)
            )
        return torch.cat([mean, std], dim=1).squeeze(-1)