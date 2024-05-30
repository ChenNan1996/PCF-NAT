# --------------------------------------------------------
# reference: Swin Transformer https://github.com/microsoft/Swin-Transformer
# reference: Neighborhood Attention Transformer https://github.com/SHI-Labs/Neighborhood-Attention-Transformer
# reference: ECAPA-TDNN https://www.isca-archive.org/interspeech_2020/desplanques20_interspeech.html
# reference: SpeechBrain https://github.com/speechbrain/speechbrain

# PCF-NATransformer: https://arxiv.org/abs/2405.12031
# Written by Nian Li
# email: 1173417216@qq.com
# --------------------------------------------------------
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math

from models.Modules import TDNNBlock, BatchNorm1d, get_posemb, DropPath
from models.Pooling import AttentiveStatisticsPooling, StatisticsPooling

from na1d_tensorcore.nattn import getNA1dFunction


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NAttention(nn.Module):
    def __init__(self, frames, dim, num_heads, win_size, dilation=1, proj_bias=True, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., dtype=torch.float32):
        super().__init__()
        assert ( isinstance(win_size, int) and win_size>0 ) or ( isinstance(win_size, list) and len(win_size)==2 )
        if isinstance(win_size, int):
            left = win_size-1-win_size//2
        else:
            left = win_size[1]
            win_size = win_size[0]
        assert win_size<=80
        assert left>=0 and win_size>left
        self.win_size = win_size
        self.left = left
        
        self.dim = dim
        self.dilation = dilation
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim%num_heads==0 and head_dim%8==0
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        #if enable_pos:
        self.relative_position_bias = nn.Parameter( torch.zeros((1, num_heads, 1, win_size)) )
        nn.init.normal_(self.relative_position_bias, mean=0., std=.02)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        
        padding = torch.zeros(1, dtype=torch.int)
        if dilation>1:
            self.unfold = nn.Unfold(kernel_size=(win_size,1), stride=1, dilation=(dilation,1)) 
            attn_mask = self.get_attn_mask_dilation(frames, win_size, left, padding)
        else:
            attn_mask = self.get_attn_mask(frames, win_size, left, padding)
        self.register_buffer("attn_mask", attn_mask)

        self.NA1d_QK_Function, self.NA1d_AV_Function = getNA1dFunction(C=dim//num_heads, dilation=dilation, dtype=dtype)
            
    def get_attn_mask(self, T, win_size, left, padding=None, dtype=torch.float32, device='cpu'):
        right = win_size-left-1
        if padding is None:
            padding = torch.zeros(1, dtype=torch.int, device=device)
        mask = torch.arange(T, device=padding.device).unsqueeze(0)>=(T-padding.unsqueeze(1)) # ( N, T )

        row = mask.unsqueeze(-1).unsqueeze(-1) # ( N, T, 1, 1 )
        col = F.pad(mask, pad=(left,right), mode='constant', value=True) # ( N, pl+T+pr )
        col = col.unfold(dimension=1, size=win_size, step=1) # ( N, T, win_size )
        col = col.unsqueeze(-2) # ( N, T, 1, win_size )
        mask = row + col # ( N, T, 1, win_size )
        mask = torch.zeros(mask.shape, device=padding.device, dtype=dtype).masked_fill(mask, float(-100.))
        #return mask.unsqueeze(-2) # ( N, T, 1, 1, win_size )
        return mask.squeeze(-2).unsqueeze(1) # ( N, 1, T, win_size )
        
    def get_attn_mask_dilation(self, T, win_size, left, padding=None, dtype=torch.float32, device='cpu'):
        right = win_size-left-1
        if padding is None:
            padding = torch.zeros(1, dtype=torch.int, device=device)
        not_mask = torch.arange(T, device=padding.device).unsqueeze(0)<(T-padding.unsqueeze(1)) # [N, T]
        not_mask = not_mask.unsqueeze(2).unsqueeze(1) # [N, 1, T, 1]
        row = not_mask
        
        col = F.pad(not_mask, pad=(0,0,left*self.dilation,right*self.dilation), mode='constant', value=False) # ( N, 1, pl+T+pr, 1 )
        
        col = self.unfold(col.type(torch.float32)).type(torch.bool) # [N, 1*win_size, T']
        col = col.transpose(1,2).unsqueeze(1) # [N, 1, T', win_size]
        not_mask = row * col # [N, 1, T, win_size]
        #not_mask = not_mask.squeeze(1).unsqueeze(2).unsqueeze(3) # ( N, T, 1, 1, win_size )
        
        mask = torch.zeros(not_mask.shape, device=padding.device, dtype=dtype).masked_fill(~not_mask, float(-100.))
        return mask
        
    def forward(self, x, padding=None):
        left = self.left
        win_size = self.win_size
        dilation = self.dilation
        num_heads = self.num_heads
        N, T, C = x.shape
        head_dim = C // num_heads
        
        x = self.qkv(x) # ( N, T, 3*C )
        x = x.view(N, T, 3, num_heads, head_dim)

        if self.training:
            attn_mask = self.attn_mask
        else:
            attn_mask = self.get_attn_mask(T, win_size, left, padding, dtype=x.dtype, device=x.device) if dilation==1 else self.get_attn_mask_dilation(T, win_size, left, padding, dtype=x.dtype, device=x.device)
            
        if x.device.type=='cuda':
            x = x.permute(2,0,3,1,4).contiguous() # ( 3, N, num_heads, T, head_dim )
            x = x.view(3, N*num_heads, T, head_dim)
            q, k, v = x[0], x[1], x[2] # ( N*num_heads, T, head_dim )
            
            attn = self.NA1d_QK_Function.apply(q*self.scale, k, win_size, left, dilation) # ( N*num_heads, T, win_size )
            
            attn = attn.view(N, num_heads, T, win_size) # ( N, num_heads, T, win_size )
            attn = attn + self.relative_position_bias
            attn = attn + attn_mask
            attn = attn.view(N*num_heads, T, win_size)
            
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = self.NA1d_AV_Function.apply(attn, v, win_size, left, dilation) # ( N*num_heads, T, head_dim )
            x = x.view(N,num_heads,T,head_dim)
            x = x.transpose(1,2).contiguous().view(N,T,C)
        else:
            right = win_size-left-1
            x = x.permute(2,0,1,3,4).contiguous() # ( 3, N, T, num_heads, head_dim )
            q, k, v = x[0], x[1], x[2] # ( N, T, num_heads, head_dim )
            
            if dilation>1:
                attn = (q.unsqueeze(-2)*self.scale) @ self.unfold(F.pad(k, pad=(0,0,0,0,left*dilation,right*dilation), mode='constant', value=0).permute(0,2,3,1).unsqueeze(-1).contiguous().view(N*num_heads, head_dim, -1, 1)).contiguous().view(N, num_heads, head_dim, win_size, -1).permute(0,4,1,2,3).contiguous()
            else:
                attn = (q.unsqueeze(-2)*self.scale) @ F.pad(k, pad=(0,0,0,0,left,right), mode='constant', value=0).unfold(dimension=1, size=win_size, step=1) # ( N, T, num_heads, 1, win_size )
                
            attn = attn + self.relative_position_bias.unsqueeze(1)
            attn = attn + attn_mask.squeeze(1).unsqueeze(2).unsqueeze(3)
            attn = self.softmax(attn)
            attn = attn.type(x.dtype)
            attn = self.attn_drop(attn)
            
            if dilation>1:
                x = attn @ self.unfold(F.pad(v, pad=(0,0,0,0,left*dilation,right*dilation), mode='constant', value=0).permute(0,2,3,1).unsqueeze(-1).contiguous().view(N*num_heads, head_dim, -1, 1)).contiguous().view(N, num_heads, head_dim, win_size, -1).permute(0,4,1,3,2).contiguous()
            else:
                x = attn @ F.pad(v, pad=(0,0,0,0,left,right), mode='constant', value=0).unfold(dimension=1, size=win_size, step=1).transpose(-2,-1) # ( N, T, num_heads, 1, head_dim )
                
            x = x.squeeze(-2).contiguous().view(N,T,C)
            
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalAttention(nn.Module):
    def __init__(self, dim, num_heads, proj_bias=True, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.linear_pos = nn.Linear(dim, dim, bias=False)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, pos_emb, attn_mask=None):
        N, T, C = x.shape

        x = x + self.linear_pos(pos_emb)
        
        x = self.qkv(x) # [N, T, 3*C]
        x = x.view(N, T, 3, self.num_heads, C//self.num_heads)
        x = x.permute(2,0,3,1,4) # (3, N, num_heads, T, head_dim)
        q, k, v = x[0], x[1], x[2]
        
        attn = (q*self.scale) @ k.transpose(-2, -1) # (N, num_heads, T, T)
        
        if attn_mask is not None:
            attn = attn + attn_mask
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = attn @ v # (N, num_heads, T, head_dim)
        x = x.transpose(1,2).contiguous().view(N,T,C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class NATFLayer(nn.Module):
    def __init__(
        self, frames, dim, num_heads, win_size, dilation=1,
        bias=True, qkv_bias=True, qk_scale=None, mlp_ratio=4., attn_drop=0., drop=0., drop_path=0.,
        act_layer=nn.GELU, norm_layer=nn.LayerNorm, dtype=torch.float32
    ):
        super().__init__()
        assert isinstance(dilation, int)
        assert isinstance(win_size, int) or isinstance(win_size, list)
        self.win_size = win_size
        if isinstance(win_size, list) or win_size>0:
            self.local = True
            self.attn = NAttention(frames, dim, num_heads, win_size, dilation, proj_bias=bias, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, dtype=dtype)
        else:
            self.local = False
            assert dilation==1
            self.attn = GlobalAttention(dim, num_heads, proj_bias=bias, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        
    def forward(self, x, padding=None, pos_emb=None, attn_mask_global=None):
        #input: ( N, T, C )
        shortcut = x
        x = self.norm1(x)

        if self.local:
            x = self.attn(x, padding=padding) # (N, T, C)
        else:
            x = self.attn(x, pos_emb=pos_emb, attn_mask=attn_mask_global) # (N, T, C)
            
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class NATFBlock(nn.Module):
    def __init__(self, frames, depth, dim, num_heads, win_size, dilation=1,
                 attn_proj_bias=True, qkv_bias=True, qk_scale=None, mlp_ratio=4., attn_drop=0., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, use_checkpoint=False, dtype=torch.float32,
                ):
        super().__init__()
        assert isinstance(num_heads, int) or (isinstance(num_heads, list) and len(num_heads)==depth)
        assert isinstance(win_size, int) or (isinstance(win_size, list) and len(win_size)==depth)
        assert isinstance(dilation, int) or (isinstance(dilation, list) and len(dilation)==depth)
        assert isinstance(use_checkpoint, bool) or (isinstance(use_checkpoint, list) and len(use_checkpoint)==depth)
        
        self.depth = depth
        self.use_checkpoint = [use_checkpoint]*depth if isinstance(use_checkpoint, bool) else use_checkpoint
            
        self.blocks = nn.ModuleList([
            NATFLayer(
                frames=frames, dim=dim, 
                num_heads=num_heads[i] if isinstance(num_heads, list) else num_heads, 
                win_size=win_size[i] if isinstance(win_size, list) else win_size, 
                dilation=dilation[i] if isinstance(dilation, list) else dilation, 
                bias=attn_proj_bias, qkv_bias=qkv_bias, qk_scale=qk_scale, mlp_ratio=mlp_ratio, attn_drop=attn_drop, drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, dtype=dtype,
            )
            for i in range(depth)])

    def forward(self, x, padding=None, pos_emb=None, attn_mask_global=None):
        for i in range(self.depth):
            if self.use_checkpoint[i] and self.training:
                x = checkpoint.checkpoint(self.blocks[i], x, padding, pos_emb, attn_mask_global)
            else:
                x = self.blocks[i](x, padding=padding, pos_emb=pos_emb, attn_mask_global=attn_mask_global)   
        return x


class PatchEmbed1(nn.Module):
    def __init__(self, in_features, embed_dim, bias=True, norm_layer=None):
        super().__init__()
        self.proj = nn.Linear(in_features, embed_dim, bias=bias)
        self.norm = None if norm_layer is None else norm_layer(embed_dim)

    def forward(self, x):
        #input: N, F, T
        x = self.proj(x.transpose(1, 2))
        if self.norm is not None:
            x = self.norm(x)
        return x # N, T_, C
    
class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size=1, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = None if norm_layer is None else norm_layer(embed_dim)

    def forward(self, x):
        #input: N, F, T
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x.transpose(1, 2) # N, T_, C
        
            
class MFA_NATransformer(nn.Module):
    def __init__(
        self, frames=300, patch_size=2, in_chans=80, output_emb_size=192,
        embed_dim=256, depths=[4]*4, num_heads=[[16]*4,[16,16,16,4],[16]*4,[16,16,16,4]],
        win_size=[[27]*4,[27,27,27,0],[27]*4,[27,27,27,0]], dilation=1,
        attn_proj_bias=True, qkv_bias=True, qk_scale=None, mlp_ratio=4, attn_drop=0., drop=0., drop_path_rate=0.,
        patch_norm=True, mfa=True, proj_channel=1536, asp=True, poolnorm=True,
        use_checkpoint=False, dtype=torch.bfloat16,
    ):
        super().__init__()
        assert isinstance(depths, list)
        num_blocks = len(depths)
        assert isinstance(num_heads, int) or (isinstance(num_heads, list) and len(num_heads)==num_blocks)
        assert isinstance(win_size, int) or (isinstance(win_size, list) and len(win_size)==num_blocks)
        assert isinstance(dilation, int) or (isinstance(dilation, list) and len(dilation)==num_blocks)
        assert isinstance(use_checkpoint, bool) or (isinstance(use_checkpoint, list) and len(use_checkpoint)==num_blocks)
        self.frames = frames
        self.win_size = win_size
        self.patch_size = patch_size
        emb_frames = frames//patch_size
        self.mfa = mfa
        
        if patch_size==1:
            self.patch_embed = PatchEmbed1(in_features=in_chans, embed_dim=embed_dim, norm_layer=BatchNorm1d if patch_norm else None)
        else:
            self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size, norm_layer=nn.BatchNorm1d if patch_norm else None)

        self.attn_global = False
        for i in range(len(win_size)):
            if isinstance(win_size[i], list):
                for j in range(len(win_size[i])):
                    if isinstance(win_size[i][j], int) and win_size[i][j]<1:
                        self.attn_global = True
            elif win_size[i]<1:
                self.attn_global = True
            
        if self.attn_global:
            pos_emb = get_posemb(dim=embed_dim, max_len=emb_frames)
            self.register_buffer("pos_emb", pos_emb)
            self.pos_emb_eval = None
        
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        print('dpr=', dpr)

        # build layers
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = NATFBlock(
                frames=emb_frames, depth=depths[i], dim=embed_dim, 
                num_heads=num_heads[i] if isinstance(num_heads, list) else num_heads, 
                win_size=win_size[i] if isinstance(win_size, list) else win_size, 
                dilation=dilation[i] if isinstance(dilation, list) else dilation, 
                attn_proj_bias=attn_proj_bias, qkv_bias=qkv_bias, qk_scale=qk_scale, mlp_ratio=mlp_ratio, attn_drop=attn_drop, drop=drop,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=BatchNorm1d, dtype=dtype,
                use_checkpoint=use_checkpoint[i] if isinstance(use_checkpoint, list) else use_checkpoint, 
            )
            self.blocks.append(block)

        self.norm = nn.BatchNorm1d(embed_dim*num_blocks if mfa else embed_dim)
        self.proj = TDNNBlock(in_channels=embed_dim*num_blocks if mfa else embed_dim, out_channels=proj_channel, kernel_size=1, norm_layer=nn.BatchNorm1d)

        self.pool = AttentiveStatisticsPooling(in_channels=proj_channel, norm_layer=nn.BatchNorm1d) if asp else StatisticsPooling()
        self.pool_norm = nn.BatchNorm1d(proj_channel*2) if poolnorm else nn.Identity()
            
        self.fc = nn.Linear(proj_channel*2, output_emb_size)

        self.apply(self._init_weights)
        
    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0., std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias'}
        
    def get_attn_mask_global(self, T, padding=None, dtype=torch.float32, device='cpu'):
        if padding is None:
            padding = torch.zeros(1, dtype=torch.int, device=device)
        mask = torch.arange(T, device=padding.device).unsqueeze(0)>=(T-padding.unsqueeze(1)) # ( N, T )
        row = mask.unsqueeze(-1)
        col = mask.unsqueeze(-2)
        mask_global = row + col # ( N, T, T )
        mask_global = torch.zeros(mask_global.shape, device=padding.device, dtype=dtype).masked_fill(mask_global, float(-100.))
        return mask_global.unsqueeze(1) # .unsqueeze to match the dim of num_heads
        
    def forward(self, x, lens=None):#input: N, F, T
        T = x.shape[-1]
        assert self.frames==T or not self.training
        
        padding = None
        if lens is not None:
            padding = T - torch.round(lens * T)
            padding = padding//self.patch_size
            
        x = self.patch_embed(x) # N, T//2, C

        T = x.shape[1]
        win_size = self.win_size

        pos_emb = None
        attn_mask_global = None
        if self.attn_global:
            if self.training:
                pos_emb = self.pos_emb
            else:
                if self.pos_emb_eval is None:
                    self.pos_emb_eval = get_posemb(dim=x.shape[-1], max_len=40000).to(x.device)
                pos_emb = self.pos_emb_eval[:,:T,:]
                
                if padding is not None:
                    attn_mask_global = self.get_attn_mask_global(T, padding=padding, dtype=x.dtype, device=x.device)
        
        if self.mfa:
            xs = []
            for block in self.blocks:
                x = block(x, padding=padding, pos_emb=pos_emb, attn_mask_global=attn_mask_global) # N, T//2, C
                xs.append(x)
            x = torch.cat(xs, dim=-1) # N, T//2, C*num_blocks
        else:
            for block in self.blocks:
                x = block(x, padding=padding, pos_emb=pos_emb, attn_mask_global=attn_mask_global) # N, T//2, C

        x = x.transpose(1, 2) # N, C_, T//2
        
        x = self.norm(x)
        x = self.proj(x)
        
        x = self.pool(x, lens=lens)
        x = self.pool_norm(x)
        
        x = self.fc(x)
        #print('fc: x.shape=', x.shape)
        return x