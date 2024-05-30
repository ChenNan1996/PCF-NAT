# --------------------------------------------------------
# PCF-NATransformer: https://arxiv.org/abs/2405.12031
# Written by Nian Li
# email: 1173417216@qq.com
# --------------------------------------------------------

import torch
from torch.utils.cpp_extension import load

folder = "./na1d_tensorcore/"

bf16_cuda_module = load(
    name="bf16_ana1d",
    extra_include_paths=[folder],
    sources=[
        folder+"bf16_ana1d.cpp",
        folder+"bf16_ana1d_kernel.cu",
    ],
    verbose=False,
)
fp16_cuda_module = load(
    name="fp16_ana1d",
    extra_include_paths=[folder],
    sources=[
        folder+"fp16_ana1d.cpp",
        folder+"fp16_ana1d_kernel.cu",
    ],
    verbose=False,
)


class bf16_ANA1d_QK_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, win_size, left, dilation=1):
        assert q.is_contiguous()
        assert k.is_contiguous()
        assert q.dtype==k.dtype
        N, T, C = q.shape
        attn = torch.empty((N,T,win_size), dtype=torch.float32, device=q.device)
        bf16_cuda_module.torch_ana_qk_forward(attn, q, k, N, T, C, win_size, left)
        ctx.save_for_backward(q, k)
        ctx.win_size = win_size
        ctx.left = left
        return attn
        
    @staticmethod
    def backward(ctx, attn_gd):
        q, k = ctx.saved_tensors
        N, T, C = q.shape
        q_gd = torch.empty(q.shape, dtype=q.dtype, device=attn_gd.device)
        bf16_cuda_module.torch_ana_q_backward(q_gd, attn_gd, k,  N, T, C, ctx.win_size, ctx.left)
        k_gd = torch.empty(k.shape, dtype=k.dtype, device=attn_gd.device)
        bf16_cuda_module.torch_ana_k_backward(k_gd, attn_gd, q,  N, T, C, ctx.win_size, ctx.left)
        return (q_gd, k_gd, None, None, None)
        
class bf16_ANA1d_AV_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn, v, win_size, left, dilation=1):
        assert attn.is_contiguous()
        assert v.is_contiguous()
        N, T, C = v.shape
        result = torch.empty(v.shape, dtype=v.dtype, device=v.device)
        bf16_cuda_module.torch_ana_av_forward(result, attn, v, N, T, C, win_size, left)
        ctx.save_for_backward(attn, v)
        ctx.win_size = win_size
        ctx.left = left
        return result
        
    @staticmethod
    def backward(ctx, result_gd):
        result_gd = result_gd.contiguous()
        N, T, C = result_gd.shape
        attn, v = ctx.saved_tensors
        attn_gd = torch.empty(attn.shape, dtype=attn.dtype, device=result_gd.device)
        bf16_cuda_module.torch_ana_a_backward(attn_gd, result_gd, v,  N, T, C, ctx.win_size, ctx.left)
        v_gd = torch.empty(v.shape, dtype=v.dtype, device=result_gd.device)
        bf16_cuda_module.torch_ana_v_backward(v_gd, result_gd, attn,  N, T, C, ctx.win_size, ctx.left)
        return (attn_gd, v_gd, None, None, None)

class fp16_ANA1d_QK_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, win_size, left, dilation=1):
        assert q.is_contiguous()
        assert k.is_contiguous()
        assert q.dtype==k.dtype
        N, T, C = q.shape
        attn = torch.empty((N,T,win_size), dtype=torch.float32, device=q.device)
        fp16_cuda_module.torch_ana_qk_forward(attn, q, k, N, T, C, win_size, left)
        ctx.save_for_backward(q, k)
        ctx.win_size = win_size
        ctx.left = left
        return attn
        
    @staticmethod
    def backward(ctx, attn_gd):
        q, k = ctx.saved_tensors
        N, T, C = q.shape
        q_gd = torch.empty(q.shape, dtype=q.dtype, device=attn_gd.device)
        fp16_cuda_module.torch_ana_q_backward(q_gd, attn_gd, k,  N, T, C, ctx.win_size, ctx.left)
        k_gd = torch.empty(k.shape, dtype=k.dtype, device=attn_gd.device)
        fp16_cuda_module.torch_ana_k_backward(k_gd, attn_gd, q,  N, T, C, ctx.win_size, ctx.left)
        return (q_gd, k_gd, None, None, None)
        
class fp16_ANA1d_AV_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn, v, win_size, left, dilation=1):
        assert attn.is_contiguous()
        assert v.is_contiguous()
        N, T, C = v.shape
        result = torch.empty(v.shape, dtype=v.dtype, device=v.device)
        fp16_cuda_module.torch_ana_av_forward(result, attn, v, N, T, C, win_size, left)
        ctx.save_for_backward(attn, v)
        ctx.win_size = win_size
        ctx.left = left
        return result
        
    @staticmethod
    def backward(ctx, result_gd):
        result_gd = result_gd.contiguous()
        N, T, C = result_gd.shape
        attn, v = ctx.saved_tensors
        attn_gd = torch.empty(attn.shape, dtype=attn.dtype, device=result_gd.device)
        fp16_cuda_module.torch_ana_a_backward(attn_gd, result_gd, v,  N, T, C, ctx.win_size, ctx.left)
        v_gd = torch.empty(v.shape, dtype=v.dtype, device=result_gd.device)
        fp16_cuda_module.torch_ana_v_backward(v_gd, result_gd, attn,  N, T, C, ctx.win_size, ctx.left)
        return (attn_gd, v_gd, None, None, None)

def getNA1dFunction(C, dilation, dtype):
    assert C%8==0
    assert dilation==1
        
    if dtype==torch.bfloat16:
        return bf16_ANA1d_QK_Function, bf16_ANA1d_AV_Function
    elif dtype==torch.float16:
        return fp16_ANA1d_QK_Function, fp16_ANA1d_AV_Function
    else:
        assert False