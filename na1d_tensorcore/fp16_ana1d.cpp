#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>
#include "cuda_fp16.h"

#include "fp16_ana1d.h"

void torch_ana_qk_forward(torch::Tensor &attn,
                       const torch::Tensor &query,
                       const torch::Tensor &key,
                       const int64_t N,
                       const int64_t T,
                       const int64_t C,
                       const int64_t win_size,
                       const int64_t left) {
    if(C%16==0){
        fp16_ana_qk_forward_C16((float *)attn.data_ptr(),
                        (const __half *)query.data_ptr(),
                        (const __half *)key.data_ptr(),
                        N, T, C, win_size, left);
    }else{
        fp16_ana_qk_forward_C8((float *)attn.data_ptr(),
                        (const __half *)query.data_ptr(),
                        (const __half *)key.data_ptr(),
                        N, T, C, win_size, left);
    }
}

void torch_ana_q_backward(torch::Tensor &query_gd,
                       const torch::Tensor &attn_gd,
                       const torch::Tensor &key,
                       const int64_t N,
                       const int64_t T,
                       const int64_t C,
                       const int64_t win_size,
                       const int64_t left) {
    fp16_ana_q_backward((__half *)query_gd.data_ptr(),
                    (const float *)attn_gd.data_ptr(),
                    (const __half *)key.data_ptr(),
                    N, T, C, win_size, left);
}

void torch_ana_k_backward(torch::Tensor &key_gd,
                       const torch::Tensor &attn_gd,
                       const torch::Tensor &query,
                       const int64_t N,
                       const int64_t T,
                       const int64_t C,
                       const int64_t win_size,
                       const int64_t left) {
    fp16_ana_k_backward((__half *)key_gd.data_ptr(),
                    (const float *)attn_gd.data_ptr(),
                    (const __half *)query.data_ptr(),
                    N, T, C, win_size, left);
}

void torch_ana_av_forward(torch::Tensor &result,
                       const torch::Tensor &attn,
                       const torch::Tensor &value,
                       const int64_t N,
                       const int64_t T,
                       const int64_t C,
                       const int64_t win_size,
                       const int64_t left) {
    fp16_ana_q_backward((__half *)result.data_ptr(),
                    (const float *)attn.data_ptr(),
                    (const __half *)value.data_ptr(),
                    N, T, C, win_size, left);
}

void torch_ana_a_backward(torch::Tensor &attn_gd,
                       const torch::Tensor &result_gd,
                       const torch::Tensor &value,
                       const int64_t N,
                       const int64_t T,
                       const int64_t C,
                       const int64_t win_size,
                       const int64_t left) {
    if(C%16==0){
        fp16_ana_qk_forward_C16((float *)attn_gd.data_ptr(),
                        (const __half *)result_gd.data_ptr(),
                        (const __half *)value.data_ptr(),
                        N, T, C, win_size, left);
    }else{
        fp16_ana_qk_forward_C8((float *)attn_gd.data_ptr(),
                        (const __half *)result_gd.data_ptr(),
                        (const __half *)value.data_ptr(),
                        N, T, C, win_size, left);
    }
}

void torch_ana_v_backward(torch::Tensor &value_gd,
                       const torch::Tensor &result_gd,
                       const torch::Tensor &attn,
                       const int64_t N,
                       const int64_t T,
                       const int64_t C,
                       const int64_t win_size,
                       const int64_t left) {
    fp16_ana_k_backward((__half *)value_gd.data_ptr(),
                    (const float *)attn.data_ptr(),
                    (const __half *)result_gd.data_ptr(),
                    N, T, C, win_size, left);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_ana_qk_forward",
          &torch_ana_qk_forward,
          "ana_qk_forward kernel warpper");
    m.def("torch_ana_q_backward",
          &torch_ana_q_backward,
          "ana_q_backward kernel warpper");
    m.def("torch_ana_k_backward",
          &torch_ana_k_backward,
          "ana_k_backward kernel warpper");
    m.def("torch_ana_av_forward",
          &torch_ana_av_forward,
          "ana_av_forward kernel warpper");
    m.def("torch_ana_a_backward",
          &torch_ana_a_backward,
          "ana_a_backward kernel warpper");
    m.def("torch_ana_v_backward",
          &torch_ana_v_backward,
          "ana_v_backward kernel warpper");
}

TORCH_LIBRARY(fp16_ana1d, m) {
    m.def("torch_ana_qk_forward", torch_ana_qk_forward);
    m.def("torch_ana_q_backward", torch_ana_q_backward);
    m.def("torch_ana_k_backward", torch_ana_k_backward);
    m.def("torch_ana_av_forward", torch_ana_av_forward);
    m.def("torch_ana_a_backward", torch_ana_a_backward);
    m.def("torch_ana_v_backward", torch_ana_v_backward);
}