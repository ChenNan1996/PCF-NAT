#include "cuda_fp16.h"

void fp16_ana_qk_forward_C16(float *attn,
                 const __half *query,
                 const __half *key,
                 const int N,
                 const int T,
                 const int C,
                 const int win_size,
                 const int left);

void fp16_ana_qk_forward_C8(float *attn,
                 const __half *query,
                 const __half *key,
                 const int N,
                 const int T,
                 const int C,
                 const int win_size,
                 const int left);

void fp16_ana_q_backward(__half *query_gd,
                 const float *attn_gd,
                 const __half *key,
                 const int N,
                 const int T,
                 const int C,
                 const int win_size,
                 const int left);

void fp16_ana_k_backward(__half *key_gd,
                 const float *attn_gd,
                 const __half *query,
                 const int N,
                 const int T,
                 const int C,
                 const int win_size,
                 const int left);