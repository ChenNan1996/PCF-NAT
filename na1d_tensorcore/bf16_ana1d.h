#include<cuda_bf16.h>

void bf16_ana_qk_forward_C16(float *attn,
                 const __nv_bfloat16 *query,
                 const __nv_bfloat16 *key,
                 const int N,
                 const int T,
                 const int C,
                 const int win_size,
                 const int left);

void bf16_ana_qk_forward_C8(float *attn,
                 const __nv_bfloat16 *query,
                 const __nv_bfloat16 *key,
                 const int N,
                 const int T,
                 const int C,
                 const int win_size,
                 const int left);

void bf16_ana_q_backward(__nv_bfloat16 *query_gd,
                 const float *attn_gd,
                 const __nv_bfloat16 *key,
                 const int N,
                 const int T,
                 const int C,
                 const int win_size,
                 const int left);

void bf16_ana_k_backward(__nv_bfloat16 *key_gd,
                 const float *attn_gd,
                 const __nv_bfloat16 *query,
                 const int N,
                 const int T,
                 const int C,
                 const int win_size,
                 const int left);