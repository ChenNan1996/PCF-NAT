#include <iostream>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>

#define WARP_SIZE 32
#define WMMA_M 16
#define WMMA_N 8
#define WMMA_K 16
#define WMMA2_K 8

#define MAX_wrap_num 16

typedef unsigned __int128 uint128_t;


#define LDMATRIX_X1(R0, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" \
                 : "=r"(R0)                                         \
                 : "r"(addr))

#define LDMATRIX_X2(R0, R1, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                 : "=r"(R0), "=r"(R1)                                         \
                 : "r"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))

#define HMMA16816(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1, RC2, RC3)                                                    \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" \
                 : "=r"(RD0), "=r"(RD1), "=r"(RD2), "=r"(RD3)                                                                              \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1), "r"(RC2), "r"(RC3))

#define HMMA1688(RD0, RD1, RD2, RD3, RA0, RA1, RB0, RC0, RC1, RC2, RC3)                                                    \
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n" \
                 : "=r"(RD0), "=r"(RD1), "=r"(RD2), "=r"(RD3)                                                                              \
                 : "r"(RA0), "r"(RA1), "r"(RB0), "r"(RC0), "r"(RC1), "r"(RC2), "r"(RC3))

__global__ void ana_qk_forward_kernel_C16(float *__restrict__ attn,
                            const uint128_t *__restrict__ query,
                            const uint128_t *__restrict__ key,
                            const int N,
                            const int T,
                            const int C,
                            const int win_size,
                            const int left,
                            const int tile_size,
                            const int right) {
    extern __shared__ uint128_t shares8[];
    const int len = left+tile_size+right;
    float *shares_attn = (float *)(shares8+len*2);

    const int part = min(tile_size, T-blockIdx.y*tile_size);

    //const int row = threadIdx.x/4;
    //const int col = threadIdx.x%4*2;
    const int t_tile = threadIdx.y*WMMA_M + threadIdx.x/4;
    const int t = blockIdx.y*tile_size+t_tile;

    #pragma unroll
    for(int c=0; c<C; c+=WMMA_K){
        uint32_t RA[4];
        {
            if(threadIdx.y*WMMA_M+threadIdx.x/2 < part){
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(__cvta_generic_to_shared(&shares8[threadIdx.y*WMMA_M+threadIdx.x/2+threadIdx.x%2*len])), "l"(&query[((blockIdx.x*T+blockIdx.y*tile_size + threadIdx.y*WMMA_M+threadIdx.x/2)*C+c)/8+threadIdx.x%2]));
            }
            asm volatile("cp.async.wait_all;\n" ::);
            __syncthreads();
            
            /*uint32_t *shares2 = (uint32_t *)shares8;
            if(t<T){
                RA[0] = shares2[t_tile*4+threadIdx.x%4];
                RA[2] = shares2[(len+t_tile)*4+threadIdx.x%4];
            }else{
                RA[0] = 0;
                RA[2] = 0;
            }
            if(t+8<T){
                RA[1] = shares2[(t_tile+8)*4+threadIdx.x%4];
                RA[3] = shares2[(len+t_tile+8)*4+threadIdx.x%4];
            }else{
                RA[1] = 0;
                RA[3] = 0;
            }*/
            uint32_t A_shmem_lane_addr = __cvta_generic_to_shared(&shares8[threadIdx.y*WMMA_M + threadIdx.x%16 + threadIdx.x/16*len]);
            LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_shmem_lane_addr);
            __syncthreads();
        }

        {
            if(threadIdx.y*WMMA_M+threadIdx.x/2 < part){
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(__cvta_generic_to_shared(&shares8[left+threadIdx.y*WMMA_M+threadIdx.x/2+threadIdx.x%2*len])), "l"(&key[((blockIdx.x*T+blockIdx.y*tile_size + threadIdx.y*WMMA_M+threadIdx.x/2)*C+c)/8+threadIdx.x%2]));
            }

            const int id = 32*threadIdx.y+threadIdx.x;
            if(id<left*2){
                if(blockIdx.y*tile_size>=left-id/2){
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(__cvta_generic_to_shared(&shares8[id/2+id%2*len])), "l"(&key[((blockIdx.x*T+blockIdx.y*tile_size-left+id/2)*C+c)/8+id%2]));
                }else{
                    shares8[id/2+id%2*len] = 0;
                }
            }else if(id+right*2>=blockDim.x*blockDim.y){
                const int index = id + right*2 - blockDim.x*blockDim.y;
                if((blockIdx.y+1)*tile_size+index/2 < T){
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(__cvta_generic_to_shared(&shares8[left+part+index/2+index%2*len])), "l"(&key[((blockIdx.x*T+(blockIdx.y+1)*tile_size+index/2)*C+c)/8+id%2]));
                }else{
                    shares8[left+part+index/2+index%2*len] = 0;
                }
            }
            
            asm volatile("cp.async.wait_all;\n" ::);
            __syncthreads();
        }

        uint32_t RB[2];
        //#pragma unroll
        for(int i=0; i<ceil((float)win_size/WMMA_N)+2; i++){//+2 is WMMA_M/WMMA_N
            uint32_t RAcc[4] = {0, 0, 0, 0};
            if(threadIdx.x<16){
                uint32_t B_shmem_lane_addr = __cvta_generic_to_shared(&shares8[threadIdx.y*WMMA_M+i*WMMA_N + threadIdx.x%8 + threadIdx.x/8*len]);
                LDMATRIX_X2(RB[0], RB[1], B_shmem_lane_addr);
            }
            HMMA16816(RAcc[0], RAcc[1], RAcc[2], RAcc[3], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RAcc[0], RAcc[1], RAcc[2], RAcc[3]);
            __syncthreads();
            
            float *acc = (float *)RAcc;
            const int k = i*WMMA_N + threadIdx.x%4*2 - threadIdx.x/4;
            if(c==0){
                //有bank conflicts
                if(t<T){
                    if(k>=0 && k<win_size){ shares_attn[t_tile*win_size+k] = acc[0]; }
                    if(k>=-1 && k+1<win_size){ shares_attn[t_tile*win_size+k+1] = acc[1]; }
                }
                if(t+8<T){
                    if(k>=8 && k-8<win_size){ shares_attn[(t_tile+8)*win_size+k-8] = acc[2]; }
                    if(k>=7 && k-7<win_size){ shares_attn[(t_tile+8)*win_size+k-7] = acc[3]; }
                }
            }else{
                if(t<T){
                    if(k>=0 && k<win_size){ shares_attn[t_tile*win_size+k] += acc[0]; }
                    if(k>=-1 && k+1<win_size){ shares_attn[t_tile*win_size+k+1] += acc[1]; }
                }
                if(t+8<T){
                    if(k>=8 && k-8<win_size){ shares_attn[(t_tile+8)*win_size+k-8] += acc[2]; }
                    if(k>=7 && k-7<win_size){ shares_attn[(t_tile+8)*win_size+k-7] += acc[3]; }
                }
            }
            //__syncthreads();
        }
    }
    __syncthreads();
    
    const int ntk = blockIdx.x*T*win_size + blockIdx.y*tile_size*win_size;
    #pragma unroll
    for(int j=0; j<ceil((float)tile_size*win_size/32/blockDim.y); j++){
        const int index = blockDim.y*j*32 + 32*threadIdx.y + threadIdx.x;
        if(index<part*win_size){ attn[ntk+index] = shares_attn[index/win_size*win_size + index%win_size]; }
    }
    
}

__global__ void ana_qk_forward_kernel_C8(float *__restrict__ attn,
                            const uint128_t *__restrict__ query,
                            const uint128_t *__restrict__ key,
                            const int N,
                            const int T,
                            const int C,
                            const int win_size,
                            const int left,
                            const int tile_size,
                            const int right) {
    extern __shared__ uint128_t shares8[];
    float *shares_attn = (float *)(shares8+left+tile_size+right);

    const int part = min(tile_size, T-blockIdx.y*tile_size);

    //const int row = threadIdx.x/4;
    //const int col = threadIdx.x%4*2;
    const int t_tile = threadIdx.y*WMMA_M + threadIdx.x/4;
    const int t = blockIdx.y*tile_size+t_tile;

    #pragma unroll
    for(int c=0; c<C; c+=WMMA2_K){
        uint32_t RA[2];
        {
            if(threadIdx.y*WMMA_M+threadIdx.x/2 < part){
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(__cvta_generic_to_shared(&shares8[threadIdx.y*WMMA_M+threadIdx.x/2])), "l"(&query[((blockIdx.x*T+blockIdx.y*tile_size + threadIdx.y*WMMA_M+threadIdx.x/2)*C+c)/8]));
            }
            asm volatile("cp.async.wait_all;\n" ::);
            __syncthreads();
            
            /*uint32_t *shares2 = (uint32_t *)shares8;
            if(t<T){
                RA[0] = shares2[t_tile*4+threadIdx.x%4];
            }else{
                RA[0] = 0;
            }
            if(t+8<T){
                RA[1] = shares2[(t_tile+8)*4+threadIdx.x%4];
            }else{
                RA[1] = 0;
            }*/

            uint32_t A_shmem_lane_addr = __cvta_generic_to_shared(&shares8[threadIdx.y*WMMA_M + threadIdx.x%16]);
            LDMATRIX_X2(RA[0], RA[1], A_shmem_lane_addr);
            __syncthreads();
        }

        {
            if(threadIdx.x<16 && threadIdx.y*WMMA_M+threadIdx.x < part){
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(__cvta_generic_to_shared(&shares8[left+threadIdx.y*WMMA_M+threadIdx.x])), "l"(&key[((blockIdx.x*T+blockIdx.y*tile_size + threadIdx.y*WMMA_M+threadIdx.x)*C+c)/8]));
            }

            const int id = 32*threadIdx.y+threadIdx.x;
            if(id<left){
                if(blockIdx.y*tile_size>=left-id){
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(__cvta_generic_to_shared(&shares8[id])), "l"(&key[((blockIdx.x*T+blockIdx.y*tile_size-left+id)*C+c)/8]));
                }else{
                    shares8[id] = 0;
                }
            }else if(id+right>=blockDim.x*blockDim.y){
                const int index = id + right - blockDim.x*blockDim.y;
                if((blockIdx.y+1)*tile_size+index < T){
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(__cvta_generic_to_shared(&shares8[left+part+index])), "l"(&key[((blockIdx.x*T+(blockIdx.y+1)*tile_size+index)*C+c)/8]));
                }else{
                    shares8[left+part+index] = 0;
                }
            }
            
            asm volatile("cp.async.wait_all;\n" ::);
            __syncthreads();
        }

        uint32_t RB[1];
        //#pragma unroll
        for(int i=0; i<ceil((float)win_size/WMMA_N)+2; i++){//+2 is WMMA_M/WMMA_N
            uint32_t RAcc[4] = {0, 0, 0, 0};
            if(threadIdx.x<8){
                uint32_t B_shmem_lane_addr = __cvta_generic_to_shared(&shares8[threadIdx.y*WMMA_M+i*WMMA_N + threadIdx.x]);
                LDMATRIX_X1(RB[0], B_shmem_lane_addr);
            }
            HMMA1688(RAcc[0], RAcc[1], RAcc[2], RAcc[3], RA[0], RA[1], RB[0], RAcc[0], RAcc[1], RAcc[2], RAcc[3]);
            __syncthreads();
            
            float *acc = (float *)RAcc;
            const int k = i*WMMA_N + threadIdx.x%4*2 - threadIdx.x/4;
            if(c==0){
                //有bank conflicts
                if(t<T){
                    if(k>=0 && k<win_size){ shares_attn[t_tile*win_size+k] = acc[0]; }
                    if(k>=-1 && k+1<win_size){ shares_attn[t_tile*win_size+k+1] = acc[1]; }
                }
                if(t+8<T){
                    if(k>=8 && k-8<win_size){ shares_attn[(t_tile+8)*win_size+k-8] = acc[2]; }
                    if(k>=7 && k-7<win_size){ shares_attn[(t_tile+8)*win_size+k-7] = acc[3]; }
                }
            }else{
                if(t<T){
                    if(k>=0 && k<win_size){ shares_attn[t_tile*win_size+k] += acc[0]; }
                    if(k>=-1 && k+1<win_size){ shares_attn[t_tile*win_size+k+1] += acc[1]; }
                }
                if(t+8<T){
                    if(k>=8 && k-8<win_size){ shares_attn[(t_tile+8)*win_size+k-8] += acc[2]; }
                    if(k>=7 && k-7<win_size){ shares_attn[(t_tile+8)*win_size+k-7] += acc[3]; }
                }
            }
            //__syncthreads();
        }
    }
    __syncthreads();
    
    const int ntk = blockIdx.x*T*win_size + blockIdx.y*tile_size*win_size;
    #pragma unroll
    for(int j=0; j<ceil((float)tile_size*win_size/32/blockDim.y); j++){
        const int index = blockDim.y*j*32 + 32*threadIdx.y + threadIdx.x;
        if(index<part*win_size){ attn[ntk+index] = shares_attn[index/win_size*win_size + index%win_size]; }
    }
    
}


__global__ void ana_q_backward_kernel(__nv_bfloat162 *__restrict__ query_gd,
                            const float *__restrict__ attn_gd,
                            const uint128_t *__restrict__ key,
                            const int N,
                            const int T,
                            const int C,
                            const int win_size,
                            const int left,
                            const int tile_size,
                            const int right) {
    extern __shared__ uint128_t shares8[];

    __nv_bfloat16 *shares = (__nv_bfloat16 *)shares8;
    __nv_bfloat16 *shares_attn_gd = (__nv_bfloat16 *)(shares8+(left+tile_size+right));
    const int part = min(tile_size, T-blockIdx.y*tile_size);
    {
        const int ntk = (blockIdx.x*T + blockIdx.y*tile_size)*win_size;
        #pragma unroll
        for(int j=0; j<ceil((float)tile_size*win_size/32/blockDim.y); j++){
            const int index = blockDim.y*j*32 + 32*threadIdx.y + threadIdx.x;
            if(index<part*win_size){ shares_attn_gd[index/win_size*win_size + index%win_size] = __float2bfloat16(attn_gd[ntk+index]); }
        }
        //__syncthreads();
    }

    const int row = threadIdx.x/4;
    const int col = threadIdx.x%4*2;
    const int t_tile = threadIdx.y*WMMA_M + row;
    const int t = blockIdx.y*tile_size+t_tile;

    #pragma unroll
    for(int c=0; c<C; c+=WMMA_N){
        {
            /*
            const int t1 = threadIdx.y*WMMA_M+threadIdx.x;///8*4+threadIdx.x%2*4+threadIdx.x/2;
            if(threadIdx.x<16 && t1 < part){
                //if(blockIdx.x==0 && blockIdx.y==0 && c==0){printf("[%d, %d, %d]\n", threadIdx.x, threadIdx.y, t1);}
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(__cvta_generic_to_shared(&shares8[left+t1])), "l"(&key[((blockIdx.x*T+blockIdx.y*tile_size + t1)*C+c)/8]));
                //shares8[left+t1] = key[((blockIdx.x*T+blockIdx.y*tile_size + t1)*C+c)/8];
            }*/

            const int id = threadIdx.y*32+threadIdx.x;
            if(id < part){
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(__cvta_generic_to_shared(&shares8[left+id])), "l"(&key[((blockIdx.x*T+blockIdx.y*tile_size + id)*C+c)/8]));
                //shares8[left+id] = key[((blockIdx.x*T+blockIdx.y*tile_size + id)*C+c)/8];
            }

            if(id<left){
                if(blockIdx.y*tile_size>=left-id){
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(__cvta_generic_to_shared(&shares8[id])), "l"(&key[((blockIdx.x*T+blockIdx.y*tile_size-left+id)*C+c)/8]));
                }else{
                    shares8[id] = 0;
                }
            }else if(id+right>=blockDim.x*blockDim.y){
                const int index = id + right - blockDim.x*blockDim.y;
                if((blockIdx.y+1)*tile_size+index < T){
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(__cvta_generic_to_shared(&shares8[left+part+index])), "l"(&key[((blockIdx.x*T+(blockIdx.y+1)*tile_size+index)*C+c)/8]));
                }else{
                    shares8[left+part+index] = 0;
                }
            }
            
            asm volatile("cp.async.wait_all;\n" ::);
            __syncthreads();
        }

        uint32_t RAcc[4] = {0, 0, 0, 0};
        //#pragma unroll
        for(int i=0; i<ceil((float)win_size/WMMA_K)+1; i++){//+1 is WMMA_M/WMMA_K
            uint32_t RA[4] = {0,0,0,0};
            {
                __nv_bfloat16 *RA_bf16 = (__nv_bfloat16 *)RA;
                const int k = i*WMMA_K + col - row;
                if(t<T){
                    if( k>=0 && k<win_size ){ RA_bf16[0]=shares_attn_gd[t_tile*win_size+k]; }
                    if( k>=-1 && k+1<win_size ){ RA_bf16[1]=shares_attn_gd[t_tile*win_size+k+1]; }
                    if( k>=-8 && k+8<win_size ){ RA_bf16[4]=shares_attn_gd[t_tile*win_size+k+8]; }
                    if( k>=-9 && k+9<win_size ){ RA_bf16[5]=shares_attn_gd[t_tile*win_size+k+9]; }
                }
                if(t+8<T){
                    if( k>=8 && k-8<win_size ){ RA_bf16[2]=shares_attn_gd[(t_tile+8)*win_size+k-8]; }
                    if( k>=7 && k-7<win_size ){ RA_bf16[3]=shares_attn_gd[(t_tile+8)*win_size+k-7]; }
                    if( k>=0 && k<win_size ){ RA_bf16[6]=shares_attn_gd[(t_tile+8)*win_size+k];}
                    if( k>=-1 && k+1<win_size ){ RA_bf16[7]=shares_attn_gd[(t_tile+8)*win_size+k+1]; }
                }
            }
            uint32_t RB[2] = {0,0};
            {
                __nv_bfloat16 *RB_bf16 = (__nv_bfloat16 *)RB;
                const int t1 = threadIdx.y*WMMA_M+i*WMMA_K+col;
                const int t2 = t1 -(left+part+right);
                if( t2<0 ){ RB_bf16[0]=shares[t1*8+row]; }
                if( t2<-1 ){ RB_bf16[1]=shares[(t1+1)*8+row]; }
                if( t2<-8 ){ RB_bf16[2]=shares[(t1+8)*8+row]; }
                if( t2<-9 ){ RB_bf16[3]=shares[(t1+9)*8+row]; }
            }
            
            HMMA16816(RAcc[0], RAcc[1], RAcc[2], RAcc[3], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RAcc[0], RAcc[1], RAcc[2], RAcc[3]);
            __syncthreads();
        }
        //__syncthreads();

        float2 *acc = (float2 *)RAcc;
        if( t<T ){ query_gd[((blockIdx.x*T+blockIdx.y*tile_size+t_tile)*C + c+col)/2] = __float22bfloat162_rn(acc[0]); }
        if( t+8<T ){ query_gd[((blockIdx.x*T+blockIdx.y*tile_size+t_tile+8)*C + c+col)/2] = __float22bfloat162_rn(acc[1]); }
    }
}


__global__ void ana_k_backward_kernel(__nv_bfloat162 *__restrict__ key_gd,
                            const float *__restrict__ attn_gd,
                            const uint128_t *__restrict__ query,
                            const int N,
                            const int T,
                            const int C,
                            const int win_size,
                            const int left,
                            const int tile_size,
                            const int right) {
    extern __shared__ uint128_t shares8[];
    __nv_bfloat16 *shares = (__nv_bfloat16 *)shares8;
    __nv_bfloat16 *shares_attn_gd = (__nv_bfloat16 *)(shares8+(left+tile_size+right));
    const int part = min(tile_size, T-blockIdx.y*tile_size);
    {
        const int start0 = (blockIdx.y*tile_size - left)*win_size;
        const int start = blockIdx.x*T*win_size + start0;
        #pragma unroll
        for(int j=0; j<ceil((float)(part+win_size-1)*win_size/32/blockDim.y); j++){
            const int index = blockDim.y*j*32 + 32*threadIdx.y + threadIdx.x;
            if(start0+index>=0 && start0+index<T*win_size){ shares_attn_gd[index/win_size*win_size + index%win_size] = __float2bfloat16(attn_gd[start+index]); }
        }
        //__syncthreads();
    }

    const int row = threadIdx.x/4;
    const int col = threadIdx.x%4*2;
    const int t_tile = threadIdx.y*WMMA_M + row;
    const int t = blockIdx.y*tile_size+t_tile;

    #pragma unroll
    for(int c=0; c<C; c+=WMMA_N){
        {
            const int id = threadIdx.y*32+threadIdx.x;
            if(id < part){
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(__cvta_generic_to_shared(&shares8[left+id])), "l"(&query[((blockIdx.x*T+blockIdx.y*tile_size + id)*C+c)/8]));
            }

            if(id<left){
                if(blockIdx.y*tile_size>=left-id){
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(__cvta_generic_to_shared(&shares8[id])), "l"(&query[((blockIdx.x*T+blockIdx.y*tile_size-left+id)*C+c)/8]));
                }else{
                    shares8[id] = 0;
                }
            }else if(id+right>=blockDim.x*blockDim.y){
                const int index = id + right - blockDim.x*blockDim.y;
                if((blockIdx.y+1)*tile_size+index < T){
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(__cvta_generic_to_shared(&shares8[left+part+index])), "l"(&query[((blockIdx.x*T+(blockIdx.y+1)*tile_size+index)*C+c)/8]));
                }else{
                    shares8[left+part+index] = 0;
                }
            }
            
            asm volatile("cp.async.wait_all;\n" ::);
            __syncthreads();
        }

        uint32_t RAcc[4] = {0, 0, 0, 0};
        //#pragma unroll
        for(int i=0; i<ceil((float)win_size/WMMA_K)+1; i++){
            uint32_t RA[4] = {0,0,0,0};
            {
                __nv_bfloat16 *RA_bf16 = (__nv_bfloat16 *)RA;
                const int k = win_size-1 - i*WMMA_K - col + row;
                const int t_tile1 = threadIdx.y*WMMA_M + i*WMMA_K + col;
                const int t1 = blockIdx.y*tile_size + t_tile1 - left;
                if( t1>=0 && t1<T && k>=0 && k<win_size ){ RA_bf16[0]=shares_attn_gd[t_tile1*win_size+k]; }
                if( t1>=-1 && t1+1<T && k>=1 && k-1<win_size ){ RA_bf16[1]=shares_attn_gd[(t_tile1+1)*win_size+k-1]; }
                if( t1>=-8 && t1+8<T && k>=8 && k-8<win_size ){ RA_bf16[4]=shares_attn_gd[(t_tile1+8)*win_size+k-8]; }
                if( t1>=-9 && t1+9<T && k>=9 && k-9<win_size ){ RA_bf16[5]=shares_attn_gd[(t_tile1+9)*win_size+k-9]; }
                
                if( t1>=0 && t1<T && k>=-8 && k+8<win_size ){ RA_bf16[2]=shares_attn_gd[t_tile1*win_size+k+8]; }
                if( t1>=-1 && t1+1<T && k>=-7 && k+7<win_size ){ RA_bf16[3]=shares_attn_gd[(t_tile1+1)*win_size+k+7]; }
                if( t1>=-8 && t1+8<T && k>=0 && k<win_size ){ RA_bf16[6]=shares_attn_gd[(t_tile1+8)*win_size+k]; }
                if( t1>=-9 && t1+9<T && k>=1 && k-1<win_size ){ RA_bf16[7]=shares_attn_gd[(t_tile1+9)*win_size+k-1]; }
            }

            uint32_t RB[2] = {0,0};
            {
                __nv_bfloat16 *RB_bf16 = (__nv_bfloat16 *)RB;
                const int t1 = threadIdx.y*WMMA_M + i*WMMA_K + col;
                const int t2 = t1 -(left+part+right);
                if( t2<0 ){ RB_bf16[0]=shares[t1*8+row]; }
                if( t2<-1 ){ RB_bf16[1]=shares[(t1+1)*8+row]; }
                if( t2<-8 ){ RB_bf16[2]=shares[(t1+8)*8+row]; }
                if( t2<-9 ){ RB_bf16[3]=shares[(t1+9)*8+row]; }
            }
            
            HMMA16816(RAcc[0], RAcc[1], RAcc[2], RAcc[3], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RAcc[0], RAcc[1], RAcc[2], RAcc[3]);
            __syncthreads();
        }
        //__syncthreads();

        float2 *acc = (float2 *)RAcc;
        if( t<T ){ key_gd[((blockIdx.x*T+blockIdx.y*tile_size+t_tile)*C + c+col)/2] = __float22bfloat162_rn(acc[0]); }
        if( t+8<T ){ key_gd[((blockIdx.x*T+blockIdx.y*tile_size+t_tile+8)*C + c+col)/2] = __float22bfloat162_rn(acc[1]); }        
    }

}


void bf16_ana_qk_forward_C16(float *attn,
                 const __nv_bfloat16 *query,
                 const __nv_bfloat16 *key,
                 const int N,
                 const int T,
                 const int C,
                 const int win_size,
                 const int left) {
    const int need_wrap = ceil((float)T/WMMA_M);
    const int wrap_num1 = need_wrap > MAX_wrap_num ? ceil((float)need_wrap/ceil((float)need_wrap/MAX_wrap_num)) : need_wrap;
    const int wrap_num2 = (12240-win_size*8) / (8+win_size) / WMMA_M;
    const int wrap_num = wrap_num1<=wrap_num2 ? wrap_num1 : wrap_num1/ceil((float)wrap_num1/wrap_num2);
    const int tile_size = min(T, wrap_num*WMMA_M);
    //printf("01[need_wrap=%d, wrap_num1=%d, wrap_num2=%d, wrap_num=%d, tile_size=%d, blockIdx.y=%d]\n", need_wrap, wrap_num1, wrap_num2, wrap_num, tile_size, (int)ceil((float)T/tile_size));
    dim3 gridSize(N, ceil((float)T/tile_size));
    dim3 blockSize(WARP_SIZE, wrap_num);

    const int len1 = ceil((float)(tile_size+left+win_size-left-1+4)/8)*8-4; // =ceil((float)(tile_size+left+win_size-left-1-4)/8)*8+4 //for key->shares_key without bankconflicts
    const int size = len1*WMMA_K*sizeof(__nv_bfloat16) + tile_size*win_size*sizeof(float);
    //printf("\t[49152-size=%d]\n", 49152-size);
    ana_qk_forward_kernel_C16<<<gridSize, blockSize, size>>>(attn, (uint128_t *)query, (uint128_t *)key, N, T, C, win_size, left, tile_size, len1-left-tile_size);
}

void bf16_ana_qk_forward_C8(float *attn,
                 const __nv_bfloat16 *query,
                 const __nv_bfloat16 *key,
                 const int N,
                 const int T,
                 const int C,
                 const int win_size,
                 const int left) {
    const int need_wrap = ceil((float)T/WMMA_M);
    const int wrap_num1 = need_wrap > MAX_wrap_num ? ceil((float)need_wrap/ceil((float)need_wrap/MAX_wrap_num)) : need_wrap;
    const int wrap_num2 = (12280-win_size*4) / (4+win_size) / WMMA_M;
    const int wrap_num = wrap_num1<=wrap_num2 ? wrap_num1 : wrap_num1/ceil((float)wrap_num1/wrap_num2);
    const int tile_size = min(T, wrap_num*WMMA_M);
    //printf("01[need_wrap=%d, wrap_num1=%d, wrap_num2=%d, wrap_num=%d, tile_size=%d, blockIdx.y=%d]\n", need_wrap, wrap_num1, wrap_num2, wrap_num, tile_size, (int)ceil((float)T/tile_size));
    dim3 gridSize(N, ceil((float)T/tile_size));
    dim3 blockSize(WARP_SIZE, wrap_num);

    const int size = (tile_size+win_size-1)*WMMA2_K*sizeof(__nv_bfloat16) + tile_size*win_size*sizeof(float);
    //printf("\t[49152-size=%d]\n", 49152-size);
    ana_qk_forward_kernel_C8<<<gridSize, blockSize, size>>>(attn, (uint128_t *)query, (uint128_t *)key, N, T, C, win_size, left, tile_size, win_size-left-1);
}

void bf16_ana_q_backward(__nv_bfloat16 *query_gd,
                 const float *attn_gd,
                 const __nv_bfloat16 *key,
                 const int N,
                 const int T,
                 const int C,
                 const int win_size,
                 const int left) {
    const int need_wrap = ceil((float)T/WMMA_M);
    const int wrap_num1 = need_wrap > MAX_wrap_num ? ceil((float)need_wrap/ceil((float)need_wrap/MAX_wrap_num)) : need_wrap;
    const int wrap_num2 = (24584-win_size*8) / (8+win_size) / WMMA_M;
    const int wrap_num = wrap_num1<=wrap_num2 ? wrap_num1 : wrap_num1/ceil((float)wrap_num1/wrap_num2);
    const int tile_size = min(T, wrap_num*WMMA_M);
    dim3 gridSize(N, ceil((float)T/tile_size));
    dim3 blockSize(WARP_SIZE, wrap_num);

    const int len1 = tile_size + win_size-1;
    const int size = len1*8*sizeof(__nv_bfloat16) + tile_size*win_size*sizeof(__nv_bfloat16);
    ana_q_backward_kernel<<<gridSize, blockSize, size>>>((__nv_bfloat162 *)query_gd, attn_gd, (uint128_t *)key, N, T, C, win_size, left, tile_size, win_size-left-1);
}

void bf16_ana_k_backward(__nv_bfloat16 *key_gd,
                 const float *attn_gd,
                 const __nv_bfloat16 *query,
                 const int N,
                 const int T,
                 const int C,
                 const int win_size,
                 const int left) {
    const int need_wrap = ceil((float)T/WMMA_M);
    const int wrap_num1 = need_wrap > MAX_wrap_num ? ceil((float)need_wrap/ceil((float)need_wrap/MAX_wrap_num)) : need_wrap;
    const int wrap_num2 = ( 24576 / (8+win_size) - win_size+1 ) / WMMA_M;
    const int wrap_num = wrap_num1<=wrap_num2 ? wrap_num1 : wrap_num1/ceil((float)wrap_num1/wrap_num2);
    const int tile_size = min(T, wrap_num*WMMA_M);
    dim3 gridSize(N, ceil((float)T/tile_size));
    dim3 blockSize(WARP_SIZE, wrap_num);

    const int len1 = tile_size + win_size-1;
    const int size = len1*8*sizeof(__nv_bfloat16) + len1*win_size*sizeof(__nv_bfloat16);
    ana_k_backward_kernel<<<gridSize, blockSize, size>>>((__nv_bfloat162 *)key_gd, attn_gd, (uint128_t *)query, N, T, C, win_size, win_size-left-1, tile_size, left);
}