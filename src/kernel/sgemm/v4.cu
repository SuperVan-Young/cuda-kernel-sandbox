#include "kernel/sgemm.cuh"

/**
 * This time we follow CUTLASS' code.
 * First, we add configurable block tiling.
 * A block of MS * NS in C is calculated with step KS
 * Here we introduce addition combination law, which affects the result accuracy
 */

// this defines a workload of a block
#define MS 64   // cannot assign more shared memory
#define NS 64  
#define KS 8

// this defines the workload of a warp
#define MW 32  
#define NW 64  // (MW/MR) * (NW/NR) == 32, number of threads in a warp

// this defines the workload of a thread
#define MR 8  // should be 2, 4 or 8
#define NR 8  // one thread should calculate MR * NR elements in C

#define WARPSIZE 32
#define NUM_THREADS (WARPSIZE*MS*NS/MW/NW)

#define DIV_CEIL(i, j) (((i) + (j) - 1) / (j))

#define A_block(i,j) A_block[(i) + (j) * M]
#define B_block(i,j) B_block[(i) + (j) * K]
#define C_block(i,j) C_block[(i) + (j) * M]
#define s_A(i,j) s_A[(i) + (j) * MS]
#define s_B(i,j) s_B[(i) + (j) * KS]
#define s_C(i,j) s_C[(i) + (j) * MS]
#define A_fragment(wx, i) A_fragment[4*(wx) + (i)]
#define B_fragment(wy, i) B_fragment[(4*(wy) + (i))*KS]
#define C_fragment(wx, wy, i, j) C_fragment[(4*(wx)+(i)) + (4*(wy)+(j))*MS]

namespace cks {namespace sgemm{

static __global__
void kernelFunc(int M, int N, int K, const float *alpha, const float *beta, 
               const float *A, const float *B, float *C) {
    // calculate Block's address
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int cx = tid % MS;
    int cy = tid / MS;
    int wx = threadIdx.x % (MW/MR);  // x position in a warp
    int wy = threadIdx.x / (MW/MR);  // y position in a warp

    const float *A_block = A + bx * MS;                // &A[MS*bx][0]
    const float *B_block = B + by * NS * K;            // &B[0][NS*by]
    float       *C_block = C + bx * MS + by * NS * K;  // &C[MS*bx][NS*by]

    __shared__ float s_A[MS*KS];  // could change reference pattern easily
    __shared__ float s_B[KS*NS];  // on 1-d array with macro

    __shared__ float s_C[MS*NS];  // prevent referencing global memory
    for (int bk = 0; bk < NS; bk += KS) {
        s_C(cx, cy+bk) = 0;  // initialize s_C
    }
    __syncthreads();

    float a0, a1, a2, a3;
    float b0, b1, b2, b3;
    float *C_fragment = &s_C(threadIdx.y*MW, threadIdx.z*NW);


    // step with KS (ignore circumstances K % KS != 0)
    for (int bk = 0; bk < K; bk += KS) {
        // copy thread block to shared memory
        for (int sk = 0; sk < MS*KS; sk += NUM_THREADS) {
            int tid_shift = tid+sk;  // shift by number of threads
            s_A[tid_shift] = A_block(tid_shift%MS, tid_shift/MS);  // column-wise
            s_B[tid_shift] = B_block(tid_shift%KS, tid_shift/KS);  // column-wise
        }
        __syncthreads();

        // for a naive kernel, we calculate 4 register tiles once at a time
        for (int wx_offset = 0; wx_offset < MW; wx_offset += 2*MR) {  // step with 4 * (MR / 2)
            for (int wy_offset = 0; wy_offset < NW; wy_offset += 4*NR) {  // step with 8 * (NR / 2)
                float *A_fragment = &s_A(threadIdx.y*MW, 0);
                float *B_fragment = &s_B(0, threadIdx.z*NW);
                
                float sum00 = 0;
                float sum01 = 0;
                float sum02 = 0;
                float sum03 = 0;
                float sum10 = 0;
                float sum11 = 0;
                float sum12 = 0;
                float sum13 = 0;
                float sum20 = 0;
                float sum21 = 0;
                float sum22 = 0;
                float sum23 = 0;
                float sum30 = 0;
                float sum31 = 0;
                float sum32 = 0;
                float sum33 = 0;

                // step through KS
                for (int l = 0; l < KS; l++) {

                    // copy fragment to registers
                    a0 = A_fragment(wx, 0+wx_offset);
                    a1 = A_fragment(wx, 1+wx_offset);
                    a2 = A_fragment(wx, 2+wx_offset);
                    a3 = A_fragment(wx, 3+wx_offset);

                    b0 = B_fragment(wy, 0+wy_offset);
                    b1 = B_fragment(wy, 1+wy_offset);
                    b2 = B_fragment(wy, 2+wy_offset);
                    b3 = B_fragment(wy, 3+wy_offset);

                    // calculate partial sum
                    sum00 += a0 * b0;
                    sum01 += a0 * b1;
                    sum02 += a0 * b2;
                    sum03 += a0 * b3;
                    sum10 += a1 * b0;
                    sum11 += a1 * b1;
                    sum12 += a1 * b2;
                    sum13 += a1 * b3;
                    sum20 += a2 * b0;
                    sum21 += a2 * b1;
                    sum22 += a2 * b2;
                    sum23 += a2 * b3;
                    sum30 += a3 * b0;
                    sum31 += a3 * b1;
                    sum32 += a3 * b2;
                    sum33 += a3 * b3;

                    // update fragment address
                    A_fragment += MS;
                    B_fragment += 1;

                }

                // add back to s_C
                C_fragment(wx, wy, 0+wx_offset, 0+wy_offset) += sum00;
                C_fragment(wx, wy, 0+wx_offset, 1+wy_offset) += sum01;
                C_fragment(wx, wy, 0+wx_offset, 2+wy_offset) += sum02;
                C_fragment(wx, wy, 0+wx_offset, 3+wy_offset) += sum03;
                C_fragment(wx, wy, 1+wx_offset, 0+wy_offset) += sum10;
                C_fragment(wx, wy, 1+wx_offset, 1+wy_offset) += sum11;
                C_fragment(wx, wy, 1+wx_offset, 2+wy_offset) += sum12;
                C_fragment(wx, wy, 1+wx_offset, 3+wy_offset) += sum13;
                C_fragment(wx, wy, 2+wx_offset, 0+wy_offset) += sum20;
                C_fragment(wx, wy, 2+wx_offset, 1+wy_offset) += sum21;
                C_fragment(wx, wy, 2+wx_offset, 2+wy_offset) += sum22;
                C_fragment(wx, wy, 2+wx_offset, 3+wy_offset) += sum23;
                C_fragment(wx, wy, 3+wx_offset, 0+wy_offset) += sum30;
                C_fragment(wx, wy, 3+wx_offset, 1+wy_offset) += sum31;
                C_fragment(wx, wy, 3+wx_offset, 2+wy_offset) += sum32;
                C_fragment(wx, wy, 3+wx_offset, 3+wy_offset) += sum33;

                __syncthreads();

            }  // loop: wy_offset
        }  // loop: wx_offset

        // update thread block's address
        A_block += KS * M;
        B_block += KS;
    }  // loop bk

    // add back to C
    for (int sk = 0; sk < NS; sk += NUM_THREADS/MS) {
        C_block(cx, cy+sk) = *alpha * s_C(cx, cy+sk) + *beta * C_block(cx, cy+sk);
    }
    __syncthreads();

    return;

}

void sgemmKernel_v4(int M, int N, int K, const float *d_alpha, const float *d_beta, 
                    const float *d_A, const float *d_B, float *d_C) {
    int num_blocks_x = DIV_CEIL(M, MS);
    int num_blocks_y = DIV_CEIL(N, NS);
    dim3 grid_size(num_blocks_x, num_blocks_y);
    dim3 block_size(32, MS/MW, NS/NW);

    kernelFunc<<<grid_size, block_size>>>(M, N, K, d_alpha, d_beta, d_A, d_B, d_C);
    
}

}};  // namespace cks::sgemm