#include "kernel/sgemm.cuh"

/**
 * This time we follow CUTLASS' code.
 * 
 * We use register to store partial sum, instead of using shared memory s_C.
 * This will prevent duplicated references to A and B's fragment
 * as well as to s_C.
 * 
 * Could use float4 and WMMA in later versions
 * Could change s_B's layout in later versions.
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
#define A_fragment(wx, i) A_fragment[4*(wx) + (i)]
#define B_fragment(wy, i) B_fragment[(4*(wy) + (i))*KS]
#define C_fragment(wx, wy, i, j) C_fragment[(4*(wx)+(i)) + (4*(wy)+(j))*M]

namespace cks {namespace sgemm{

static __global__
void kernelFunc(int M, int N, int K, const float *alpha, const float *beta, 
               const float *A, const float *B, float *C) {
    // calculate Block's address
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int wx = threadIdx.x % (MW/MR);  // x position in a warp
    int wy = threadIdx.x / (MW/MR);  // y position in a warp

    const float *A_block = A + bx * MS;                // &A[MS*bx][0]
    const float *B_block = B + by * NS * K;            // &B[0][NS*by]
    float       *C_block = C + bx * MS + by * NS * K;  // &C[MS*bx][NS*by]

    __shared__ float s_A[MS*KS];  // could change reference pattern easily
    __shared__ float s_B[KS*NS];  // on 1-d array with macro

    float a00, a01, a02, a03;
    float a10, a11, a12, a13;
    float b00, b01, b02, b03;
    float b10, b11, b12, b13;
    // the first index indicates the position of a 4x4 register tile
    // 0 for upper-left, 1 for lower-left, 2 for upper-right, 3 for lower-right
    // the second and third index are used inside the register tile
    float sum000 = 0, sum010 = 0, sum020 = 0, sum030 = 0;
    float sum001 = 0, sum011 = 0, sum021 = 0, sum031 = 0;
    float sum002 = 0, sum012 = 0, sum022 = 0, sum032 = 0;
    float sum003 = 0, sum013 = 0, sum023 = 0, sum033 = 0;
    float sum100 = 0, sum110 = 0, sum120 = 0, sum130 = 0;
    float sum101 = 0, sum111 = 0, sum121 = 0, sum131 = 0;
    float sum102 = 0, sum112 = 0, sum122 = 0, sum132 = 0;
    float sum103 = 0, sum113 = 0, sum123 = 0, sum133 = 0;
    float sum200 = 0, sum210 = 0, sum220 = 0, sum230 = 0;
    float sum201 = 0, sum211 = 0, sum221 = 0, sum231 = 0;
    float sum202 = 0, sum212 = 0, sum222 = 0, sum232 = 0;
    float sum203 = 0, sum213 = 0, sum223 = 0, sum233 = 0;
    float sum300 = 0, sum310 = 0, sum320 = 0, sum330 = 0;
    float sum301 = 0, sum311 = 0, sum321 = 0, sum331 = 0;
    float sum302 = 0, sum312 = 0, sum322 = 0, sum332 = 0;
    float sum303 = 0, sum313 = 0, sum323 = 0, sum333 = 0;

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
        float *A_fragment = &s_A(threadIdx.y*MW, 0);
        float *B_fragment = &s_B(0, threadIdx.z*NW);

        // step through KS
        for (int l = 0; l < KS; l++) {
            int wx_offset = 0;  // = 0 or 2*MR
            int wy_offset = 0;  // = 0 or 4*NR

            // calculate the 0th tile's partial sum
            // copy in A's first fragment and B's fragment
            a00 = A_fragment(wx, 0+wx_offset);
            a01 = A_fragment(wx, 1+wx_offset);
            a02 = A_fragment(wx, 2+wx_offset);
            a03 = A_fragment(wx, 3+wx_offset);

            b00 = B_fragment(wy, 0+wy_offset);
            b01 = B_fragment(wy, 1+wy_offset);
            b02 = B_fragment(wy, 2+wy_offset);
            b03 = B_fragment(wy, 3+wy_offset);

            sum000 += a00 * b00;
            sum001 += a00 * b01;
            sum002 += a00 * b02;
            sum003 += a00 * b03;
            sum010 += a01 * b00;
            sum011 += a01 * b01;
            sum012 += a01 * b02;
            sum013 += a01 * b03;
            sum020 += a02 * b00;
            sum021 += a02 * b01;
            sum022 += a02 * b02;
            sum023 += a02 * b03;
            sum030 += a03 * b00;
            sum031 += a03 * b01;
            sum032 += a03 * b02;
            sum033 += a03 * b03;

            // calculate the 2nd tile's partial sum
            // reuse A's first fragment and copy in B's second fragment
            wy_offset = 4 * NR;
            b10 = B_fragment(wy, 0+wy_offset);
            b11 = B_fragment(wy, 1+wy_offset);
            b12 = B_fragment(wy, 2+wy_offset);
            b13 = B_fragment(wy, 3+wy_offset);

            sum200 += a00 * b10;
            sum201 += a00 * b11;
            sum202 += a00 * b12;
            sum203 += a00 * b13;
            sum210 += a01 * b10;
            sum211 += a01 * b11;
            sum212 += a01 * b12;
            sum213 += a01 * b13;
            sum220 += a02 * b10;
            sum221 += a02 * b11;
            sum222 += a02 * b12;
            sum223 += a02 * b13;
            sum230 += a03 * b10;
            sum231 += a03 * b11;
            sum232 += a03 * b12;
            sum233 += a03 * b13;

            // calculate the 3rd tile's partial sum
            // reuse B's second fragment and copy in A's second fragment
            wx_offset = 2 * MR;
            a10 = A_fragment(wx, 0+wx_offset);
            a11 = A_fragment(wx, 1+wx_offset);
            a12 = A_fragment(wx, 2+wx_offset);
            a13 = A_fragment(wx, 3+wx_offset);

            sum300 += a10 * b10;
            sum301 += a10 * b11;
            sum302 += a10 * b12;
            sum303 += a10 * b13;
            sum310 += a11 * b10;
            sum311 += a11 * b11;
            sum312 += a11 * b12;
            sum313 += a11 * b13;
            sum320 += a12 * b10;
            sum321 += a12 * b11;
            sum322 += a12 * b12;
            sum323 += a12 * b13;
            sum330 += a13 * b10;
            sum331 += a13 * b11;
            sum332 += a13 * b12;
            sum333 += a13 * b13;

            // calculate the 1st tile's partial sum
            // reuse A's second fragment and reuse B's first fragment

            sum100 += a10 * b00;
            sum101 += a10 * b01;
            sum102 += a10 * b02;
            sum103 += a10 * b03;
            sum110 += a11 * b00;
            sum111 += a11 * b01;
            sum112 += a11 * b02;
            sum113 += a11 * b03;
            sum120 += a12 * b00;
            sum121 += a12 * b01;
            sum122 += a12 * b02;
            sum123 += a12 * b03;
            sum130 += a13 * b00;
            sum131 += a13 * b01;
            sum132 += a13 * b02;
            sum133 += a13 * b03;

            // update fragment address
            A_fragment += MS;
            B_fragment += 1;
        }

        // update thread block's address
        A_block += KS * M;
        B_block += KS;
    }  // loop bk

    // add back to C
    float *C_fragment = &C_block(threadIdx.y*MW, threadIdx.z*NW);
    C_fragment(wx, wy, 0, 0)           = *alpha * C_fragment(wx, wy, 0, 0)            + *beta * sum000;
    C_fragment(wx, wy, 0, 1)           = *alpha * C_fragment(wx, wy, 0, 1)            + *beta * sum001;
    C_fragment(wx, wy, 0, 2)           = *alpha * C_fragment(wx, wy, 0, 2)            + *beta * sum002;
    C_fragment(wx, wy, 0, 3)           = *alpha * C_fragment(wx, wy, 0, 3)            + *beta * sum003;
    C_fragment(wx, wy, 1, 0)           = *alpha * C_fragment(wx, wy, 1, 0)            + *beta * sum010;
    C_fragment(wx, wy, 1, 1)           = *alpha * C_fragment(wx, wy, 1, 1)            + *beta * sum011;
    C_fragment(wx, wy, 1, 2)           = *alpha * C_fragment(wx, wy, 1, 2)            + *beta * sum012;
    C_fragment(wx, wy, 1, 3)           = *alpha * C_fragment(wx, wy, 1, 3)            + *beta * sum013;
    C_fragment(wx, wy, 2, 0)           = *alpha * C_fragment(wx, wy, 2, 0)            + *beta * sum020;
    C_fragment(wx, wy, 2, 1)           = *alpha * C_fragment(wx, wy, 2, 1)            + *beta * sum021;
    C_fragment(wx, wy, 2, 2)           = *alpha * C_fragment(wx, wy, 2, 2)            + *beta * sum022;
    C_fragment(wx, wy, 2, 3)           = *alpha * C_fragment(wx, wy, 2, 3)            + *beta * sum023;
    C_fragment(wx, wy, 3, 0)           = *alpha * C_fragment(wx, wy, 3, 0)            + *beta * sum030;
    C_fragment(wx, wy, 3, 1)           = *alpha * C_fragment(wx, wy, 3, 1)            + *beta * sum031;
    C_fragment(wx, wy, 3, 2)           = *alpha * C_fragment(wx, wy, 3, 2)            + *beta * sum032;
    C_fragment(wx, wy, 3, 3)           = *alpha * C_fragment(wx, wy, 3, 3)            + *beta * sum033;
    C_fragment(wx, wy, 0+2*MR, 0)      = *alpha * C_fragment(wx, wy, 0+2*MR, 0)       + *beta * sum100;
    C_fragment(wx, wy, 0+2*MR, 1)      = *alpha * C_fragment(wx, wy, 0+2*MR, 1)       + *beta * sum101;
    C_fragment(wx, wy, 0+2*MR, 2)      = *alpha * C_fragment(wx, wy, 0+2*MR, 2)       + *beta * sum102;
    C_fragment(wx, wy, 0+2*MR, 3)      = *alpha * C_fragment(wx, wy, 0+2*MR, 3)       + *beta * sum103;
    C_fragment(wx, wy, 1+2*MR, 0)      = *alpha * C_fragment(wx, wy, 1+2*MR, 0)       + *beta * sum110;
    C_fragment(wx, wy, 1+2*MR, 1)      = *alpha * C_fragment(wx, wy, 1+2*MR, 1)       + *beta * sum111;
    C_fragment(wx, wy, 1+2*MR, 2)      = *alpha * C_fragment(wx, wy, 1+2*MR, 2)       + *beta * sum112;
    C_fragment(wx, wy, 1+2*MR, 3)      = *alpha * C_fragment(wx, wy, 1+2*MR, 3)       + *beta * sum113;
    C_fragment(wx, wy, 2+2*MR, 0)      = *alpha * C_fragment(wx, wy, 2+2*MR, 0)       + *beta * sum120;
    C_fragment(wx, wy, 2+2*MR, 1)      = *alpha * C_fragment(wx, wy, 2+2*MR, 1)       + *beta * sum121;
    C_fragment(wx, wy, 2+2*MR, 2)      = *alpha * C_fragment(wx, wy, 2+2*MR, 2)       + *beta * sum122;
    C_fragment(wx, wy, 2+2*MR, 3)      = *alpha * C_fragment(wx, wy, 2+2*MR, 3)       + *beta * sum123;
    C_fragment(wx, wy, 3+2*MR, 0)      = *alpha * C_fragment(wx, wy, 3+2*MR, 0)       + *beta * sum130;
    C_fragment(wx, wy, 3+2*MR, 1)      = *alpha * C_fragment(wx, wy, 3+2*MR, 1)       + *beta * sum131;
    C_fragment(wx, wy, 3+2*MR, 2)      = *alpha * C_fragment(wx, wy, 3+2*MR, 2)       + *beta * sum132;
    C_fragment(wx, wy, 3+2*MR, 3)      = *alpha * C_fragment(wx, wy, 3+2*MR, 3)       + *beta * sum133;
    C_fragment(wx, wy, 0, 0+4*NR)      = *alpha * C_fragment(wx, wy, 0, 0+4*NR)       + *beta * sum200;
    C_fragment(wx, wy, 0, 1+4*NR)      = *alpha * C_fragment(wx, wy, 0, 1+4*NR)       + *beta * sum201;
    C_fragment(wx, wy, 0, 2+4*NR)      = *alpha * C_fragment(wx, wy, 0, 2+4*NR)       + *beta * sum202;
    C_fragment(wx, wy, 0, 3+4*NR)      = *alpha * C_fragment(wx, wy, 0, 3+4*NR)       + *beta * sum203;
    C_fragment(wx, wy, 1, 0+4*NR)      = *alpha * C_fragment(wx, wy, 1, 0+4*NR)       + *beta * sum210;
    C_fragment(wx, wy, 1, 1+4*NR)      = *alpha * C_fragment(wx, wy, 1, 1+4*NR)       + *beta * sum211;
    C_fragment(wx, wy, 1, 2+4*NR)      = *alpha * C_fragment(wx, wy, 1, 2+4*NR)       + *beta * sum212;
    C_fragment(wx, wy, 1, 3+4*NR)      = *alpha * C_fragment(wx, wy, 1, 3+4*NR)       + *beta * sum213;
    C_fragment(wx, wy, 2, 0+4*NR)      = *alpha * C_fragment(wx, wy, 2, 0+4*NR)       + *beta * sum220;
    C_fragment(wx, wy, 2, 1+4*NR)      = *alpha * C_fragment(wx, wy, 2, 1+4*NR)       + *beta * sum221;
    C_fragment(wx, wy, 2, 2+4*NR)      = *alpha * C_fragment(wx, wy, 2, 2+4*NR)       + *beta * sum222;
    C_fragment(wx, wy, 2, 3+4*NR)      = *alpha * C_fragment(wx, wy, 2, 3+4*NR)       + *beta * sum223;
    C_fragment(wx, wy, 3, 0+4*NR)      = *alpha * C_fragment(wx, wy, 3, 0+4*NR)       + *beta * sum230;
    C_fragment(wx, wy, 3, 1+4*NR)      = *alpha * C_fragment(wx, wy, 3, 1+4*NR)       + *beta * sum231;
    C_fragment(wx, wy, 3, 2+4*NR)      = *alpha * C_fragment(wx, wy, 3, 2+4*NR)       + *beta * sum232;
    C_fragment(wx, wy, 3, 3+4*NR)      = *alpha * C_fragment(wx, wy, 3, 3+4*NR)       + *beta * sum233;
    C_fragment(wx, wy, 0+2*MR, 0+4*NR) = *alpha * C_fragment(wx, wy, 0+2*MR, 0+4*NR)  + *beta * sum300;
    C_fragment(wx, wy, 0+2*MR, 1+4*NR) = *alpha * C_fragment(wx, wy, 0+2*MR, 1+4*NR)  + *beta * sum301;
    C_fragment(wx, wy, 0+2*MR, 2+4*NR) = *alpha * C_fragment(wx, wy, 0+2*MR, 2+4*NR)  + *beta * sum302;
    C_fragment(wx, wy, 0+2*MR, 3+4*NR) = *alpha * C_fragment(wx, wy, 0+2*MR, 3+4*NR)  + *beta * sum303;
    C_fragment(wx, wy, 1+2*MR, 0+4*NR) = *alpha * C_fragment(wx, wy, 1+2*MR, 0+4*NR)  + *beta * sum310;
    C_fragment(wx, wy, 1+2*MR, 1+4*NR) = *alpha * C_fragment(wx, wy, 1+2*MR, 1+4*NR)  + *beta * sum311;
    C_fragment(wx, wy, 1+2*MR, 2+4*NR) = *alpha * C_fragment(wx, wy, 1+2*MR, 2+4*NR)  + *beta * sum312;
    C_fragment(wx, wy, 1+2*MR, 3+4*NR) = *alpha * C_fragment(wx, wy, 1+2*MR, 3+4*NR)  + *beta * sum313;
    C_fragment(wx, wy, 2+2*MR, 0+4*NR) = *alpha * C_fragment(wx, wy, 2+2*MR, 0+4*NR)  + *beta * sum320;
    C_fragment(wx, wy, 2+2*MR, 1+4*NR) = *alpha * C_fragment(wx, wy, 2+2*MR, 1+4*NR)  + *beta * sum321;
    C_fragment(wx, wy, 2+2*MR, 2+4*NR) = *alpha * C_fragment(wx, wy, 2+2*MR, 2+4*NR)  + *beta * sum322;
    C_fragment(wx, wy, 2+2*MR, 3+4*NR) = *alpha * C_fragment(wx, wy, 2+2*MR, 3+4*NR)  + *beta * sum323;
    C_fragment(wx, wy, 3+2*MR, 0+4*NR) = *alpha * C_fragment(wx, wy, 3+2*MR, 0+4*NR)  + *beta * sum330;
    C_fragment(wx, wy, 3+2*MR, 1+4*NR) = *alpha * C_fragment(wx, wy, 3+2*MR, 1+4*NR)  + *beta * sum331;
    C_fragment(wx, wy, 3+2*MR, 2+4*NR) = *alpha * C_fragment(wx, wy, 3+2*MR, 2+4*NR)  + *beta * sum332;
    C_fragment(wx, wy, 3+2*MR, 3+4*NR) = *alpha * C_fragment(wx, wy, 3+2*MR, 3+4*NR)  + *beta * sum333;
    
    return;

}

void sgemmKernel_v5(int M, int N, int K, const float *d_alpha, const float *d_beta, 
                    const float *d_A, const float *d_B, float *d_C) {
    int num_blocks_x = DIV_CEIL(M, MS);
    int num_blocks_y = DIV_CEIL(N, NS);
    dim3 grid_size(num_blocks_x, num_blocks_y);
    dim3 block_size(32, MS/MW, NS/NW);

    kernelFunc<<<grid_size, block_size>>>(M, N, K, d_alpha, d_beta, d_A, d_B, d_C);
    
}

}};  // namespace cks::sgemm