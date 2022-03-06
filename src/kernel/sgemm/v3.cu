#include "kernel/sgemm.cuh"

/**
 * Notice that s_B(l, ty) could be reused for every thread with Idx y
 * To reuse s_B(l,ty), we introduce a microkernel to compute a 4*1 tile
 * For convenience, we only support matrix width divisible by `KERNEL_SIZE`
 * 
 * Use float4 might accelerate a little bit, I leave it for future modification.
 */

#define KERNEL_SIZE 32
#define MICROKERNEL_SIZE 4
#define DIV(i) (((i) + KERNEL_SIZE - 1) / KERNEL_SIZE)

#define A_(i,j) A_[(i) + (j) * lda]
#define B_(i,j) B_[(i) + (j) * ldb]
#define C_(i,j) C_[(i) + (j) * ldc]
#define s_A(i,j) s_A[(i) + (j) * KERNEL_SIZE]
#define s_B(i,j) s_B[(i) + (j) * KERNEL_SIZE]

namespace cks {namespace sgemm{

static __global__
void kernelFunc(int M, int N, int K, const float *alpha, const float *beta, 
               const float *A, const float *B, float *C) {
    int lda = M;
    int ldb = K;
    int ldc = M;

    int tilex = threadIdx.x * MICROKERNEL_SIZE;
    int tiley = threadIdx.y;
    int n = DIV(K);  // number of tiles on A and B
    float sum0 = 0;
    float sum1 = 0;
    float sum2 = 0;
    float sum3 = 0;
    const float *A_ = A + blockIdx.x * KERNEL_SIZE;
    const float *B_ = B + blockIdx.y * KERNEL_SIZE * K;

    __shared__ float s_A[KERNEL_SIZE*KERNEL_SIZE];
    __shared__ float s_B[KERNEL_SIZE*KERNEL_SIZE];

    for (int tile = 0; tile < n; tile++) {
        // copy tile of A and B to s_A and s_B
        for (int i = 0; i < MICROKERNEL_SIZE; i++) {
            s_A(tilex + i, tiley) = A_(tilex + i, tiley);
            s_B(tilex + i, tiley) = B_(tilex + i, tiley);
        }
        __syncthreads();
        
        // compute partial sum in a 4x1 manner
        for (int l = 0; l < KERNEL_SIZE; l++) {
            sum0 += s_A(tilex + 0, l) * s_B(l, tiley);
            sum1 += s_A(tilex + 1, l) * s_B(l, tiley);
            sum2 += s_A(tilex + 2, l) * s_B(l, tiley);
            sum3 += s_A(tilex + 3, l) * s_B(l, tiley);
        }
        // adjust A_ and B_
        A_ = A_ + KERNEL_SIZE * M;
        B_ = B_ + KERNEL_SIZE;
        __syncthreads();
    }

    float *C_ = C + blockIdx.x * KERNEL_SIZE + blockIdx.y * KERNEL_SIZE * M;
    float alpha_ = *alpha;
    float beta_ = *beta;
    C_(tilex + 0, tiley) = alpha_ * sum0 + beta_ * C_(tilex + 0, tiley); 
    C_(tilex + 1, tiley) = alpha_ * sum1 + beta_ * C_(tilex + 1, tiley); 
    C_(tilex + 2, tiley) = alpha_ * sum2 + beta_ * C_(tilex + 2, tiley); 
    C_(tilex + 3, tiley) = alpha_ * sum3 + beta_ * C_(tilex + 3, tiley); 

    return;

}

void sgemmKernel_v3(int M, int N, int K, const float *d_alpha, const float *d_beta, 
                    const float *d_A, const float *d_B, float *d_C) {

    int div_M = DIV(M);
    int div_N = DIV(N);
    dim3 grid_size(div_M, div_N);
    dim3 block_size(KERNEL_SIZE/MICROKERNEL_SIZE, KERNEL_SIZE);

    kernelFunc<<<grid_size, block_size>>>(M, N, K, d_alpha, d_beta, d_A, d_B, d_C);
    
}

}};  // namespace cks::sgemm