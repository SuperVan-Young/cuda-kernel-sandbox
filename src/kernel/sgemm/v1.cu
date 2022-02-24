#include "kernel/sgemm.cuh"

#define KERNEL_SIZE 32
#define DIV(i) (((i) + KERNEL_SIZE - 1) / KERNEL_SIZE)

#define A(i,j) A[(i) + (j) * lda]
#define B(i,j) B[(i) + (j) * ldb]
#define C(i,j) C[(i) + (j) * ldc]

namespace cks {namespace sgemm{

static __global__
void kernelFunc(int M, int N, int K, const float *alpha, const float *beta, 
               const float *A, const float *B, float *C) {
    int lda = M;
    int ldb = K;
    int ldc = M;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = bx * KERNEL_SIZE + tx;
    int j = by * KERNEL_SIZE + ty;
    if (i >= M || j >= N)
        return;

    float sum = 0;
    for (int k = 0; k < K; k++) {
        sum += A(i, k) * B(k, j);
    }
    C(i, j) = *alpha * sum + *beta * C(i, j);

    return;

}

void sgemmKernel_v1(int M, int N, int K, const float *d_alpha, const float *d_beta, 
                    const float *d_A, const float *d_B, float *d_C) {

    int div_M = DIV(M);
    int div_N = DIV(N);
    dim3 grid_size(div_M, div_N);
    dim3 block_size(KERNEL_SIZE, KERNEL_SIZE);

    kernelFunc<<<grid_size, block_size>>>(M, N, K, d_alpha, d_beta, d_A, d_B, d_C);
    
}

}};  // namespace cks::sgemm