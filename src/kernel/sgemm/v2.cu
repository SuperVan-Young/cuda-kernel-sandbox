#include "kernel/sgemm.cuh"

/**
 * Use shared memory to coalesce memory reference to B.
 * Read a 32 * 32 tile from A and B into shared memory
 * 
 * Transposing s_B could, theoretically, lead to more coalesced memory refernce,
 * thus boost the speed. But the experiment result fails to support this idea.
 * The following setting is the best I could get.
 */

#define KERNEL_SIZE 32
#define DIV(i) (((i) + KERNEL_SIZE - 1) / KERNEL_SIZE)

#define A(i,j) A[(i) + (j) * lda]
#define B(i,j) B[(i) + (j) * ldb]
#define C(i,j) C[(i) + (j) * ldc]
#define s_A(i,j) s_A[(i) + (j) * KERNEL_SIZE]
#define s_B(i,j) s_B[(i) + (j) * KERNEL_SIZE]  // try switching i and j

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
    int k;
    int n = DIV(K);  // number of tiles on A and B
    float sum = 0;

    __shared__ float s_A[KERNEL_SIZE*KERNEL_SIZE];
    __shared__ float s_B[KERNEL_SIZE*KERNEL_SIZE];

    if (i >= M || j >= N)
        return;

    for (int tile = 0; tile < n; tile++) {
        // copy data to shared memory
        k = tile * KERNEL_SIZE + ty;
        s_A(tx, ty) = k < K ? A(i, k) : 0;

        k = tile * KERNEL_SIZE + tx;
        s_B(tx, ty) = k < K ? B(k, j) : 0;
        __syncthreads();

        for (int l = 0; l < KERNEL_SIZE; l++) {
            sum += s_A(tx, l) * s_B(l, ty);
        }
        __syncthreads();  // this is vital!
    }
    C(i, j) = *alpha * sum + *beta * C(i, j);

    return;

}

void sgemmKernel_v2(int M, int N, int K, const float *d_alpha, const float *d_beta, 
                    const float *d_A, const float *d_B, float *d_C) {

    int div_M = DIV(M);
    int div_N = DIV(N);
    dim3 grid_size(div_M, div_N);
    dim3 block_size(KERNEL_SIZE, KERNEL_SIZE);

    kernelFunc<<<grid_size, block_size>>>(M, N, K, d_alpha, d_beta, d_A, d_B, d_C);
    
}

}};  // namespace cks::sgemm