#include "kernel/sgemm/cuBLAS.cuh"

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

cks::common::retCode_t sgemmKernel_v1(SgemmArgs *args) {
    int M = args->M;
    int N = args->N;
    int K = args->K;
    float *h_alpha = args->alpha;
    float *h_beta = args->beta;
    float *h_A = args->A;
    float *h_B = args->B;
    float *h_C = args->C;

    int length_A = M * K;
    int length_B = K * N;
    int length_C = M * N;

    float *d_alpha, *d_beta, *d_A, *d_B, *d_C;
    CUDA_CALL(cudaMalloc((void **)&d_alpha, sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&d_beta, sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&d_A, sizeof(float)*length_A));
    CUDA_CALL(cudaMalloc((void **)&d_B, sizeof(float)*length_B));
    CUDA_CALL(cudaMalloc((void **)&d_C, sizeof(float)*length_C));

    CUDA_CALL(cudaMemcpy(d_alpha, h_alpha, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_beta, h_beta, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_A, h_A, sizeof(float)*length_A, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_B, h_B, sizeof(float)*length_B, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_C, h_C, sizeof(float)*length_C, cudaMemcpyHostToDevice));

    int div_M = DIV(M);
    int div_N = DIV(N);
    dim3 grid_size(div_M, div_N);
    dim3 block_size(KERNEL_SIZE, KERNEL_SIZE);

    kernelFunc<<<grid_size, block_size>>>(M, N, K, d_alpha, d_beta, d_A, d_B, d_C);

    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize()); // for writing more robust program

    CUDA_CALL(cudaMemcpy(h_C, d_C, sizeof(float)*length_C, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(d_alpha));
    CUDA_CALL(cudaFree(d_beta));
    CUDA_CALL(cudaFree(d_A));
    CUDA_CALL(cudaFree(d_B));
    CUDA_CALL(cudaFree(d_C));

    return cks::common::RC_SUCCESS;
}

}};  // namespace cks::sgemm