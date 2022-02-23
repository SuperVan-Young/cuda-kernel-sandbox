#include "kernel/sgemm/cuBLAS.cuh"

typedef cks::common::retCode_t retCode_t;

namespace cks {namespace sgemm{

retCode_t sgemmKernel_cuBLAS(SgemmArgs *args) {
    int M = args->M;
    int N = args->N;
    int K = args->K;
    float *alpha = args->alpha;
    float *beta = args->beta;

    int length_A = M * K;
    int length_B = K * N;
    int length_C = M * N;

    float *d_A, *d_B, *d_C;
    CUDA_CALL(cudaMalloc((void **)&d_A, sizeof(float)*length_A));
    CUDA_CALL(cudaMalloc((void **)&d_B, sizeof(float)*length_B));
    CUDA_CALL(cudaMalloc((void **)&d_C, sizeof(float)*length_C));

    cublasSetVector(length_A, sizeof(float), args->A, 1, d_A, 1);
    cublasSetVector(length_B, sizeof(float), args->B, 1, d_B, 1);
    cublasSetVector(length_C, sizeof(float), args->C, 1, d_C, 1);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, 
                alpha, d_A, M, d_B, K, beta, d_C, M);
    cublasDestroy(handle);

    cublasGetVector(length_C, sizeof(float), d_C, 1, args->C, 1);

    CUDA_CALL(cudaFree(d_A));
    CUDA_CALL(cudaFree(d_B));
    CUDA_CALL(cudaFree(d_C));

    return cks::common::RC_SUCCESS;
}

}};  // namespace cks::sgemm