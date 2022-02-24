#include "kernel/sgemm.cuh"

typedef cks::common::retCode_t retCode_t;

namespace cks {namespace sgemm{

void sgemmKernel_cuBLAS(int M, int N, int K, const float *h_alpha, const float *h_beta, 
                        const float *d_A, const float *d_B, float *d_C) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, 
                h_alpha, d_A, M, d_B, K, h_beta, d_C, M);
    cublasDestroy(handle);
}

}};  // namespace cks::sgemm