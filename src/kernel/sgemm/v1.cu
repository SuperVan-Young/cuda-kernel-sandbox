#include "kernel/sgemm/cuBLAS.cuh"

#define A(i,j) A[(i) + (j) * lda]
#define B(i,j) B[(i) + (j) * ldb]
#define C(i,j) C[(i) + (j) * ldc]

namespace cks {namespace sgemm{

static __global__
void kernelFunc(int M, int N, int K, const float *alpha, const float *beta, 
               const float *A, const float *B, float *C) {
    return;
}

cks::common::retCode_t sgemmKernel_v1(SgemmArgs *args) {

    //TODO: finish this kernel
    return cks::common::RC_SUCCESS;
}

}};  // namespace cks::sgemm