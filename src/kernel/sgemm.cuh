#ifndef CKS_KERNEL_SGEMM_CUH_
#define CKS_KERNEL_SGEMM_CUH_

#include "common/dtype.cuh"
#include "common/error.cuh"

namespace cks { namespace sgemm {

typedef cks::common::retCode_t retCode_t;

class SgemmArgs : public cks::common::KernelArgs {
public:
    int M;
    int N;
    int K;
    const float *alpha;
    const float *beta;
    const float *A;
    const float *B;
    float *C;

    SgemmArgs(int M_, int N_, int K_, const float *alpha_, const float *beta_, \
              const float *A_, const float *B_, const float *C_) {
        M = M_;
        N = N_;
        K = K_;
        alpha = alpha_;
        beta = beta_;
        A = A_;
        B = B_;
        C = C_;
    } 
};

retCode_t runKernel(int version, SgemmArgs *args) {
    switch (version) {
        case 0:  return sgemmKernel_cuBLAS(args);
        case 1:  return sgemmKernel_v1(args);
        default: return cks::common::RC_ERROR;
    }
}

bool verifyKernel(int version, SgemmArgs *args);

double speedTestKernel(int version, SgemmArgs *args);

}}  // namespace cks::sgemm

#endif