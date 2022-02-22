#ifndef CKS_DATA_SGEMM_CUH_
#define CKS_DATA_SGEMM_CUH_

#include "common/dtype.cuh"
#include "common/error.cuh"

namespace cks { namespace sgemm {

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
                const float *A_, const float *B_, float *C_) {
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

class SgemmDataLoader : public cks::common::DataLoader {
public:
    cks::common::retCode_t loadData(cks::common::KernelArgs **p_data);
    cks::common::retCode_t freeData(cks::common::KernelArgs *p_data);
    int len();
    cks::common::retCode_t step();
    cks::common::retCode_t log();
};

}}  // namespace cks::sgemm

#endif