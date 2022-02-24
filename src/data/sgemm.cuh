#ifndef CKS_DATA_SGEMM_CUH_
#define CKS_DATA_SGEMM_CUH_

#include "common/dtype.cuh"
#include "common/error.cuh"

#include <malloc.h>

namespace cks { namespace sgemm {

class SgemmArgs : public cks::common::KernelArgs {
public:
    int M;
    int N;
    int K;
    float *alpha;
    float *beta;
    float *A;
    float *B;
    float *C;

    SgemmArgs(int M_, int N_, int K_, float *alpha_, float *beta_,
              float *A_, float *B_, float *C_) {
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
    int width, num, inc;
    float alpha, beta;
    SgemmDataLoader() {
        width = 1024;
        num = 10;
        inc = 1024;
        alpha = 1.0;
        beta = 1.0;
    }

    cks::common::retCode_t loadData(cks::common::KernelArgs **p_data);
    cks::common::retCode_t freeData(cks::common::KernelArgs *p_data);
    cks::common::retCode_t deepcopyData(cks::common::KernelArgs **p_dst, cks::common::KernelArgs *p_src);
    bool equalResult(cks::common::KernelArgs *p_1, cks::common::KernelArgs *p_2);
    int len();
    cks::common::retCode_t step();
    cks::common::retCode_t log(float perf);
};

}}  // namespace cks::sgemm

#endif