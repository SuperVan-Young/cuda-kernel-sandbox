#ifndef CKS_DATA_SGEMM_CUH_
#define CKS_DATA_SGEMM_CUH_

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

}}  // namespace cks::sgemm

#endif