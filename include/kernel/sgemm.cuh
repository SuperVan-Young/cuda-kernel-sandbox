#ifndef _CKS_KERNEL_SGEMM_CUH_
#define _CKS_KERNEL_SGEMM_CUH_

using namespace std;
namespace cks { namespace sgemm {

class SGEMMArgs : public cks::common::KernelArgs {
public:
    int M;
    int N;
    int K;
    const double *alpha;
    const double *A;
    const double *B;
    const double *beta;
    double *C;
};

void runKernel(int version, KernelArgs* args);

bool verifyKernel(int version, KernelArgs* args);

double speedTestKernel(int version, KernelArgs* args);

}}  // namespace cks::sgemm

#endif