#include "kernel/sgemm.cuh"

namespace cks { namespace sgemm {

cks::common::retCode_t runKernel(int version, SgemmArgs *args) {
    return cks::common::RC_SUCCESS;
    
    //TODO: finish this part
    switch (version) {
        case 0:  return sgemmKernel_cuBLAS(args);
        case 1:  return sgemmKernel_v1(args);
        default: return cks::common::RC_ERROR;
    }
}

bool verifyKernel(int version, SgemmArgs *args) {
    return false;
}

double speedTestKernel(int version, SgemmArgs *args) {
    return 0.0;
}

}}  // namespace cks::sgemm