#include "kernel/sgemm.cuh"

namespace cks { namespace sgemm {

cks::common::retCode_t runKernel(int version, SgemmArgs *args) {    
    switch (version) {
        case 0:  return sgemmKernel_cuBLAS(args);
        case 1:  return sgemmKernel_v1(args);
        default: return cks::common::RC_ERROR;
    }
}

}}  // namespace cks::sgemm