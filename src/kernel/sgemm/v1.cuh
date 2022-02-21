#ifndef CKS_KERNEL_SGEMM_v1_CUH_
#define CKS_KERNEL_SGEMM_v1_CUH_

#include "common/dtype.cuh"
#include "common/error.cuh"

typedef cks::common::retCode_t retCode_t;

namespace cks {namespace sgemm{

retCode_t sgemmKernel_v1(SgemmArgs *args);

}};  // namespace cks::sgemm


#endif