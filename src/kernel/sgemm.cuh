#ifndef CKS_KERNEL_SGEMM_CUH_
#define CKS_KERNEL_SGEMM_CUH_

#include "common/dtype.cuh"
#include "common/error.cuh"
#include "data/sgemm.cuh"

#include "kernel/sgemm/cuBLAS.cuh"
#include "kernel/sgemm/v1.cuh"

namespace cks { namespace sgemm {

cks::common::retCode_t runKernel(int version, cks::sgemm::SgemmArgs *args);

}}  // namespace cks::sgemm

#endif