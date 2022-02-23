#ifndef CKS_KERNEL_SGEMM_CUBLAS_CUH_
#define CKS_KERNEL_SGEMM_CUBLAS_CUH_

#include "common/dtype.cuh"
#include "common/error.cuh"

#include "data/sgemm.cuh"

#include <cublas_v2.h>

using cks::common::retCode_t;

namespace cks {namespace sgemm{

retCode_t sgemmKernel_cuBLAS(cks::sgemm::SgemmArgs *args);

}}  // namespace cks::sgemm


#endif