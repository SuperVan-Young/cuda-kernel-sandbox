#ifndef CKS_KERNEL_SGEMM_CUBLAS_CUH_
#define CKS_KERNEL_SGEMM_CUBLAS_CUH_

#include "common/dtype.cuh"
#include "common/error.cuh"

typedef cks::common::retCode_t retCode_t;

namespace cks {namespace sgemm{

retCode_t sgemmKernel_cuBLAS(SgemmArgs *args);

}};  // namespace cks::sgemm


#endif