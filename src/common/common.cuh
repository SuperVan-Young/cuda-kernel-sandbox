#ifndef CKS_COMMON_CUH_
#define CKS_COMMON_CUH_

#include "common/dtype.cuh"
#include "common/error.cuh"

#include "data/sgemm.cuh"
#include "kernel/sgemm.cuh"

#include <stdio.h>

// kernel ID
#define KER_SGEMM   0

namespace cks { namespace common {

retCode_t runKernel(int kernel, int version, KernelArgs* args);

bool verifyKernel(int kernel, int version, KernelArgs *args);

double speedTestKernel(int kernel, int version, KernelArgs *args);

}} // namespace cks::common

#endif