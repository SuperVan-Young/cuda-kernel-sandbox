#ifndef CKS_COMMON_CUH_
#define CKS_COMMON_CUH_

#include "common/dtype.cuh"
#include "common/error.cuh"
#include "kernel/sgemm.cuh"

#include <stdio.h>

// kernel ID
#define KER_SGEMM   0

namespace cks { namespace common {

retCode_t runKernel(int kernel, int version, KernelArgs* args) {
    switch (kernel) {
        case KER_SGEMM: return cks::sgemm::runKernel(version, args);
        default: return RC_ERROR;
    }
}

bool verifyKernel(int kernel, int version, KernelArgs *args) {
    switch (kernel) {
        case KER_SGEMM: return cks::sgemm::verifyKernel(version, args)
        default: return false;
    }
}

double speedTestKernel(int kernel, int version, KernelArgs *args) {
    switch (kernel) {
        case KER_SGEMM: return cks::sgemm::speedTestKernel(version, args)
        default: return 0.0;
    }
}

}} // namespace cks::common

#endif