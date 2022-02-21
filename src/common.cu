#include "common.cuh"

using namespace std;
using namespace cks::common;

namespace cks {namespace common {

RetCode runKernel(int kernel, int version, KernelArgs* args) {
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

}}  // namespace cks common