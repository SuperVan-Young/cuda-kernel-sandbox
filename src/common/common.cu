#include "common/common.cuh"

namespace cks { namespace common {

retCode_t runKernel(int kernel, int version, KernelArgs* args) {
    switch (kernel) {
        case KER_SGEMM: return cks::sgemm::runKernel(version, (cks::sgemm::SgemmArgs *)args);
        default: return RC_ERROR;
    }
}

bool verifyKernel(int kernel, int version, KernelArgs *args) {
    switch (kernel) {
        case KER_SGEMM: return cks::sgemm::verifyKernel(version, (cks::sgemm::SgemmArgs *)args);
        default: return false;
    }
}

double speedTestKernel(int kernel, int version, KernelArgs *args) {
    switch (kernel) {
        case KER_SGEMM: return cks::sgemm::speedTestKernel(version, (cks::sgemm::SgemmArgs *)args);
        default: return 0.0;
    }
}

DataLoader *createDataLoader(int kernel) {
    DataLoader *p = nullptr;
    switch (kernel) {
        case KER_SGEMM: p = new cks::sgemm::SgemmDataLoader(); break;
        default: break;
    }
    return p;  // should delete p in the end!
}

}}  // namespace cks::common