#include "common/common.cuh"

namespace cks { namespace common {

float runKernel(int kernel, int version, KernelArgs* args) {
    switch (kernel) {
        case KER_SGEMM: return cks::sgemm::runKernel(version, (cks::sgemm::SgemmArgs *)args);
        default: return 0.0;
    }
}

bool verifyKernel(int kernel, int version, DataLoader *dataloader) {
    KernelArgs *p_test, *p_valid;
    CKS_CALL(dataloader->loadData(&p_test));
    CKS_CALL(dataloader->deepcopyData(&p_valid, p_test));

    runKernel(kernel, 0, p_valid);
    runKernel(kernel, version, p_test);

    bool valid = dataloader->equalResult(p_valid, p_test);
    
    CKS_CALL(dataloader->freeData(p_test));
    CKS_CALL(dataloader->freeData(p_valid));

    return valid;
}

DataLoader *createDataLoader(int kernel) {
    DataLoader *p = nullptr;
    switch (kernel) {
        case KER_SGEMM: p = new cks::sgemm::SgemmDataLoader(); break;
        default: break;
    }
    return p;  // should delete p in the end!
}

void destroyDataLoader(DataLoader *p) {
    delete p;
    return;
}

}}  // namespace cks::common