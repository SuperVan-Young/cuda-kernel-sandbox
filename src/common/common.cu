#include "common/common.cuh"

namespace cks { namespace common {

retCode_t runKernel(int kernel, int version, KernelArgs* args) {
    switch (kernel) {
        case KER_SGEMM: return cks::sgemm::runKernel(version, (cks::sgemm::SgemmArgs *)args);
        default: return RC_ERROR;
    }
}

bool verifyKernel(int kernel, int version, KernelArgs *args) {
    KernelArgs *args_copy = nullptr;

    CKS_CALL(args->deepcopy(&args_copy));

    CKS_CALL(runKernel(kernel, version, args));
    CKS_CALL(runKernel(kernel, version, args_copy));

    retCode_t ret;
    if (args->equalResult(args_copy))
        ret = RC_SUCCESS;
    else
        ret = RC_ERROR;
    
    CKS_CALL(args->destroyCopy(args_copy));
    return ret;
}

float speedTestKernel(int kernel, int version, KernelArgs *args) {
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    CUDA_CALL(cudaEventRecord(start));
    cudaEventQuery(start);

    CKS_CALL(runKernel(kernel, version, args));

    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));
    float elapsed_time;
    CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));

    return elapsed_time;
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