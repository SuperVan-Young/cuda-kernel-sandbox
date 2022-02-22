#include "data/sgemm.cuh"

#include <random>

namespace cks { namespace sgemm{

static void randomInitArray(float *array, int length) {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-10.0, 10.0);
    for (int i = 0; i < length; i++) {
        array[i] = distribution(generator);
    }
}

cks::common::retCode_t SgemmArgs::deepcopy(cks::common::KernelArgs **p_target) {
    int length_A = M*K;
    int length_B = K*N;
    int length_C = M*N;
    float *A_ = (float *)malloc(sizeof(float)*length_A);
    float *B_ = (float *)malloc(sizeof(float)*length_B);
    float *C_ = (float *)malloc(sizeof(float)*length_C);
    memcpy(A_, A, sizeof(float)*length_A);
    memcpy(B_, A, sizeof(float)*length_B);
    memcpy(C_, A, sizeof(float)*length_C);

    *p_target = new SgemmArgs(M, N, K, alpha, beta, A_, B_, C_);
    return cks::common::RC_SUCCESS;
}

cks::common::retCode_t SgemmArgs::destroyCopy(cks::common::KernelArgs *p_target) {
    SgemmArgs *p_argscopy = (SgemmArgs*)p_target;
    free(p_argscopy->A);
    free(p_argscopy->B);
    free(p_argscopy->C);
    delete p_argscopy;
    return cks::common::RC_SUCCESS;
}

bool SgemmArgs::equalResult(cks::common::KernelArgs *p_target) {
    int length_C = M*N;
    SgemmArgs *p_argscopy = (SgemmArgs*)p_target;

    if (length_C != ((p_argscopy->M) * (p_argscopy->N)))
        return false;
    float eps = 1e-6;
    for (int i = 0; i < length_C; i++) {
        if (abs(C[i] - p_argscopy->C[i]) > eps)
            return false;
    }
    return true;
}

cks::common::retCode_t SgemmDataLoader::loadData(cks::common::KernelArgs **p_data) {
    int length = width * width;

    float *A = (float *)malloc(sizeof(float)*length);
    float *B = (float *)malloc(sizeof(float)*length);
    float *C = (float *)malloc(sizeof(float)*length);

    randomInitArray(A, length);
    randomInitArray(B, length);
    memset(C, 0, sizeof(C));

    SgemmArgs *p_args = new SgemmArgs(width, width, width, &alpha, &beta, A, B, C);
    *p_data = (cks::common::KernelArgs *)p_args;

    return cks::common::RC_SUCCESS;
}

cks::common::retCode_t SgemmDataLoader::freeData(cks::common::KernelArgs *p_data) {
    SgemmArgs *p_args = (SgemmArgs *)p_data;
    
    float *A = p_args->A;
    float *B = p_args->B;
    float *C = p_args->C;
    free(A);
    free(B);
    free(C);

    return cks::common::RC_SUCCESS;
}

int SgemmDataLoader::len() {
    return num;
}

cks::common::retCode_t SgemmDataLoader::step() {
    width += inc;
    return cks::common::RC_SUCCESS;
}

cks::common::retCode_t SgemmDataLoader::log(float perf) {
    //TODO: save performance to log file, after speedTest
    return cks::common::RC_SUCCESS;
}


}}  // namespace cks::sgemm