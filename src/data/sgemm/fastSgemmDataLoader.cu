#include "data/sgemm.cuh"

#include <random>

static const int MAXSIZE = 1 << 14;

namespace cks { namespace sgemm{

static void randomInitArray(float *array, int length) {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1.0, 1.0);
    for (int i = 0; i < length; i++) {
        array[i] = distribution(generator);
    }
}

FastSgemmDataLoader::FastSgemmDataLoader() {
    width = 1024;
    num = 10;
    inc = 1024;
    alpha = 1.0;
    beta = 1.0;
    const int max_length = MAXSIZE*MAXSIZE;
    A = (float *)malloc(sizeof(float)*max_length);
    B = (float *)malloc(sizeof(float)*max_length);
    C = (float *)malloc(sizeof(float)*max_length);
    C_ref = (float *)malloc(sizeof(float)*max_length);
    randomInitArray(A, max_length);
    randomInitArray(B, max_length);
    randomInitArray(C, max_length);
    memcpy(C_ref, C, sizeof(float)*max_length);
}

cks::common::retCode_t FastSgemmDataLoader::loadData(cks::common::KernelArgs **p_data) {

    SgemmArgs *p_args = new SgemmArgs(width, width, width, &alpha, &beta, A, B, C);
    *p_data = (cks::common::KernelArgs *)p_args;

    return cks::common::RC_SUCCESS;
}

cks::common::retCode_t FastSgemmDataLoader::freeData(cks::common::KernelArgs *p_data) {
    SgemmArgs *p_args = (SgemmArgs *)p_data;
    delete p_args;

    return cks::common::RC_SUCCESS;
}

int FastSgemmDataLoader::len() {
    return num;
}

cks::common::retCode_t FastSgemmDataLoader::step() {
    width += inc;
    return cks::common::RC_SUCCESS;
}

cks::common::retCode_t FastSgemmDataLoader::log(float elapsed_time) {
    float gflops = 2.*1e-6 * width * width * width / elapsed_time;
    printf("(%5d): %.5f ms    %f GFLOPS\n", width, elapsed_time, gflops);
    return cks::common::RC_SUCCESS;
}

cks::common::retCode_t
FastSgemmDataLoader::deepcopyData(cks::common::KernelArgs **p_dst, cks::common::KernelArgs *p_src) {
    SgemmArgs *p_src_ = (SgemmArgs*)p_src;
    int M = p_src_->M;
    int N = p_src_->N;
    int K = p_src_->K;
    memcpy(C_ref, C, sizeof(float)*M*N);

    *p_dst = (cks::common::KernelArgs *) new SgemmArgs(M, N, K, &alpha, &beta, A, B, C_ref);
    return cks::common::RC_SUCCESS;
}

bool FastSgemmDataLoader::equalResult(cks::common::KernelArgs *p_1, cks::common::KernelArgs *p_2) {
    SgemmArgs *p_1_ = (SgemmArgs*) p_1;
    SgemmArgs *p_2_ = (SgemmArgs*) p_2;

    int length_C_1 = p_1_->M * p_1_->N;
    int length_C_2 = p_2_->M * p_2_->N;

    float *C_1 = p_1_->C;
    float *C_2 = p_2_->C;

    if (length_C_1 != length_C_2)
        return false;
    float eps = 1e1;
    for (int i = 0; i < length_C_1; i++) {
        if (abs(C_1[i] - C_2[i]) > eps) {
            printf("|%f - %f| > %f\n", C_1[i], C_2[i], eps);
            return false;
        }
    }
    
    return true;
}

}}  // namespace cks::sgemm