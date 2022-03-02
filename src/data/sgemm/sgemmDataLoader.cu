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

SgemmDataLoader::SgemmDataLoader(){
    width = 1024;
    num = 8;
    inc = 1024;
    alpha = 1.0;
    beta = 1.0;
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

    delete p_args;

    return cks::common::RC_SUCCESS;
}

int SgemmDataLoader::len() {
    return num;
}

cks::common::retCode_t SgemmDataLoader::step() {
    width += inc;
    return cks::common::RC_SUCCESS;
}

cks::common::retCode_t SgemmDataLoader::log(float elapsed_time) {
    float gflops = 2.*1e-6 * width * width * width / elapsed_time;
    printf("(%5d): %.5f ms    %f GFLOPS\n", width, elapsed_time, gflops);
    return cks::common::RC_SUCCESS;
}

cks::common::retCode_t
SgemmDataLoader::deepcopyData(cks::common::KernelArgs **p_dst, cks::common::KernelArgs *p_src) {
    SgemmArgs *p_src_ = (SgemmArgs*)p_src;
    int M = p_src_->M;
    int N = p_src_->N;
    int K = p_src_->K;
    float *A = p_src_->A;
    float *B = p_src_->B;
    float *C = p_src_->C;
    float *alpha = p_src_->alpha;
    float *beta = p_src_->beta;

    int length_A = M*K;
    int length_B = K*N;
    int length_C = M*N;
    float *A_ = (float *)malloc(sizeof(float)*length_A);
    float *B_ = (float *)malloc(sizeof(float)*length_B);
    float *C_ = (float *)malloc(sizeof(float)*length_C);
    memcpy(A_, A, sizeof(float)*length_A);
    memcpy(B_, B, sizeof(float)*length_B);
    memcpy(C_, C, sizeof(float)*length_C);

    *p_dst = (cks::common::KernelArgs *) new SgemmArgs(M, N, K, alpha, beta, A_, B_, C_);
    return cks::common::RC_SUCCESS;
}

bool SgemmDataLoader::equalResult(cks::common::KernelArgs *p_1, cks::common::KernelArgs *p_2) {
    SgemmArgs *p_1_ = (SgemmArgs*) p_1;
    SgemmArgs *p_2_ = (SgemmArgs*) p_2;

    int length_C_1 = p_1_->M * p_1_->N;
    int length_C_2 = p_2_->M * p_2_->N;

    float *C_1 = p_1_->C;
    float *C_2 = p_2_->C;

    if (length_C_1 != length_C_2)
        return false;
    float eps = 1e-6;
    for (int i = 0; i < length_C_1; i++) {
        if (abs(C_1[i] - C_2[i]) > eps)
            return false;
    }
    return true;
}

}}  // namespace cks::sgemm