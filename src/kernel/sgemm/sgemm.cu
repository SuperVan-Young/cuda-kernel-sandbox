#include "kernel/sgemm.cuh"

namespace cks { namespace sgemm {

float runKernel(int version, SgemmArgs *args) {
    // prepare function arguments and copy them to device

    int M = args->M;
    int N = args->N;
    int K = args->K;
    float *h_alpha = args->alpha;
    float *h_beta = args->beta;
    float *h_A = args->A;
    float *h_B = args->B;
    float *h_C = args->C;

    int length_A = M * K;
    int length_B = K * N;
    int length_C = M * N;

    float *d_alpha, *d_beta, *d_A, *d_B, *d_C;
    CUDA_CALL(cudaMalloc((void **)&d_alpha, sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&d_beta, sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&d_A, sizeof(float)*length_A));
    CUDA_CALL(cudaMalloc((void **)&d_B, sizeof(float)*length_B));
    CUDA_CALL(cudaMalloc((void **)&d_C, sizeof(float)*length_C));

    CUDA_CALL(cudaMemcpy(d_alpha, h_alpha, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_beta, h_beta, sizeof(float), cudaMemcpyHostToDevice));
    if (version == 0) {
        cublasSetVector(length_A, sizeof(float), h_A, 1, d_A, 1);
        cublasSetVector(length_B, sizeof(float), h_B, 1, d_B, 1);
        cublasSetVector(length_C, sizeof(float), h_C, 1, d_C, 1);
    }
    else {
        CUDA_CALL(cudaMemcpy(d_A, h_A, sizeof(float)*length_A, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_B, h_B, sizeof(float)*length_B, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_C, h_C, sizeof(float)*length_C, cudaMemcpyHostToDevice));
    }

    // start timing and call kernel function
    CUDA_CALL(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    CUDA_CALL(cudaEventRecord(start));
    cudaEventQuery(start);
    //==========================================================================

    CUDA_CALL(cudaDeviceSynchronize());
    switch (version) {
        case 0:  sgemmKernel_cuBLAS(M, N, K, h_alpha, h_beta, d_A, d_B, d_C); break;
        case 1:  sgemmKernel_v1(M, N, K, d_alpha, d_beta, d_A, d_B, d_C); break;
        case 2:  sgemmKernel_v2(M, N, K, d_alpha, d_beta, d_A, d_B, d_C); break;
        case 3:  sgemmKernel_v3(M, N, K, d_alpha, d_beta, d_A, d_B, d_C); break;
        default: break;
    }
    CUDA_CALL(cudaDeviceSynchronize());
    
    //==========================================================================

    // CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));
    float elapsed_time;
    CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));

    // clean up the memory
    CUDA_CALL(cudaMemcpy(h_C, d_C, sizeof(float)*length_C, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaFree(d_alpha));
    CUDA_CALL(cudaFree(d_beta));
    CUDA_CALL(cudaFree(d_A));
    CUDA_CALL(cudaFree(d_B));
    CUDA_CALL(cudaFree(d_C));

    return elapsed_time;
}

float speedTestKernel(int version, SgemmArgs *args) {
    // prepare function arguments and copy them to device

    int M = args->M;
    int N = args->N;
    int K = args->K;
    float *h_alpha = args->alpha;
    float *h_beta = args->beta;
    float *h_A = args->A;
    float *h_B = args->B;
    float *h_C = args->C;

    int length_A = M * K;
    int length_B = K * N;
    int length_C = M * N;

    float *d_alpha, *d_beta, *d_A, *d_B, *d_C;
    CUDA_CALL(cudaMalloc((void **)&d_alpha, sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&d_beta, sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&d_A, sizeof(float)*length_A));
    CUDA_CALL(cudaMalloc((void **)&d_B, sizeof(float)*length_B));
    CUDA_CALL(cudaMalloc((void **)&d_C, sizeof(float)*length_C));

    CUDA_CALL(cudaMemcpy(d_alpha, h_alpha, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_beta, h_beta, sizeof(float), cudaMemcpyHostToDevice));
    if (version == 0) {
        cublasSetVector(length_A, sizeof(float), h_A, 1, d_A, 1);
        cublasSetVector(length_B, sizeof(float), h_B, 1, d_B, 1);
        cublasSetVector(length_C, sizeof(float), h_C, 1, d_C, 1);
    }
    else {
        CUDA_CALL(cudaMemcpy(d_A, h_A, sizeof(float)*length_A, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_B, h_B, sizeof(float)*length_B, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_C, h_C, sizeof(float)*length_C, cudaMemcpyHostToDevice));
    }

    // start timing and call kernel function
    CUDA_CALL(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    CUDA_CALL(cudaEventRecord(start));
    cudaEventQuery(start);
    //==========================================================================

    CUDA_CALL(cudaDeviceSynchronize());
    int num_iteration = 10;
    for (int i = 0; i < num_iteration; i++) {
        switch (version) {
            case 0:  sgemmKernel_cuBLAS(M, N, K, h_alpha, h_beta, d_A, d_B, d_C); break;
            case 1:  sgemmKernel_v1(M, N, K, d_alpha, d_beta, d_A, d_B, d_C); break;
            case 2:  sgemmKernel_v2(M, N, K, d_alpha, d_beta, d_A, d_B, d_C); break;
            case 3:  sgemmKernel_v3(M, N, K, d_alpha, d_beta, d_A, d_B, d_C); break;
            default: break;
        }
        CUDA_CALL(cudaDeviceSynchronize());
    }
    
    //==========================================================================

    // CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));
    float elapsed_time;
    CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));
    elapsed_time /= num_iteration;
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));

    // clean up the memory (without returning the result)
    // CUDA_CALL(cudaMemcpy(h_C, d_C, sizeof(float)*length_C, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaFree(d_alpha));
    CUDA_CALL(cudaFree(d_beta));
    CUDA_CALL(cudaFree(d_A));
    CUDA_CALL(cudaFree(d_B));
    CUDA_CALL(cudaFree(d_C));

    return elapsed_time;
}

}}  // namespace cks::sgemm