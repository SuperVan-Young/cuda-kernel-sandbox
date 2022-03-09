#ifndef CKS_KERNEL_SGEMM_CUH_
#define CKS_KERNEL_SGEMM_CUH_

#include "common/dtype.cuh"
#include "common/error.cuh"

#include "data/sgemm.cuh"

#include <cublas_v2.h>

namespace cks { namespace sgemm {

float runKernel(int version, cks::sgemm::SgemmArgs *args);

float speedTestKernel(int version, cks::sgemm::SgemmArgs *args);

void sgemmKernel_cuBLAS(int M, int N, int K, const float *d_alpha, const float *d_beta, 
                        const float *d_A, const float *d_B, float *d_C);

void sgemmKernel_v1(int M, int N, int K, const float *d_alpha, const float *d_beta, 
                    const float *d_A, const float *d_B, float *d_C);

void sgemmKernel_v2(int M, int N, int K, const float *d_alpha, const float *d_beta, 
                    const float *d_A, const float *d_B, float *d_C);

void sgemmKernel_v3(int M, int N, int K, const float *d_alpha, const float *d_beta, 
                    const float *d_A, const float *d_B, float *d_C);

void sgemmKernel_v4(int M, int N, int K, const float *d_alpha, const float *d_beta, 
                    const float *d_A, const float *d_B, float *d_C);

void sgemmKernel_v5(int M, int N, int K, const float *d_alpha, const float *d_beta, 
                    const float *d_A, const float *d_B, float *d_C);
}}  // namespace cks::sgemm

#endif