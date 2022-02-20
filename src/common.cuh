#ifndef _CKS_COMMON_CUH_
#define _CKS_COMMON_CUH_

#include <stdio.h>

#define CUDA_CALL(call)                                                  \
do {                                                                     \
    const cudaError_t error_code = call;                                 \
    if (error_code != cudaSuccess) {                                     \
        printf("CUDA Error:\n");                                         \
        printf("    File:       %s\n", __FILE__);                        \
        printf("    Line:       %s\n", __LINE__);                        \
        printf("    Error Code: %d\n", error_code);                      \
        printf("    Error Text: %s\n", cudaGetErrorString(error_code));  \
        exit(1);                                                         \
    }                                                                    \
} while (0)

#define CALL_ERROR(msg)                        \
do {                                           \
    printf("ERROR:\n");                        \
    printf("    File:       %s\n", __FILE__);  \
    printf("    Line:       %s\n", __LINE__);  \
    printf("    Error Text: %s\n", msg);       \
    exit(1);                                   \
} while (0)

using namespace std;
namespace cks { namespace common {

typedef int RetCode;

#define RC_SUCCESS  0
#define RC_ERROR   -1

class KernelArgs {};

class DataLoader {
public:
    char logdir[100];

    RetCode loadData(KernelArgs **p_data);

    RetCode freeData(KernelArgs *p_data);

    RetCode step();

    RetCode log();
};

RetCode runKernel(int kernel, int version, KernelArgs* args);

bool verifyKernel(int kernel, int version, KernelArgs* args);

double speedTestKernel(int kernel, int version, KernelArgs* args);

}} // namespace cks::common

#endif