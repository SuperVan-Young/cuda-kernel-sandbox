#ifndef CKS_COMMON_ERROR_CUH_
#define CKS_COMMON_ERROR_CUH_

#include "common/dtype.cuh"

// macros for retCode_t
typedef cks::common::retCode_t retCode_t;

#define RC_SUCCESS  0
#define RC_ERROR   -1

static cksGetErrorString(retCode_t error_code) {
    switch (error_code) {
        case RC_ERROR: return "Error";
        default: return "Unknown error";
    }
}

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

#define CKS_CALL(call)                                                   \
do {                                                                     \
    const cks::common::retCode_t error_code = call;                      \
    if (error_code != RC_SUCCESS) {                                      \
        printf("CKS Error:\n");                                          \
        printf("    File:       %s\n", __FILE__);                        \
        printf("    Line:       %s\n", __LINE__);                        \
        printf("    Error Code: %d\n", error_code);                      \
        printf("    Error Text: %s\n", cksGetErrorString(error_code));   \
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

#endif