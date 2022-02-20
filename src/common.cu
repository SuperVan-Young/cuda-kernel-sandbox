#include "common.cuh"

using namespace std;
using namespace cks::common;

RetCode cks::common::runKernel(int kernel, int version, KernelArgs* args) {
    printf("Hello world!\n");
    return RC_SUCCESS;
}
