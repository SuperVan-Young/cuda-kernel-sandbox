#ifndef CKS_COMMON_DTYPE_CUH_
#define CKS_COMMON_DTYPE_CUH_

#include "error.cuh"

namespace cks {namespace common{

/**
 * Base Class for kernel arguments.
 */
class KernelArgs {
};

/**
 * Base Class for loading data
 */
class DataLoader {
public:

    virtual retCode_t loadData(KernelArgs **p_data) = 0;
    
    virtual retCode_t freeData(KernelArgs *p_data) = 0;

    virtual retCode_t deepcopyData(KernelArgs **p_dst, KernelArgs *p_src) = 0;

    virtual bool equalResult(KernelArgs *p_1, KernelArgs *p_2) = 0;

    virtual int len() = 0;
    
    virtual retCode_t step() = 0;
    
    virtual retCode_t log(float perf) = 0;
};

}}  // namespace cks::common

#endif