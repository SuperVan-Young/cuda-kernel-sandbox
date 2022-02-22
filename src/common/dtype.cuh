#ifndef CKS_COMMON_DTYPE_CUH_
#define CKS_COMMON_DTYPE_CUH_

namespace cks {namespace common{

/**
 * Return type of low-level APIs
 * It's recommended to call these APIs with macro `CKS_CALL`
 */
typedef enum {
    RC_SUCCESS = 0,
    RC_ERROR =  -1,
} retCode_t;

/**
 * Base Class for kernel arguments.
 */
class KernelArgs {
public:
    virtual retCode_t deepcopy(KernelArgs **p_target) = 0;

    virtual retCode_t destroyCopy(KernelArgs *p_target) = 0;
    
    virtual bool equalResult(KernelArgs *p_target) = 0;
};


/**
 * Base Class for loading data
 */
class DataLoader {
public:

    virtual retCode_t loadData(KernelArgs **p_data) = 0;
    
    virtual retCode_t freeData(KernelArgs *p_data) = 0;

    virtual int len() = 0;
    
    virtual retCode_t step() = 0;
    
    virtual retCode_t log(float perf) = 0;
};

}}  // namespace cks::common

#endif