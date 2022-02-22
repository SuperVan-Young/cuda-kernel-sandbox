#ifndef CKS_COMMON_DTYPE_CUH_
#define CKS_COMMON_DTYPE_CUH_

namespace cks {namespace common{

// Return type for low-level APIs
typedef enum {
    RC_SUCCESS = 0,
    RC_ERROR =  -1,
} retCode_t;

// bass class for kernel arguments
class KernelArgs {};

// base class for data loading
class DataLoader {
public:

    virtual retCode_t loadData(KernelArgs **p_data) = 0;
    
    virtual retCode_t freeData(KernelArgs *p_data) = 0;

    virtual int len() = 0;
    
    virtual retCode_t step() = 0;
    
    virtual retCode_t log() = 0;
};

}}  // namespace cks::common

#endif