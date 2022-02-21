#ifndef CKS_COMMON_DTYPE_CUH_
#define CKS_COMMON_DTYPE_CUH_

namespace cks {namespace common{

// Return type for low-level APIs
typedef int retCode_t;

// bass class for kernel arguments
class KernelArgs {};

// base class for data loading
class DataLoader {
public:    
    virtual RetCode loadData(KernelArgs **p_data);
    
    virtual RetCode freeData(KernelArgs *p_data);
    
    virtual RetCode step();
    
    virtual RetCode log();
};

}}  // namespace cks::common

#endif