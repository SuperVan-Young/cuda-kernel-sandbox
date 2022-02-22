enable_language(CUDA)

set(CKS_MAIN_SRC)

file(GLOB_RECURSE CKS_DATA_SRC 
    src/common/*.cu
    src/data/*.cu
    src/kernel/*.cu)

add_executable(main main.cu ${CKS_MAIN_SRC})