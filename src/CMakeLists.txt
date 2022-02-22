enable_language(CUDA)

file(GLOB_RECURSE CKS_MAIN_SRC 
    common/*.cu
    data/*.cu
    kernel/*.cu)

# message(STATUS "${CKS_MAIN_SRC}")
add_executable(main main.cu ${CKS_MAIN_SRC})