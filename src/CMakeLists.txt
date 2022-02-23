enable_language(CUDA)

set(CMAKE_INSTALL_BINDIR "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_INSTALL_LIBDIR "${CMAKE_SOURCE_DIR}/lib")

file(GLOB_RECURSE CKS_MAIN_LIB
    common/*.cu
    data/*.cu
    kernel/*.cu)

add_library(cks ${CKS_MAIN_LIB})

add_executable(main main.cu)
target_compile_features(main PRIVATE cxx_std_11)
target_link_libraries(main cks ${CUBLAS_LIBRARIES})

install(TARGETS main cks
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
