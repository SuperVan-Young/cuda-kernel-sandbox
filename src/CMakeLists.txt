enable_language(CUDA)

set(CMAKE_INSTALL_BINDIR "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_INSTALL_LIBDIR "${CMAKE_SOURCE_DIR}/lib")

file(GLOB_RECURSE CKS_MAIN_LIB
    common/*.cu
    data/*.cu
    kernel/*.cu)

add_library(cks ${CKS_MAIN_LIB})

add_executable(print print.cu)
target_compile_features(print PRIVATE cxx_std_11)
target_link_libraries(print cks ${CUBLAS_LIBRARIES})

add_executable(speedtest speedtest.cu)
target_compile_features(speedtest PRIVATE cxx_std_11)
target_link_libraries(speedtest cks ${CUBLAS_LIBRARIES})

add_executable(verify verify.cu)
target_compile_features(verify PRIVATE cxx_std_11)
target_link_libraries(verify cks ${CUBLAS_LIBRARIES})

install(TARGETS speedtest verify cks print
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
