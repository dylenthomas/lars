cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(OnnxGpuTest LANGUAGES C CUDA)

set(PROJECT_SOURCE "/home/dylenthomas/LiveASRonRPi-4")
set(LIBS "${PROJECT_SOURCE}/libs")
set(ORT_LIB_DIR "${LIBS}/onnxruntime")

find_library(ORT_LIB onnxruntime PATHS ${ORT_LIB_DIR} NO_DEFAULT_PATH REQUIRED)
find_library(MIC_LIB mic_access PATHS ${LIBS} NO_DEFAULT_PATH REQUIRED)
find_library(ALSA_LIB asound REQUIRED)

add_executable(test onnxrt_vad_gpu_test.c)

include_directories(test PRIVATE
    "${PROJECT_SOURCE}/include"
    "${PROJECT_SOURCE}/include/onnxruntime"
)


target_link_libraries(test PRIVATE
    ${ORT_LIB}
    ${ALSA_LIB}
    ${MIC_LIB}
    m
)

set_target_properties(test PROPERTIES 
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${ORT_LIB_DIR}:${LOCAL_LIB_DIR}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
