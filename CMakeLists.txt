cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(LARS LANGUAGES C CUDA)

set(LIBS "${CMAKE_SOURCE_DIR}/libs")
set(ORT_LIB_DIR "${LIBS}/onnxruntime")
set(SHERPA_LIB_DIR "${LIBS}/sherpa-onnx")

find_package(CUDAToolkit REQUIRED)

find_library(ORT_LIB    onnxruntime         PATHS ${ORT_LIB_DIR}    NO_DEFAULT_PATH REQUIRED)
find_library(ALSA_LIB   asound              REQUIRED)
find_library(SHERPA_LIB sherpa-onnx-c-api   PATHS ${SHERPA_LIB_DIR} NO_DEFAULT_PATH REQUIRED)

add_subdirectory(src)
#enable_testing()

#add_subdirectory(tests)
