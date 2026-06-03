cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(dual_mic_test LANGUAGES C)

set(PROJECT_SOURCE "/home/dylenthomas/LiveASRonRPi-4")
set(LIBS "${PROJECT_SOURCE}/libs")

find_library(MIC_LIB mic_access PATHS ${LIBS} NO_DEFAULT_PATH REQUIRED)
find_library(FFT_LIB fft PATHS ${LIBS} NO_DEFAULT_PATH REQUIRED)
find_library(ALSA_LIB asound REQUIRED)

add_executable(dual-mic-test dual-mic-test.c)

include_directories(dual-mic-test PRIVATE
    "${PROJECT_SOURCE}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(dual-mic-test PRIVATE
    ${ALSA_LIB}
    ${MIC_LIB}
    ${FFT_LIB}
    m
)

set_target_properties(dual-mic-test PROPERTIES 
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
