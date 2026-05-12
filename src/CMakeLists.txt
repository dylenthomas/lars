cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(LARS LANGUAGES C CUDA)

set(PROJECT_SOURCE "/home/dylenthomas/LiveASRonRPi-4")
set(LIBS "${PROJECT_SOURCE}/libs")
set(ORT_LIB_DIR "${LIBS}/onnxruntime")
set(VENV_DIR ${CMAKE_BINARY_DIR}/venv)

find_package(CUDAToolkit REQUIRED)

find_package(Python3 3.14 EXACT REQUIRED COMPONENTS Interpreter)

if(NOT EXISTS ${VENV_DIR})
    message(STATUS "Creating virtual environment as ${VENV_DIR}...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m venv ${VENV_DIR}
        RESULT_VARIABLE VENV_RESULT
    )
    if(NOT VENV_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to create virtual environment")
    endif()
endif()

message(STATUS "Installing Python dependencies...")
execute_process(
    COMMAND ${VENV_DIR}/bin/pip install numpy torch av ctranslate2 tokenizers torchaudio
    RESULT_VARIABLE PIP_RESULT
)
if(NOT PIP_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to install Python dependencies!")
endif()

set(Python3_ROOT_DIR ${VENV_DIR})
set(Python3_EXECUTABLE ${VENV_DIR}/bin/python3)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

message(STATUS "Using Python: ${Python3_EXECUTABLE}")

find_library(ORT_LIB onnxruntime PATHS ${ORT_LIB_DIR} NO_DEFAULT_PATH REQUIRED)
find_library(MIC_LIB mic_access PATHS ${LIBS} NO_DEFAULT_PATH REQUIRED)
find_library(FFT_LIB fft PATHS ${LIBS} NO_DEFAULT_PATH REQUIRED)
find_library(ALSA_LIB asound REQUIRED)
find_library(CNFG_PARSR config_parser PATHS ${LIBS} NO_DEFAULT_PATH REQUIRED)

add_executable(main main.c)

target_include_directories(main PRIVATE
    "${PROJECT_SOURCE}/include"
    "${PROJECT_SOURCE}/include/onnxruntime"
    ${Python3_INCLUDE_DIRS}
)

target_link_libraries(main PRIVATE
    ${ORT_LIB}
    ${ALSA_LIB}
    ${MIC_LIB}
    ${FFT_LIB}
    ${TRANSC_LIB}
    ${CNFG_PARSR}
    ${Python3_LIBRARIES}
    m
    CUDA::cudart
    CUDA::cublas
)

set_target_properties(main PROPERTIES 
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${ORT_LIB_DIR}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
