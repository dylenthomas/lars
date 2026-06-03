cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(LARS C)

set(PROJECT_SOURCE "/home/dylenthomas/LiveASRonRPi-4")
set(VENV_DIR ${CMAKE_BINARY_DIR}/venv)

find_package(Python3 3.14 EXACT REQUIRED COMPONENTS Interpreter)

if(NOT EXISTS ${VENV_DIR})
    message(STATUS "Creating virtual environment at ${VENV_DIR}...")
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
    COMMAND ${VENV_DIR}/bin/pip install numpy torch
    RESULT_VARIABLE PIP_RESULT
)
if(NOT PIP_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to install Python dependencies")
endif()

set(Python3_ROOT_DIR ${VENV_DIR})
set(Python3_EXECUTABLE ${VENV_DIR}/bin/python3)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

message(STATUS "Using Python: ${Python3_EXECUTABLE}")

add_executable(main main.c)

target_include_directories(main PRIVATE
    ${Python3_INCLUDE_DIRS}
)

target_link_libraries(main PRIVATE
    ${Python3_LIBRARIES}
)

set_target_properties(main PROPERTIES 
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
