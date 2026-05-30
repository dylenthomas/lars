add_library(lars_core STATIC
        config_parser.c
        fft.c
        mic_access.c
)

target_include_directories(lars_core PUBLIC
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_SOURCE_DIR}/include/onnxruntime"
        "${CMAKE_SOURCE_DIR}/include/sherpa-onnx"
)

target_link_libraries(lars_core PUBLIC
        ${ORT_LIB}
        ${ALSA_LIB}
        ${MIC_LIB}
        ${FFT_LIB}
        ${TRANSC_LIB}
        ${CNFG_PARSR}
        ${SHERPA_LIB}
        m
        CUDA::cudart
        CUDA::cublas
)

add_executable(lars main.c
)
target_link_libraries(lars PRIVATE lars_core)

set_target_properties(lars PROPERTIES
        SKIP_BUILD_RPATH FALSE
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
        INSTALL_RPATH "${ORT_LIB_DIR};${SHERPA_LIB_DIR}"
)
