# Command to build:
# cmake -B build -DGGML_ZENDNN=ON -DGGML_ZENDNN_PATH=/path/to/zendnn

ggml_add_backend_library(ggml-zendnn
                         ggml-zendnn.cpp
                        )

# Get ZenDNN path
if (NOT DEFINED GGML_ZENDNN_PATH)
    set(GGML_ZENDNN_PATH "$ENV{GGML_ZENDNN_PATH}")
endif()

if ("${GGML_ZENDNN_PATH}" STREQUAL "")
    message(FATAL_ERROR "GGML_ZENDNN_PATH is not set. Please set it to the path of ZenDNN installation.")
endif()

# ZenDNN headers + libs
target_include_directories(ggml-zendnn PRIVATE
    ${GGML_ZENDNN_PATH}/inc
    ${GGML_ZENDNN_PATH}/third_party/amd-blis/include/LP64
)

target_link_directories(ggml-zendnn PRIVATE
    ${GGML_ZENDNN_PATH}/_out/lib
    ${GGML_ZENDNN_PATH}/third_party/amd-blis/lib/LP64
)

target_link_libraries(ggml-zendnn PRIVATE
    amdZenDNN
    blis-mt
    m
    pthread
)

if (GGML_OPENMP)
    target_link_libraries(ggml-zendnn PRIVATE OpenMP::OpenMP_CXX)
endif()
