if (GGML_STATIC)
    set(BLA_STATIC ON)
endif()

set(BLA_VENDOR ${GGML_BLAS_VENDOR})
find_package(BLAS)

if (BLAS_FOUND)
    message(STATUS "BLAS found, Libraries: ${BLAS_LIBRARIES}")

    ggml_add_backend_library(ggml-blas
                             ggml-blas.cpp
                            )

    if (${GGML_BLAS_VENDOR} MATCHES "Apple")
        add_compile_definitions(ACCELERATE_NEW_LAPACK)
        add_compile_definitions(ACCELERATE_LAPACK_ILP64)
        add_compile_definitions(GGML_BLAS_USE_ACCELERATE)
    elseif ("${BLAS_INCLUDE_DIRS}" STREQUAL "")
        find_package(PkgConfig REQUIRED)
        if (${GGML_BLAS_VENDOR} MATCHES "Generic")
            pkg_check_modules(DepBLAS blas)
        elseif (${GGML_BLAS_VENDOR} MATCHES "OpenBLAS")
            pkg_check_modules(DepBLAS openblas64)
            if (NOT DepBLAS_FOUND)
                pkg_check_modules(DepBLAS openblas)
            endif()
        elseif (${GGML_BLAS_VENDOR} MATCHES "FLAME")
            add_compile_definitions(GGML_BLAS_USE_BLIS)
            pkg_check_modules(DepBLAS blis)
        elseif (${GGML_BLAS_VENDOR} MATCHES "ATLAS")
            pkg_check_modules(DepBLAS blas-atlas)
        elseif (${GGML_BLAS_VENDOR} MATCHES "FlexiBLAS")
            pkg_check_modules(DepBLAS flexiblas_api)
        elseif (${GGML_BLAS_VENDOR} MATCHES "Intel")
            add_compile_definitions(GGML_BLAS_USE_MKL)
            pkg_check_modules(DepBLAS mkl-sdl)
        elseif (${GGML_BLAS_VENDOR} MATCHES "NVHPC")
            if ("${NVHPC_VERSION}" STREQUAL "")
                message(WARNING "Better to set NVHPC_VERSION")
            else()
                set(DepBLAS_FOUND ON)
                set(DepBLAS_INCLUDE_DIRS "/opt/nvidia/hpc_sdk/${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}/${NVHPC_VERSION}/math_libs/include")
            endif()
        endif()
        if (DepBLAS_FOUND)
            set(BLAS_INCLUDE_DIRS ${DepBLAS_INCLUDE_DIRS})
        else()
            message(WARNING "BLAS_INCLUDE_DIRS neither been provided nor been automatically"
            " detected by pkgconfig, trying to find cblas.h from possible paths...")
            find_path(BLAS_INCLUDE_DIRS
                NAMES cblas.h
                HINTS
                    /usr/include
                    /usr/local/include
                    /usr/include/openblas
                    /opt/homebrew/opt/openblas/include
                    /usr/local/opt/openblas/include
                    /usr/include/x86_64-linux-gnu/openblas/include
            )
        endif()
    endif()

    message(STATUS "BLAS found, Includes: ${BLAS_INCLUDE_DIRS}")

    target_compile_options(ggml-blas PRIVATE ${BLAS_LINKER_FLAGS})

    if ("${BLAS_INCLUDE_DIRS}" MATCHES "mkl" AND (${GGML_BLAS_VENDOR} MATCHES "Generic" OR ${GGML_BLAS_VENDOR} MATCHES "Intel"))
        add_compile_definitions(GGML_BLAS_USE_MKL)
    endif()

    target_link_libraries(ggml-blas PRIVATE ${BLAS_LIBRARIES})
    target_include_directories(ggml-blas PRIVATE ${BLAS_INCLUDE_DIRS})

    if (GGML_OPENMP)
        find_package(OpenMP)
        if (OpenMP_FOUND)
            message(STATUS "OpenMP found, enabling parallelization for ggml-blas")
            target_compile_definitions(ggml-blas PRIVATE GGML_USE_OPENMP)
            target_link_libraries(ggml-blas PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
        else()
            message(WARNING "OpenMP requested but not found")
        endif()
    endif()

    if (GGML_ZENDNN)
        message(STATUS "Searching for ZenDNN in ${ZENDNN_PATH}")
        set(ZENDNN_SEARCH_DIRS
            ${ZENDNN_PATH}
            ${ZENDNN_PATH}/lib
            ${ZENDNN_PATH}/build/lib
            ${ZENDNN_PATH}/build_Release/lib
            ${ZENDNN_PATH}/build/Linux-x86_64-Release/lib
            ${ZENDNN_PATH}/_out
            ${ZENDNN_PATH}/_out/lib
        )
        find_library(ZENDNN_LIBRARY
            NAMES amdZenDNN ZenDNN
            PATHS ${ZENDNN_SEARCH_DIRS}
            NO_DEFAULT_PATH
        )
        find_path(ZENDNN_INCLUDE_DIR
            NAMES zendnn.hpp
            PATHS ${ZENDNN_PATH}/include ${ZENDNN_PATH}/inc
            NO_DEFAULT_PATH
        )
        if (ZENDNN_LIBRARY AND ZENDNN_INCLUDE_DIR)
            message(STATUS "Found ZenDNN library: ${ZENDNN_LIBRARY}")
            message(STATUS "Found ZenDNN headers: ${ZENDNN_INCLUDE_DIR}")
            target_link_libraries(ggml-blas PRIVATE ${ZENDNN_LIBRARY})
            target_include_directories(ggml-blas PRIVATE ${ZENDNN_INCLUDE_DIR})
            target_compile_definitions(ggml-blas PRIVATE GGML_USE_ZENDNN)
        else()
            message(WARNING "ZenDNN requested but not found in ${ZENDNN_PATH}")
            message(WARNING "  Library: ${ZENDNN_LIBRARY}")
            message(WARNING "  Include: ${ZENDNN_INCLUDE_DIR}")
        endif()
    endif()
else()
    message(FATAL_ERROR "BLAS not found, please refer to "
                        "https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors"
                        " to set correct GGML_BLAS_VENDOR")
endif()
