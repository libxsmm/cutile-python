# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# LLVM builds the compiler as an external project, separately from the launchers.
if(TARGET TileIRToMLIR)
    include("${CMAKE_CURRENT_SOURCE_DIR}/compiler.cmake")
    return()
endif()

# Builds the native parts of the non-CUDA backend selected by CUTILE_BUILD_VARIANT:
# the native compiler and the backend's launcher extension. Installing
# the build (`cmake --install <build> --prefix <site> --component cuTileBackend`)
# stages them and their runtime files into <site>/cuda/tile/_backends, next to the
# Python backend in src/cuda/tile/_backends.

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The native CPU compiler currently supports Linux only")
endif()

if (NOT CUTILE_BUILD_VARIANT STREQUAL "cpu")
    message(FATAL_ERROR "CUTILE_BUILD_VARIANT must be one of: cuda, cpu")
endif()

# Installing only this component skips the install rules of cuTile's other
# dependencies, such as dlpack.
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME cuTileBackend)

set(CUTILE_NATIVE_SDK_DIR "" CACHE PATH "Installed Python-independent native SDK")
if(CUTILE_BUILD_NATIVE_SDK)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchTileIRToMLIR.cmake")
    fetch_tileir_to_mlir()
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/TileIRToMLIR.cmake")
    return()
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
find_package(nanobind CONFIG REQUIRED)

# Package directory of the backends, relative to the install prefix.
set(CUTILE_BACKEND_INSTALL_DIR cuda/tile/_backends)

if(CUTILE_NATIVE_SDK_DIR)
    list(PREPEND CMAKE_PREFIX_PATH "${CUTILE_NATIVE_SDK_DIR}")
    find_package(TileIRToMLIR REQUIRED CONFIG
        PATHS "${CUTILE_NATIVE_SDK_DIR}" NO_DEFAULT_PATH)
    find_package(LLD REQUIRED CONFIG)
    add_library(TileIRToMLIR ALIAS TileIRToMLIR::TileIRToMLIR)
    if(NOT TARGET CudaTileBytecodeReader)
        add_library(CudaTileBytecodeReader ALIAS TileIRToMLIR::CudaTileBytecodeReader)
    endif()
    include("${CMAKE_CURRENT_SOURCE_DIR}/compiler.cmake")
    install(FILES "${CUTILE_NATIVE_SDK_DIR}/share/cutile/llvm/LICENSE.TXT"
        DESTINATION "${CUTILE_BACKEND_INSTALL_DIR}/_licenses/llvm")
    install(FILES "${CUTILE_NATIVE_SDK_DIR}/share/TileIRToMLIR/LICENSE"
        DESTINATION "${CUTILE_BACKEND_INSTALL_DIR}/_licenses/tileir-to-mlir")
    install(FILES "${CUTILE_NATIVE_SDK_DIR}/share/TileIRToMLIR/cuda-tile/LICENSE.txt"
        DESTINATION "${CUTILE_BACKEND_INSTALL_DIR}/_licenses/cuda-tile")
else()
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchTileIRToMLIR.cmake")
    fetch_tileir_to_mlir()
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/TileIRToMLIR.cmake")

    install(FILES "${CUTILE_COMPILER_PATH}" DESTINATION "${CUTILE_BACKEND_INSTALL_DIR}")
    install(FILES "${CUTILE_LLVM_LICENSE_PATH}"
        DESTINATION "${CUTILE_BACKEND_INSTALL_DIR}/_licenses/llvm")
    install(FILES "${CUTILE_TILEIR_TO_MLIR_DIR}/LICENSE"
        DESTINATION "${CUTILE_BACKEND_INSTALL_DIR}/_licenses/tileir-to-mlir")
    install(FILES "${CUTILE_CUDA_TILE_SOURCE_DIR}/LICENSE.txt"
        DESTINATION "${CUTILE_BACKEND_INSTALL_DIR}/_licenses/cuda-tile")
endif()

add_subdirectory("${CUTILE_BUILD_VARIANT}")
