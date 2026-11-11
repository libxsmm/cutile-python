# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET TileIRToMLIR)
    message(FATAL_ERROR "Build cuTileCompiler alongside the TileIRToMLIR LLVM external project")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_compiler FREE_THREADED mlir/compiler.cpp)
target_include_directories(_compiler PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
if(CUTILE_NATIVE_SDK_DIR)
    target_include_directories(_compiler SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
    separate_arguments(_llvm_definitions NATIVE_COMMAND "${LLVM_DEFINITIONS}")
    target_compile_definitions(_compiler PRIVATE ${_llvm_definitions})
else()
    target_include_directories(_compiler SYSTEM PRIVATE
        "${LLVM_MAIN_SRC_DIR}/../mlir/include"
        "${LLVM_BINARY_DIR}/tools/mlir/include")
endif()
target_link_libraries(_compiler PRIVATE
    TileIRToMLIR CudaTileBytecodeReader
    MLIRParser MLIRTransforms
    MLIRAffineDialect MLIRArithDialect MLIRControlFlowDialect MLIRFuncDialect
    MLIRGPUDialect MLIRIndexDialect MLIRLLVMDialect MLIRMathDialect
    MLIRMemRefDialect MLIRSCFDialect MLIRUBDialect
    MLIRVectorDialect MLIRX86Dialect MLIRFuncInlinerExtension
    MLIRArithValueBoundsOpInterfaceImpl
    MLIRArithToLLVM MLIRControlFlowToLLVM MLIRFuncToLLVM MLIRIndexToLLVM
    MLIRMathToLLVM MLIRMemRefToLLVM MLIRUBToLLVM MLIRVectorToLLVM
    MLIRX86Transforms
    MLIRBuiltinToLLVMIRTranslation MLIRLLVMToLLVMIRTranslation
    MLIRVectorTransforms)
target_sources(_compiler PRIVATE mlir/LowerVectorScan.cpp)
target_link_options(_compiler PRIVATE "LINKER:--exclude-libs,ALL")
set_target_properties(_compiler PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
if(CUTILE_NATIVE_SDK_DIR)
    set_target_properties(_compiler PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
else()
    set_target_properties(_compiler PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${LLVM_BINARY_DIR}/lib")
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/${CUTILE_BUILD_VARIANT}/compiler.cmake")

if(NOT LLVM_ENABLE_RTTI)
    set_source_files_properties(
        mlir/LowerVectorScan.cpp
        PROPERTIES COMPILE_OPTIONS "-fno-rtti")
endif()

if(CUTILE_NATIVE_SDK_DIR)
    install(TARGETS _compiler LIBRARY DESTINATION "${CUTILE_BACKEND_INSTALL_DIR}")
else()
    install(TARGETS _compiler LIBRARY DESTINATION lib COMPONENT cuTileCompiler)
    install(FILES "${LLVM_MAIN_SRC_DIR}/LICENSE.TXT"
        DESTINATION share/cutile/llvm COMPONENT cuTileCompiler)
endif()
