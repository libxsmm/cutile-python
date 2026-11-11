# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.19)

foreach(_required GIT_EXECUTABLE SOURCE_DIR TILEIR_REQUIRED_LLVM_REVISION)
    if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
        message(FATAL_ERROR "${_required} is required")
    endif()
endforeach()

if(NOT EXISTS "${SOURCE_DIR}/.git")
    if(EXISTS "${SOURCE_DIR}")
        file(GLOB _source_contents "${SOURCE_DIR}/*" "${SOURCE_DIR}/.[!.]*")
        if(_source_contents)
            message(FATAL_ERROR
                "LLVM source directory exists but is not a Git checkout: ${SOURCE_DIR}")
        endif()
    endif()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" clone --filter=blob:none --no-checkout
                https://github.com/llvm/llvm-project.git "${SOURCE_DIR}"
        COMMAND_ERROR_IS_FATAL ANY)
endif()

set(_required_revisions "${TILEIR_REQUIRED_LLVM_REVISION}")
if(DEFINED REVISION AND NOT REVISION STREQUAL "")
    list(APPEND _required_revisions "${REVISION}")
endif()

foreach(_revision IN LISTS _required_revisions)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" cat-file -e "${_revision}^{commit}"
        RESULT_VARIABLE _revision_exists
        ERROR_QUIET)
    if(NOT _revision_exists EQUAL 0)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" fetch --filter=blob:none
                    origin "${_revision}"
            COMMAND_ERROR_IS_FATAL ANY)
    endif()
endforeach()

set(_selected_revision "${TILEIR_REQUIRED_LLVM_REVISION}")
if(DEFINED REVISION AND NOT REVISION STREQUAL "")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" merge-base --is-ancestor
                "${_selected_revision}" "${REVISION}"
        RESULT_VARIABLE _required_is_ancestor)
    if(NOT _required_is_ancestor EQUAL 0)
        message(FATAL_ERROR
            "LLVM override ${REVISION} does not descend from required revision "
            "${_selected_revision}")
    endif()
    set(_selected_revision "${REVISION}")
endif()

set(_patch "")
if(DEFINED PATCH AND NOT PATCH STREQUAL "")
    file(READ "${PATCH}" _patch)
endif()
execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" rev-parse --absolute-git-dir
    OUTPUT_VARIABLE _git_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
# Copy of the patch applied to the working tree, needed to revert it.
set(_applied_patch "${_git_dir}/cutile-llvm.patch")
set(_applied "")
if(EXISTS "${_applied_patch}")
    file(READ "${_applied_patch}" _applied)
endif()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" rev-parse HEAD
    RESULT_VARIABLE _head_result
    OUTPUT_VARIABLE _head_revision
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(_head_result EQUAL 0 AND _head_revision STREQUAL _selected_revision
   AND _applied STREQUAL _patch)
    return()
endif()

if(EXISTS "${_applied_patch}")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" apply --reverse "${_applied_patch}"
        COMMAND_ERROR_IS_FATAL ANY)
    file(REMOVE "${_applied_patch}")
endif()

if(NOT _head_revision STREQUAL _selected_revision)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" checkout --detach "${_selected_revision}"
        COMMAND_ERROR_IS_FATAL ANY)
endif()

if(NOT _patch STREQUAL "")
    # Newer LLVM revisions may already contain the patch.
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" apply --reverse --check "${PATCH}"
        RESULT_VARIABLE _patch_present
        OUTPUT_QUIET
        ERROR_QUIET)
    if(NOT _patch_present EQUAL 0)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" apply "${PATCH}"
            COMMAND_ERROR_IS_FATAL ANY)
        file(WRITE "${_applied_patch}" "${_patch}")
    endif()
endif()
