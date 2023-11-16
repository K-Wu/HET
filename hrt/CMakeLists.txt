# From sputnik/sputnik/CMakeLists.txt For use in subdirs.
set(HETEROEDGESOFTMAX_SRCS)
set(HETEROEDGESOFTMAX_INCLUDE_FILES)
add_subdirectory(include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Get git status from https://stackoverflow.com/a/63924044/5555077 Get the
# current working branch execute_process only runs during cmake configure time;
# what we want is to run it during cmake build time by add_custom_target
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# Get the latest commit hash
execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND git diff HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_DIRTY_MESSAGE OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT GIT_DIRTY_MESSAGE STREQUAL "")
  set(GIT_DIRTY_MESSAGE "-dirty")
endif()

# From
# https://stackoverflow.com/questions/73069159/use-git-hash-in-building-with-cmake-from-c-source
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/generated/include/generated/gitsha.h
  COMMAND mkdir -p ${CMAKE_BINARY_DIR}/generated/include/generated
  COMMAND
    echo -n
    "#pragma once\\nconst char* git_version = \"${GIT_BRANCH}/${GIT_COMMIT_HASH}${GIT_DIRTY_MESSAGE}\";"
    > ${CMAKE_BINARY_DIR}/generated/include/gitsha.h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Generating gitsha.h"
  VERBATIM)

add_custom_target(
  home ALL
  DEPENDS ${CMAKE_BINARY_DIR}/generated/include/generated/DebugInfo.inc.h
          ${CMAKE_BINARY_DIR}/generated/include/generated/gitsha.h
          ${CMAKE_BINARY_DIR}/generated/include/CMakeLists.txt
          ${CMAKE_BINARY_DIR}/generated/src/exported_op.cu.cc)

# Generate header files by calling buildutils/genutils/gen_headers.py and
# outputing to build/generated/include/generated.inc.h
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/generated/include/generated/DebugInfo.inc.h
  COMMAND mkdir -p ${CMAKE_BINARY_DIR}/generated/include/generated
  COMMAND
    ${PYTHON_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/hrt/buildutils/genutils/gen_headers.py
    --gen_debug_info >
    ${CMAKE_BINARY_DIR}/generated/include/generated/DebugInfo.inc.h
  COMMENT "Generating DebugInfo.inc.h"
  VERBATIM)
# Copy include/CMakeLists.txt to build/generated/include/CMakeLists.txt
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/generated/include/CMakeLists.txt
  COMMAND
    ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/include/CMakeLists.txt
    ${CMAKE_BINARY_DIR}/generated/include/CMakeLists.txt
  COMMENT
    "Copying include/CMakeLists.txt to build/generated/include/CMakeLists.txt"
  VERBATIM)
include_directories(${CMAKE_BINARY_DIR}/generated/include)

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/generated/src/exported_op.cu.cc
  COMMAND mkdir -p ${CMAKE_BINARY_DIR}/generated/src
  COMMAND
    ${PYTHON_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/hrt/buildutils/genutils/gen_torch_export.py
    ${PROJECT_SOURCE_DIR} > ${CMAKE_BINARY_DIR}/generated/src/exported_op.cu.cc
  COMMENT "Generating torch export definitions"
  VERBATIM)

# Set .cu.cc files to be compiled as CUDA.
#
set(HETEROEDGESOFTMAX_CUDA_SRCS ${HETEROEDGESOFTMAX_SRCS})
list(FILTER HETEROEDGESOFTMAX_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
set_source_files_properties(${HETEROEDGESOFTMAX_CUDA_SRCS} PROPERTIES LANGUAGE
                                                                      CUDA)

# Build the .cu.cc files in the src directory.
#
# Build kernel.cu.cc
set(HETEROEDGESOFTMAX_KERNEL_CU_CC_SRCS)
list(APPEND HETEROEDGESOFTMAX_KERNEL_CU_CC_SRCS
     ${HETEROEDGESOFTMAX_INCLUDE_FILES})
list(APPEND HETEROEDGESOFTMAX_KERNEL_CU_CC_SRCS "src/kernel.cu.cc")
set(HETEROEDGESOFTMAX_KERNEL_CU_CC_CUDA_SRCS
    ${HETEROEDGESOFTMAX_KERNEL_CU_CC_SRCS})
list(FILTER HETEROEDGESOFTMAX_KERNEL_CU_CC_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
set_source_files_properties(${HETEROEDGESOFTMAX_KERNEL_CU_CC_CUDA_SRCS}
                            PROPERTIES LANGUAGE CUDA)
add_executable(kernel_cu_cc ${HETEROEDGESOFTMAX_KERNEL_CU_CC_SRCS})
target_link_libraries(kernel_cu_cc ${HETEROEDGESOFTMAX_LIBS})

# Build test_compiler_discover_sputnik_library.cu.cc
set(HETEROEDGESOFTMAX_TEST_SPUTNIK_CU_CC_SRCS)
# We need to involve test utilities sources from sputnik repo.
set(SPUTNIK_TEST_SRCS)
file(GLOB TMP ${PROJECT_SOURCE_DIR}/third_party/sputnik/sputnik/matrix_utils*)
list(APPEND SPUTNIK_TEST_SRCS ${TMP})
file(GLOB TMP ${PROJECT_SOURCE_DIR}/third_party/sputnik/sputnik/test_utils*)
list(APPEND SPUTNIK_TEST_SRCS ${TMP})
list(APPEND HETEROEDGESOFTMAX_TEST_SPUTNIK_CU_CC_SRCS ${SPUTNIK_TEST_SRCS})
list(APPEND HETEROEDGESOFTMAX_TEST_SPUTNIK_CU_CC_SRCS
     ${HETEROEDGESOFTMAX_INCLUDE_FILES})
list(APPEND HETEROEDGESOFTMAX_TEST_SPUTNIK_CU_CC_SRCS
     "src/e2e_cases/test_compiler_discover_sputnik_library.cu.cc")
set(HETEROEDGESOFTMAX_TEST_SPUTNIK_CU_CC_CUDA_SRCS
    ${HETEROEDGESOFTMAX_TEST_SPUTNIK_CU_CC_SRCS})
list(FILTER HETEROEDGESOFTMAX_TEST_SPUTNIK_CU_CC_CUDA_SRCS INCLUDE REGEX
     "\.cu\.cc")
set_source_files_properties(${HETEROEDGESOFTMAX_TEST_SPUTNIK_CU_CC_CUDA_SRCS}
                            PROPERTIES LANGUAGE CUDA)
add_executable(test_compiler_discover_sputnik_library_cu_cc
               ${HETEROEDGESOFTMAX_TEST_SPUTNIK_CU_CC_SRCS})
# We need to link it to libraries involving gtest, abseil, and sputnik. Thus we
# have a dedicated new variable HETEROEDGESOFTMAX_TEST_LIBS here
target_link_libraries(test_compiler_discover_sputnik_library_cu_cc
                      ${HETEROEDGESOFTMAX_TEST_LIBS})

# Build test_compiler_discover_cutlass_library.cu.cc
set(HETEROEDGESOFTMAX_TEST_CUTLASS_CU_CC_SRCS)
list(APPEND HETEROEDGESOFTMAX_TEST_CUTLASS_CU_CC_SRCS
     ${HETEROEDGESOFTMAX_INCLUDE_FILES})
list(APPEND HETEROEDGESOFTMAX_TEST_CUTLASS_CU_CC_SRCS
     "src/test_compiler_discover_cutlass_library.cu.cc")
set(HETEROEDGESOFTMAX_TEST_CUTLASS_CU_CC_CUDA_SRCS
    ${HETEROEDGESOFTMAX_TEST_CUTLASS_CU_CC_SRCS})
list(FILTER HETEROEDGESOFTMAX_TEST_CUTLASS_CU_CC_CUDA_SRCS INCLUDE REGEX
     "\.cu\.cc")
set_source_files_properties(${HETEROEDGESOFTMAX_TEST_CUTLASS_CU_CC_CUDA_SRCS}
                            PROPERTIES LANGUAGE CUDA)
add_executable(test_compiler_discover_cutlass_library_cu_cc
               ${HETEROEDGESOFTMAX_TEST_CUTLASS_CU_CC_SRCS})
target_link_libraries(test_compiler_discover_cutlass_library_cu_cc
                      ${HETEROEDGESOFTMAX_LIBS})

# Build test_compiler_discover_cublas.cu.cc
set(HETEROEDGESOFTMAX_TEST_CUBLAS_CU_CC_SRCS)
list(APPEND HETEROEDGESOFTMAX_TEST_CUBLAS_CU_CC_SRCS
     ${HETEROEDGESOFTMAX_INCLUDE_FILES})
list(APPEND HETEROEDGESOFTMAX_TEST_CUBLAS_CU_CC_SRCS
     "src/test_compiler_discover_cublas.cu.cc")
set(HETEROEDGESOFTMAX_TEST_CUBLAS_CU_CC_CUDA_SRCS
    ${HETEROEDGESOFTMAX_TEST_CUBLAS_CU_CC_SRCS})
list(FILTER HETEROEDGESOFTMAX_TEST_CUBLAS_CU_CC_CUDA_SRCS INCLUDE REGEX
     "\.cu\.cc")
set_source_files_properties(${HETEROEDGESOFTMAX_TEST_CUBLAS_CU_CC_CUDA_SRCS}
                            PROPERTIES LANGUAGE CUDA)
add_executable(test_compiler_discover_cublas_cu_cc
               ${HETEROEDGESOFTMAX_TEST_CUBLAS_CU_CC_SRCS})
target_link_libraries(test_compiler_discover_cublas_cu_cc
                      ${HETEROEDGESOFTMAX_LIBS})

# Build test_compiler_discover_cublaslt.cu.cc
set(HETEROEDGESOFTMAX_TEST_CUBLASLT_CU_CC_SRCS)
list(APPEND HETEROEDGESOFTMAX_TEST_CUBLASLT_CU_CC_SRCS
     ${HETEROEDGESOFTMAX_INCLUDE_FILES})
list(APPEND HETEROEDGESOFTMAX_TEST_CUBLASLT_CU_CC_SRCS
     "src/test_compiler_discover_cublaslt.cu.cc")
set(HETEROEDGESOFTMAX_TEST_CUBLASLT_CU_CC_CUDA_SRCS
    ${HETEROEDGESOFTMAX_TEST_CUBLASLT_CU_CC_SRCS})
list(FILTER HETEROEDGESOFTMAX_TEST_CUBLASLT_CU_CC_CUDA_SRCS INCLUDE REGEX
     "\.cu\.cc")
set_source_files_properties(${HETEROEDGESOFTMAX_TEST_CUBLASLT_CU_CC_CUDA_SRCS}
                            PROPERTIES LANGUAGE CUDA)
add_executable(test_compiler_discover_cublaslt_cu_cc
               ${HETEROEDGESOFTMAX_TEST_CUBLASLT_CU_CC_SRCS})
target_link_libraries(test_compiler_discover_cublaslt_cu_cc
                      ${HETEROEDGESOFTMAX_LIBS})

# Build test_DGLHackHGTBackwardFusion.cu.cc
set(HETEROEDGESOFTMAX_TEST_DGLHACKHGTBACKWARDFUSION_CU_CC_SRCS)
list(APPEND HETEROEDGESOFTMAX_TEST_DGLHACKHGTBACKWARDFUSION_CU_CC_SRCS
     ${HETEROEDGESOFTMAX_INCLUDE_FILES})
list(APPEND HETEROEDGESOFTMAX_TEST_DGLHACKHGTBACKWARDFUSION_CU_CC_SRCS
     "src/test_DGLHackHGTBackwardFusion.cu.cc")
set(HETEROEDGESOFTMAX_TEST_DGLHACKHGTBACKWARDFUSION_CU_CC_CUDA_SRCS
    ${HETEROEDGESOFTMAX_TEST_DGLHACKHGTBACKWARDFUSION_CU_CC_SRCS})
list(FILTER HETEROEDGESOFTMAX_TEST_DGLHACKHGTBACKWARDFUSION_CU_CC_CUDA_SRCS
     INCLUDE REGEX "\.cu\.cc")
set_source_files_properties(
  ${HETEROEDGESOFTMAX_TEST_DGLHACKHGTBACKWARDFUSION_CU_CC_CUDA_SRCS}
  PROPERTIES LANGUAGE CUDA)
add_executable(test_DGLHackHGTBackwardFusion_cu_cc
               ${HETEROEDGESOFTMAX_TEST_DGLHACKHGTBACKWARDFUSION_CU_CC_SRCS})
target_link_libraries(test_DGLHackHGTBackwardFusion_cu_cc
                      ${HETEROEDGESOFTMAX_LIBS})

# Build test_DGLHackKernel.cu.cc
set(HETEROEDGESOFTMAX_TEST_DGLHACKKERNEL_CU_CC_SRCS)
list(APPEND HETEROEDGESOFTMAX_TEST_DGLHACKKERNEL_CU_CC_SRCS
     ${HETEROEDGESOFTMAX_INCLUDE_FILES})
list(APPEND HETEROEDGESOFTMAX_TEST_DGLHACKKERNEL_CU_CC_SRCS
     "src/test_DGLHackKernel.cu.cc")
set(HETEROEDGESOFTMAX_TEST_DGLHACKKERNEL_CU_CC_CUDA_SRCS
    ${HETEROEDGESOFTMAX_TEST_DGLHACKKERNEL_CU_CC_SRCS})
list(FILTER HETEROEDGESOFTMAX_TEST_DGLHACKKERNEL_CU_CC_CUDA_SRCS INCLUDE REGEX
     "\.cu\.cc")
set_source_files_properties(
  ${HETEROEDGESOFTMAX_TEST_DGLHACKKERNEL_CU_CC_CUDA_SRCS} PROPERTIES LANGUAGE
                                                                     CUDA)
add_executable(test_DGLHackKernel_cu_cc
               ${HETEROEDGESOFTMAX_TEST_DGLHACKKERNEL_CU_CC_SRCS})
target_link_libraries(test_DGLHackKernel_cu_cc ${HETEROEDGESOFTMAX_LIBS})

# Build test_DGLHackSimpleLoadBalance.cu.cc
set(HETEROEDGESOFTMAX_TEST_DGLHACKKERNEL_SIMPLELOADBALANCE_CU_CC_SRCS)
list(APPEND HETEROEDGESOFTMAX_TEST_DGLHACKKERNEL_SIMPLELOADBALANCE_CU_CC_SRCS
     ${HETEROEDGESOFTMAX_INCLUDE_FILES})
list(APPEND HETEROEDGESOFTMAX_TEST_DGLHACKKERNEL_SIMPLELOADBALANCE_CU_CC_SRCS
     "src/test_DGLHackSimpleLoadBalance.cu.cc")
set(HETEROEDGESOFTMAX_TEST_DGLHACKKERNEL_SIMPLELOADBALANCE_CU_CC_CUDA_SRCS
    ${HETEROEDGESOFTMAX_TEST_DGLHACKKERNEL_SIMPLELOADBALANCE_CU_CC_SRCS})
list(FILTER
     HETEROEDGESOFTMAX_TEST_DGLHACKKERNEL_SIMPLELOADBALANCE_CU_CC_CUDA_SRCS
     INCLUDE REGEX "\.cu\.cc")
set_source_files_properties(
  ${HETEROEDGESOFTMAX_TEST_DGLHACKKERNEL_SIMPLELOADBALANCE_CU_CC_CUDA_SRCS}
  PROPERTIES LANGUAGE CUDA)
add_executable(
  test_DGLHackSimpleLoadBalance_cu_cc
  ${HETEROEDGESOFTMAX_TEST_DGLHACKKERNEL_SIMPLELOADBALANCE_CU_CC_SRCS})
target_link_libraries(test_DGLHackSimpleLoadBalance_cu_cc
                      ${HETEROEDGESOFTMAX_LIBS})

# Build test_hyb.cu.cc
set(HETEROEDGESOFTMAX_TEST_HYB_CU_CC_SRCS)
list(APPEND HETEROEDGESOFTMAX_TEST_HYB_CU_CC_SRCS
     ${HETEROEDGESOFTMAX_INCLUDE_FILES})
list(APPEND HETEROEDGESOFTMAX_TEST_HYB_CU_CC_SRCS "src/test_hyb.cu.cc")
set(HETEROEDGESOFTMAX_TEST_HYB_CU_CC_CUDA_SRCS
    ${HETEROEDGESOFTMAX_TEST_HYB_CU_CC_SRCS})
list(FILTER HETEROEDGESOFTMAX_TEST_HYB_CU_CC_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
set_source_files_properties(${HETEROEDGESOFTMAX_TEST_HYB_CU_CC_CUDA_SRCS}
                            PROPERTIES LANGUAGE CUDA)
add_executable(test_hyb_cu_cc ${HETEROEDGESOFTMAX_TEST_HYB_CU_CC_SRCS})
# TODO: Clean up this hacky code
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${CMAKE_BINARY_DIR}/sputnik)
target_link_libraries(test_hyb_cu_cc ${HETEROEDGESOFTMAX_LIBS})

# Build test_ndarray_utils.cu.cc
set(HETEROEDGESOFTMAX_TEST_NDARRAY_UTILS_CU_CC_SRCS)
list(APPEND HETEROEDGESOFTMAX_TEST_NDARRAY_UTILS_CU_CC_SRCS
     ${HETEROEDGESOFTMAX_INCLUDE_FILES})
list(APPEND HETEROEDGESOFTMAX_TEST_NDARRAY_UTILS_CU_CC_SRCS
     "src/test_ndarray_utils.cu.cc")
set(HETEROEDGESOFTMAX_TEST_NDARRAY_UTILS_CU_CC_CUDA_SRCS
    ${HETEROEDGESOFTMAX_TEST_NDARRAY_UTILS_CU_CC_SRCS})
list(FILTER HETEROEDGESOFTMAX_TEST_NDARRAY_UTILS_CU_CC_CUDA_SRCS INCLUDE REGEX
     "\.cu\.cc")
set_source_files_properties(
  ${HETEROEDGESOFTMAX_TEST_NDARRAY_UTILS_CU_CC_CUDA_SRCS} PROPERTIES LANGUAGE
                                                                     CUDA)
add_executable(test_ndarray_utils_cu_cc
               ${HETEROEDGESOFTMAX_TEST_NDARRAY_UTILS_CU_CC_SRCS})
# TODO: Clean up this hacky code
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${CMAKE_BINARY_DIR}/sputnik)
target_link_libraries(test_ndarray_utils_cu_cc ${HETEROEDGESOFTMAX_LIBS})

# Build exported to pytorch partially from
# https://stackoverflow.com/questions/68401650/how-can-i-make-a-pytorch-extension-with-cmake
if(BUILD_TORCH_API)
  set(TORCH_HETEROEDGESOFTMAX_EXPORT_CU_CC_SRCS)
  list(APPEND TORCH_HETEROEDGESOFTMAX_EXPORT_CU_CC_SRCS
       ${HETEROEDGESOFTMAX_INCLUDE_FILES})
  list(APPEND TORCH_HETEROEDGESOFTMAX_EXPORT_CU_CC_SRCS
       ${CMAKE_BINARY_DIR}/generated/src/exported_op.cu.cc)
  set(TORCH_HETEROEDGESOFTMAX_EXPORT_CU_CC_CUDA_SRCS
      ${TORCH_HETEROEDGESOFTMAX_EXPORT_CU_CC_SRCS})
  list(FILTER TORCH_HETEROEDGESOFTMAX_EXPORT_CU_CC_CUDA_SRCS INCLUDE REGEX
       "\.cu\.cc")
  set_source_files_properties(${TORCH_HETEROEDGESOFTMAX_EXPORT_CU_CC_CUDA_SRCS}
                              PROPERTIES LANGUAGE CUDA)
  add_library(torch_hrt SHARED ${TORCH_HETEROEDGESOFTMAX_EXPORT_CU_CC_SRCS})
  target_compile_features(torch_hrt PRIVATE cxx_std_14)
  target_link_libraries(torch_hrt ${HETEROEDGESOFTMAX_LIBS} ${TORCH_LIBRARIES}
                        Python::Python)
  # Use if the default GCC version gives issues. Similar syntax is used if we
  # need better compilation flags.
  target_compile_options(torch_hrt PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-ccbin
                                           g++-9>)
endif()
