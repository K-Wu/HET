# From sputnik/cmake/Dependencies.cmake
# NB: add cusp and libnpy. Both of them are header only 3-rd party libraries and thus can be simply added by include_directories() command.
include(cmake/Cuda.cmake)

# NB: added curand
# NB: removed cuda_find_library invocations that create unused variables
# TODO(tgale): Move cuSPARSE, cuBLAS deps to test & benchmark only.
#cuda_find_library(CUDART_LIBRARY cudart_static)
#cuda_find_library(CUBLAS_LIBRARY cublas_static)
#cuda_find_library(CUSPARSE_LIBRARY cusparse_static)
list(APPEND HETEROEDGESOFTMAX_LIBS "cudart_static;cublas_static;cusparse_static;cublasLt_static;culibos")
list(APPEND HETEROEDGESOFTMAX_LIBS "curand")

# Google Glog.
find_package(Glog REQUIRED)
list(APPEND HETEROEDGESOFTMAX_LIBS ${GLOG_LIBRARIES})

# Header-only libraries cusp and libnpy.
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/cusplibrary)
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/libnpy/include)

# Header-only libraries CUTLASS.
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/cutlass/include)
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/cutlass/examples/common)
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/cutlass/tools/util/include)


# Google-Research Sputnik.
add_subdirectory(third_party/sputnik)
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/sputnik)

# add abseil and gtest for building the test sputnik discovery executable source file 
# Google Abseil.
# add_subdirectory(third_party/sputnik/third_party/abseil-cpp)
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/abseil-cpp)

# Google Test and Google Mock.
# add_subdirectory(third_party/sputnik/third_party/googletest)
# set(BUILD_GTEST ON CACHE INTERNAL "Build gtest submodule.")
# set(BUILD_GMOCK ON CACHE INTERNAL "Build gmock submodule.")
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/sputnik/third_party/googletest/googletest/include)
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/sputnik/third_party/googletest/googlemock/include)


list(APPEND HETEROEDGESOFTMAX_TEST_LIBS "gtest_main;gmock;absl::random_random")
# we also need to refer to sputnik library in the test example.
list(APPEND HETEROEDGESOFTMAX_TEST_LIBS "sputnik")
