#pragma once

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <iostream>

#include "DGLHackKernel/RGNN/mysgemm_KernelsBlockConfigurations.h"

void try_get_schedule_by_relations(int64_t num_relations, int64_t num_blocks) {
  std::vector<int64_t> mock_job_entries_for_all_prev_relation_vec(
      num_relations + 1, 1000);
  auto [num_blocks_along_dimx_for_same_relation_vect,
        num_blocks_along_dimx_for_all_prev_relation_vect] =
      get_schedule_by_relation_kernel_launch_metadata<
          false, false, std::vector<int64_t>::iterator>(
          num_relations, num_blocks, -1,
          mock_job_entries_for_all_prev_relation_vec.begin(),
          mock_job_entries_for_all_prev_relation_vec.end());

  // std::vector<int> mock_vect(100, 100);
  // std::vector<int> mock_vect2(100, 100);
  // thrust::device_vector<int>
  // dev_num_blocks_along_dimx_for_same_relation_vect(100);
  // thrust::device_vector<int>
  // dev_num_blocks_along_dimx_for_all_prev_relation_vect(100); for (int idx =
  // 0; idx < 100; idx++){
  //     dev_num_blocks_along_dimx_for_same_relation_vect[idx] = mock_vect[idx];
  //     dev_num_blocks_along_dimx_for_all_prev_relation_vect[idx] =
  //     mock_vect2[idx];
  // }
  // thrust::device_vector<int>
  // dev_num_blocks_along_dimx_for_same_relation_vect(mock_vect.begin(),
  // mock_vect.end()); thrust::device_vector<int>
  // dev_num_blocks_along_dimx_for_all_prev_relation_vect(mock_vect2.begin(),
  // mock_vect2.end());
  thrust::device_vector<int> dev_num_blocks_along_dimx_for_same_relation_vect(
      num_blocks_along_dimx_for_same_relation_vect.begin(),
      num_blocks_along_dimx_for_same_relation_vect.end());
  thrust::device_vector<int>
      dev_num_blocks_along_dimx_for_all_prev_relation_vect(
          num_blocks_along_dimx_for_all_prev_relation_vect.begin(),
          num_blocks_along_dimx_for_all_prev_relation_vect.end());
  return;
}

#ifndef GIT_COMMIT_HASH
#define GIT_COMMIT_HASH "?"
#endif

void build_debug_info() {
  std::cout << "GIT_COMMIT_HASH: " << GIT_COMMIT_HASH << std::endl;
#ifdef ENABLE_DEBUG_MACRO
  std::cout << "WARNING: library built in debug mode without -O3" << std::endl;
#else
  std::cout << "library built in release mode with -O3" << std::endl;
#endif
  std::cout << "library compiled by gcc " << __GNUC__ << "." << __GNUC_MINOR__
            << "." << __GNUC_PATCHLEVEL__ << ", nvcc " << __CUDACC_VER_MAJOR__
            << "." << __CUDACC_VER_MINOR__ << "." << __CUDACC_VER_BUILD__
            << std::endl;
}

std::vector<std::vector<at::Tensor>> biops_tensor_info(
    at::Tensor &one_tensor, at::Tensor &other_tensor) {
  std::cout << "one_tensor device: " << one_tensor.device() << std::endl;
  std::cout << "other_tensor device: " << other_tensor.device() << std::endl;

  std::vector<std::vector<at::Tensor>> result = {
      {one_tensor.clone()}, {one_tensor.clone(), other_tensor.clone()}};
  return result;
}

at::Tensor tensor_info(at::Tensor &one_tensor) {
  // NB: storage_offset does play a role in tensor metadata, see in
  // github/pytorch/pytorch repo, pytorch/pytorch/c10/core/TensorImpl.h
  // implements `inline T* data_ptr_impl() const` as `return
  // storage_.unsafe_data<T>() + storage_offset_;`. Notice that storage_offset
  // count in number of elements, not bytes.
  std::cout << "one_tensor device: " << one_tensor.device() << std::endl;
  std::cout << "one_tensor dtype: " << one_tensor.dtype() << std::endl;
  std::cout << "one_tensor ndim: " << one_tensor.dim() << std::endl;
  std::cout << "one_tensor shape: " << one_tensor.sizes() << std::endl;
  std::cout << "one_tensor numel: " << one_tensor.numel() << std::endl;
  std::cout << "one_tensor nbytes: " << one_tensor.nbytes() << std::endl;
  std::cout << "one_tensor storage_offset: " << one_tensor.storage_offset()
            << std::endl;
  std::cout << "one_tensor itemsize: " << one_tensor.element_size()
            << std::endl;
  return one_tensor.clone();
}

torch::Dict<std::string, int64_t> test_argument_takein(
    std::string str, bool flag, torch::Dict<std::string, int64_t> dictionary) {
  std::cout << "test_string_takein: " << str << std::endl;
  std::cout << "test_bool_takein: " << flag << std::endl;
  std::cout << "test_dict_takein: " << dictionary.at("key1") << std::endl;
  torch::Dict<std::string, int64_t> result;
  result.insert("key1", dictionary.at("key1") + 1);
  result.insert("flag", flag);
  return result;
}

static void printTensor(at::Tensor &oneTensor) {
  int x_dim = oneTensor.sizes()[0];
  int y_dim = oneTensor.sizes()[1];

  for (int i = 0; i < x_dim; i++) {
    for (int j = 0; j < y_dim; j++) {
      auto accessor = oneTensor.accessor<float, 2>();
      float element = accessor[i][j];
      printf("%f ", element);
    }

    printf("\n");
  }
}

__global__ static void simplified_rectangular_basic_MatMulKernel(
    float *A, float *B, float *C, int A_width, int A_height, int B_width) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int C_height = A_height;
  int C_width = B_width;
  const int SHMEM_BLOCK_SIZE = 2;
  int Row = by * SHMEM_BLOCK_SIZE + ty;
  int Col = bx * SHMEM_BLOCK_SIZE + tx;

  // Create submatrix Asub and Bsub in shared memory
  __shared__ float As[SHMEM_BLOCK_SIZE][SHMEM_BLOCK_SIZE];
  __shared__ float Bs[SHMEM_BLOCK_SIZE][SHMEM_BLOCK_SIZE];

  for (int q = 0; q < ((A_width - 1) / SHMEM_BLOCK_SIZE) + 1; q++) {
    float Cvalue = 0.0;

    // Load matrix
    // For As
    if (Row < C_height && q * SHMEM_BLOCK_SIZE + tx < C_width) {
      // If in range, load the proper element
      As[ty][tx] = A[Row * A_width + q * SHMEM_BLOCK_SIZE + tx];

    } else {
      // Else, fill 0
      As[ty][tx] = 0;
    }

    // For Bs
    if (q * SHMEM_BLOCK_SIZE + ty < C_height && Col < C_width) {
      // If in range, load the proper element
      Bs[ty][tx] = B[(q * SHMEM_BLOCK_SIZE + ty) * B_width + Col];

    } else {
      // Else, fill 0
      Bs[ty][tx] = 0;
    }

    // Synchronize to make sure all sub-matrices are loaded
    __syncthreads();

    // Matrix Sum Code
    // Also, Check if in range, if not, do nothing
    if (Row < C_height && Col < C_width) {
      for (int e = 0; e < SHMEM_BLOCK_SIZE; e++) {
        Cvalue += As[ty][e] * Bs[e][tx];
      }

      printf("bx = %d, by = %d, tx = %d, ty = %d, Cvalue = %f \n", bx, by, tx,
             ty, Cvalue);
      C[Row * C_width + Col] = Cvalue;
    }

    __syncthreads();
    // Synchronize to make sure all matrix mul are completed
  }
}

static void rectangular_MatMul(at::Tensor &A, at::Tensor &B, at::Tensor &C) {
  // Print input tensors
  printf("Tensor A = \n");
  printTensor(A);
  printf("Tensor B = \n");
  printTensor(B);

  // load tensors into arrays
  float *Array_A = A.data_ptr<float>();
  float *Array_B = B.data_ptr<float>();
  float *Array_C = C.data_ptr<float>();

  // Get Array sizes
  int A_height = A.sizes()[0];
  int A_width = A.sizes()[1];
  int A_size_in_byte = A_width * A_height * sizeof(float);
  int B_height = B.sizes()[0];
  int B_width = B.sizes()[1];
  int B_size_in_byte = B_width * B_height * sizeof(float);
  int C_size_in_byte = A_height * B_width * sizeof(float);
  const int SHMEM_BLOCK_SIZE = 2;

  // Printing sizes
  printf("A_height = C_height = %d \n", A_height);
  printf("B_width = C_width = %d \n", B_width);

  // Initialize device arrays
  float *Array_A_device = nullptr;
  float *Array_B_device = nullptr;
  float *Array_C_device = nullptr;

  // Allocate Memory with error return
  cudaError_t err = cudaMalloc((void **)&Array_A_device, A_size_in_byte);
  printf("CUDA malloc A: %s\n", cudaGetErrorString(err));

  err = cudaMalloc((void **)&Array_B_device, B_size_in_byte);
  printf("CUDA malloc B: %s\n", cudaGetErrorString(err));

  err = cudaMalloc((void **)&Array_C_device, C_size_in_byte);
  printf("CUDA malloc C: %s\n", cudaGetErrorString(err));

  // Copy Memory for A and B
  err = cudaMemcpy(Array_A_device, Array_A, A_size_in_byte,
                   cudaMemcpyHostToDevice);
  printf("CUDA copy A to device: %s\n", cudaGetErrorString(err));

  err = cudaMemcpy(Array_B_device, Array_B, B_size_in_byte,
                   cudaMemcpyHostToDevice);
  printf("CUDA copy B to device: %s\n", cudaGetErrorString(err));

  // Define block and grid size
  // the (size - 1)/SHMEM_BLOCK_SIZE is a cheap way of doing ceil()
  dim3 DimGrid((B_width - 1) / SHMEM_BLOCK_SIZE + 1,
               ((A_height - 1) / SHMEM_BLOCK_SIZE) + 1, 1);
  dim3 DimBlock(SHMEM_BLOCK_SIZE, SHMEM_BLOCK_SIZE, 1);

  // Run Kernel
  simplified_rectangular_basic_MatMulKernel<<<DimGrid, DimBlock>>>(
      Array_A_device, Array_B_device, Array_C_device, A_width, A_height,
      B_width);

  // Kernel Error Detection
  err = cudaDeviceSynchronize();
  printf("Run kernel: %s\n", cudaGetErrorString(err));

  err = cudaMemcpy(Array_C, Array_C_device, C_size_in_byte,
                   cudaMemcpyDeviceToHost);
  printf("Array_C Pointer Value: %lu\n", (unsigned long)Array_C);
  printf("Array_C_device Pointer Value: %lu\n", (unsigned long)Array_C_device);
  printf("Copy C off of device: %s\n", cudaGetErrorString(err));

  // Resize C to make sure it's right
  C.resize_({A_height, B_width});

  // Free Memory
  cudaFree(Array_A_device);
  cudaFree(Array_B_device);
  cudaFree(Array_C_device);
}

TORCH_LIBRARY_FRAGMENT(torch_hetero_edgesoftmax, m) {
  // Utility and debugging functions
  m.def("build_debug_info", build_debug_info);
  m.def("try_get_schedule_by_relations", try_get_schedule_by_relations);
  m.def("biops_tensor_info", biops_tensor_info);
  m.def("tensor_info", tensor_info);
  m.def("test_argument_takein", test_argument_takein);
  m.def("rectangular_MatMul ", rectangular_MatMul);
  m.def("printTensor", printTensor);
}
