#pragma once

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "DGLHackKernel/mysgemm/mysgemm_KernelsBlockConfigurations.h"

void try_get_schedule_by_relations(int64_t num_relations, int64_t num_blocks) {
  std::vector<int64_t> mock_job_entries_per_relation_vec(num_relations, 1000);
  auto [num_blocks_along_dimx_for_same_relation_vect,
        num_blocks_along_dimx_for_all_prev_relation_vect] =
      get_schedule_by_relation_kernel_launch_metadata<
          false, false, std::vector<int64_t>::iterator>(
          num_relations, num_blocks, -1,
          mock_job_entries_per_relation_vec.begin(),
          mock_job_entries_per_relation_vec.end());

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
    at::Tensor& one_tensor, at::Tensor& other_tensor) {
  std::cout << "one_tensor device: " << one_tensor.device() << std::endl;
  std::cout << "other_tensor device: " << other_tensor.device() << std::endl;

  std::vector<std::vector<at::Tensor>> result = {
      {one_tensor.clone()}, {one_tensor.clone(), other_tensor.clone()}};
  return result;
}

at::Tensor tensor_info(at::Tensor& one_tensor) {
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