#pragma once

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>

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

std::vector<std::vector<at::Tensor>> biops_tensor_info(
    at::Tensor &one_tensor, at::Tensor &other_tensor) {
  std::cout << "one_tensor device: " << one_tensor.device() << std::endl;
  std::cout << "other_tensor device: " << other_tensor.device() << std::endl;
  std::cout << "one_tensor dtype: " << one_tensor.dtype() << std::endl;
  std::cout << "other_tensor dtype: " << other_tensor.dtype() << std::endl;
  std::cout << "one_tensor dataptr" << one_tensor.data_ptr() << std::endl;
  std::cout << "other_tensor dataptr" << other_tensor.data_ptr() << std::endl;
  std::vector<std::vector<at::Tensor>> result = {
      {one_tensor.clone()}, {one_tensor.clone(), other_tensor.clone()}};
  return result;
}

at::Tensor tensor_info(const at::Tensor &one_tensor) {
  // NB: storage_offset does play a role in tensor metadata, see in
  // github/pytorch/pytorch repo, pytorch/pytorch/c10/core/TensorImpl.h
  // implements `inline T* data_ptr_impl() const` as `return
  // storage_.unsafe_data<T>() + storage_offset_;`. Notice that storage_offset
  // count in number of elements, not bytes.
  // NB: unlike data_ptr(), data_ptr<dtype> will trigger error if the tensor is
  // not of the dtype.
  std::cout << "one_tensor dataptr" << one_tensor.data_ptr() << std::endl;
  std::cout << "one_tensor device: " << one_tensor.device() << std::endl;
  std::cout << "one_tensor dtype: " << one_tensor.dtype() << std::endl;
  // print is_floating_point_type is_integral_type size
  std::cout << "one_tensor is_floating_point: "
            << one_tensor.is_floating_point() << std::endl;
  std::cout << "one_tensor is_integral: "
            << at::isIntegralType(one_tensor.scalar_type()) << std::endl;
  std::cout << "one_tensor is_signed: " << one_tensor.is_signed() << std::endl;
  std::cout << "one_tensor is_complex: " << one_tensor.is_complex()
            << std::endl;
  std::cout << "one_tensor scalar type: "
            << at::elementSize(one_tensor.scalar_type()) << std::endl;

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

void print_tensor_dict_info(torch::Dict<std::string, at::Tensor> dictionary) {
  // print key and tensor info for each tensor in the dictionary
  for (auto const &pair : dictionary) {
    std::cout << "key: " << pair.key() << std::endl;
    std::cout << "tensor info: " << std::endl;
    tensor_info(dictionary.at(pair.key()));
  }
}

TORCH_LIBRARY_FRAGMENT(torch_hetero_edgesoftmax, m) {
  // Utility and debugging functions
  m.def("try_get_schedule_by_relations", try_get_schedule_by_relations);
  m.def("biops_tensor_info", biops_tensor_info);
  m.def("tensor_info", tensor_info);
  m.def("test_argument_takein", test_argument_takein);
  m.def("print_tensor_dict_info", print_tensor_dict_info);
}
