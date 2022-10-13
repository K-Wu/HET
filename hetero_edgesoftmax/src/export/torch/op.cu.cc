// from
// https://stackoverflow.com/questions/68401650/how-can-i-make-a-pytorch-extension-with-cmake
// kernels defined in this file are simply wrapped up in
// hetero_edgesoftmax/python/kernels.py to provide python API, then used to
// define autograd functions and layers in hetero_edgesoftmax/python/kernels.py,
// which is finally referred to by end2end cases in
// hetero_edgesoftmax/python/<model name>/.*.py
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <iostream>
#include "DGLHackKernel/DGLHackKernel.h"
#include "hetero_edgesoftmax.h"

at::Tensor biops_tensor_info(at::Tensor& one_tensor, at::Tensor& other_tensor) {
  std::cout << "one_tensor device: " << one_tensor.device() << std::endl;
  std::cout << "other_tensor device: " << other_tensor.device() << std::endl;
  return one_tensor.clone();
}

at::Tensor tensor_info(at::Tensor& one_tensor) {
  // NB: storage_offset does play a role in tensor metadata, see in
  // github/pytorch/pytorch repo, pytorch/pytorch/c10/core/TensorImpl.h
  // implements `inline T* data_ptr_impl() const` as `return
  // storage_.unsafe_data<T>() + storage_offset_;`. Notice that storage_offset
  // count in number of elements, not bytes.
  std::cout << "library compiled by gcc " << __GNUC__ << "." << __GNUC_MINOR__
            << "." << __GNUC_PATCHLEVEL__ << ", nvcc " << __CUDACC_VER_MAJOR__
            << "." << __CUDACC_VER_MINOR__ << "." << __CUDACC_VER_BUILD__
            << std::endl;
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

TORCH_LIBRARY(torch_hetero_edgesoftmax, m) {
  m.def("biops_tensor_info", biops_tensor_info);
  m.def("tensor_info", tensor_info);
}