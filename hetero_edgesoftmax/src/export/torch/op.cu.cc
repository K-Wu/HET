// from
// https://stackoverflow.com/questions/68401650/how-can-i-make-a-pytorch-extension-with-cmake
// kernels defined in this file are simply wrapped up in
// hetero_edgesoftmax/python/kernels.py to provide python API, then used to
// define autograd functions and layers in hetero_edgesoftmax/python/kernels.py,
// which is finally referred to by end2end cases in
// hetero_edgesoftmax/python/<model name>/.*.py
// NB: This contains wrapper versions for python api export originally
// implemented at hetero_edgesoftmax/include/DGLHackKernel/RGCNLayers.h. Please
// update accordingly whenever there is update.
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <iostream>
#include "DGLHackKernel/DGLHackKernel.h"
#include "hetero_edgesoftmax.h"
// TODO: assume int32_t and float32 for now. but we may need to support other
// types
// TODO: check if torch builtin has the same encoding as int32_t and float32
#include "DGLHackKernel/OpExport/HGTOps.inc"
#include "DGLHackKernel/OpExport/HGTPrepToTensors.inc"
#include "DGLHackKernel/OpExport/RGCNOps.inc"
//#include "DGLHackKernel/OpExport/RGATOps.inc"

std::vector<at::Tensor> biops_tensor_info(at::Tensor& one_tensor,
                                          at::Tensor& other_tensor) {
  std::cout << "one_tensor device: " << one_tensor.device() << std::endl;
  std::cout << "other_tensor device: " << other_tensor.device() << std::endl;

  std::vector<at::Tensor> result = {one_tensor.clone(), other_tensor.clone()};
  return result;
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

std::vector<at::Tensor> transpose_csr(at::Tensor& csr_rowptr,
                                      at::Tensor& csr_col_idx,
                                      at::Tensor& csr_reltypes,
                                      at::Tensor& csr_eids) {
  // NB: graphiler, seastar by default uses int64_t
  assert(csr_rowptr.is_contiguous());
  assert(csr_col_idx.is_contiguous());
  assert(csr_reltypes.is_contiguous());
  assert(csr_eids.is_contiguous());
  assert(csr_rowptr.device().is_cpu());
  assert(csr_col_idx.device().is_cpu());
  assert(csr_reltypes.device().is_cpu());
  assert(csr_eids.device().is_cpu());

  thrust::host_vector<int64_t> csr_rowptr_thrust(
      csr_rowptr.data_ptr<int64_t>(),
      csr_rowptr.data_ptr<int64_t>() + csr_rowptr.numel());
  thrust::host_vector<int64_t> csr_col_idx_thrust(
      csr_col_idx.data_ptr<int64_t>(),
      csr_col_idx.data_ptr<int64_t>() + csr_col_idx.numel());
  thrust::host_vector<int64_t> csr_reltypes_thrust(
      csr_reltypes.data_ptr<int64_t>(),
      csr_reltypes.data_ptr<int64_t>() + csr_reltypes.numel());
  thrust::host_vector<int64_t> csr_eids_thrust(
      csr_eids.data_ptr<int64_t>(),
      csr_eids.data_ptr<int64_t>() + csr_eids.numel());

  MyHeteroIntegratedCSR<int64_t, std::allocator<int64_t>> csr(
      csr_rowptr_thrust, csr_col_idx_thrust, csr_reltypes_thrust,
      csr_eids_thrust);
  csr.Transpose();

  at::Tensor transposed_rowptr =
      at::empty({csr_rowptr.numel()}, csr_rowptr.options());
  at::Tensor transposed_col_idx =
      at::empty({csr_col_idx.numel()}, csr_col_idx.options());
  at::Tensor transposed_reltypes =
      at::empty({csr_reltypes.numel()}, csr_reltypes.options());
  at::Tensor transposed_eids =
      at::empty({csr_eids.numel()}, csr_eids.options());
  assert(transposed_rowptr.is_contiguous());
  assert(transposed_col_idx.is_contiguous());
  assert(transposed_reltypes.is_contiguous());
  assert(transposed_eids.is_contiguous());
  assert(transposed_rowptr.device().is_cpu());
  assert(transposed_col_idx.device().is_cpu());
  assert(transposed_reltypes.device().is_cpu());
  assert(transposed_eids.device().is_cpu());

  std::copy(csr.row_ptr.begin(), csr.row_ptr.end(),
            transposed_rowptr.data_ptr<int64_t>());
  std::copy(csr.col_idx.begin(), csr.col_idx.end(),
            transposed_col_idx.data_ptr<int64_t>());
  std::copy(csr.rel_type.begin(), csr.rel_type.end(),
            transposed_reltypes.data_ptr<int64_t>());
  std::copy(csr.eids.begin(), csr.eids.end(),
            transposed_eids.data_ptr<int64_t>());

  std::vector<at::Tensor> result = {transposed_rowptr, transposed_col_idx,
                                    transposed_reltypes, transposed_eids};
  return result;
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

// std::vector<at::Tensor> load_fb15k237(bool sorted, bool sorted_by_src,
// std::string data_path_prefix){
//   cusp::csr_matrix<int, int, cusp::host_memory> fb15k237_graph =
//   LoadFB15k237Data(sorted, sorted_by_src, data_path_prefix);
// }

// std::vector<at::Tensor> load_ogbn_wikikg2(bool sorted, std::string
// data_path_prefix){
//   cusp::csr_matrix<int, int, cusp::host_memory> ogbn_wikikg2_graph =
//   LoadOGBNWikiKG2Data(sorted, data_path_prefix);
// }

// std::vector<at::Tensor> load_mag(std::string data_path_prefix){
//   MyHeteroIntegratedCSR<int, std::allocator<int>>  mag_graph =
//   LoadOGBN_MAG(data_path_prefix);

// }

TORCH_LIBRARY(torch_hetero_edgesoftmax, m) {
  m.def("biops_tensor_info", biops_tensor_info);
  m.def("tensor_info", tensor_info);
  m.def("rgcn_layer0_csr", RgcnLayer0Impl_wrapper_integratedcsr);
  m.def("rgcn_layer0_backward_csr",
        RgcnLayer0BackwardImpl_wrapper_integratedcsr);
  m.def("rgcn_layer1_csr", RgcnLayer1Impl_wrapper_integratedcsr);
  m.def("rgcn_layer1_backward_csr",
        RgcnLayer1BackwardImpl_wrapper_integratedcsr);
  m.def("transpose_csr", transpose_csr);
  m.def("test_argument_takein", test_argument_takein);
  // m.def("load_fb15k237", load_fb15k237);
  // m.def("load_ogbn_wikikg2", load_ogbn_wikikg2);
  // m.def("load_mag", load_mag);
}