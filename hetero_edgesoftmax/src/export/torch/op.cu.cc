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

template </*int XPU, */ typename Idx, typename DType>
void _RgcnLayerImpl_wrapper_integratedcsr(
    at::Tensor& csr_rowptr, at::Tensor& csr_col_idx, at::Tensor& csr_eids,
    at::Tensor& csr_reltypes, at::Tensor& hidden, at::Tensor& weight,
    at::Tensor& norm, at::Tensor& ret, bool layer1_flag) {
  auto range_data = csr_rowptr.data_ptr<Idx>();
  auto ids_data = csr_col_idx.data_ptr<Idx>();
  auto eids_data = csr_eids.data_ptr<Idx>();
  auto typeids_data = csr_reltypes.data_ptr<Idx>();
  auto hidden_data = hidden.data_ptr<DType>();
  auto weight_data = weight.data_ptr<DType>();
  auto norm_data = norm.data_ptr<DType>();
  auto ret_data = ret.data_ptr<DType>();

  Idx num_nodes = csr_rowptr.numel() - 1;
  Idx num_edges = csr_eids.numel();
  int nblks = num_nodes;

  if (layer1_flag) {
    Idx ntypes = weight.size(0);
    Idx feat_len_y = weight.size(1);
    Idx feat_len_x = weight.size(2);
    int nthrs = feat_len_y * feat_len_x;
    RgcnLayer1KernelImpl<Idx, DType>
        <<<nblks, nthrs /*, 0, thr_entry->stream*/>>>(
            range_data, ids_data, eids_data, typeids_data, hidden_data,
            weight_data, norm_data, ret_data, num_nodes, feat_len_y, feat_len_x,
            ntypes);
  } else {
    Idx ntypes = weight.size(1);
    Idx feat_len = weight.size(2);
    int nthrs = feat_len;
    RgcnLayer0KernelImpl<Idx, DType>
        <<<nblks, nthrs /*, 0, thr_entry->stream*/>>>(
            range_data, ids_data, eids_data, typeids_data, weight_data,
            norm_data, ret_data, num_nodes, feat_len, ntypes);
  }
}

// template </*int XPU, */ typename Idx, typename DType>
bool RgcnLayer0Impl_wrapper_integratedcsr(at::Tensor& csr_rowptr,
                                          at::Tensor& csr_col_idx,
                                          at::Tensor& csr_eids,
                                          at::Tensor& csr_reltypes,
                                          at::Tensor& weight, at::Tensor& norm,
                                          at::Tensor& ret) {
  // NB: graphiler, seastar by default uses int64_t
  _RgcnLayerImpl_wrapper_integratedcsr<int64_t, float>(
      csr_rowptr, csr_col_idx, csr_eids, csr_reltypes, /*dummy_tensor*/ weight,
      weight, norm, ret, false);
  return false;
}

// template </*int XPU, */ typename Idx, typename DType>
bool RgcnLayer1Impl_wrapper_integratedcsr(
    at::Tensor& csr_rowptr, at::Tensor& csr_col_idx, at::Tensor& csr_eids,
    at::Tensor& csr_reltypes, at::Tensor& hidden, at::Tensor& weight,
    at::Tensor& norm, at::Tensor& ret) {
  // NB: graphiler, seastar by default uses int64_t
  _RgcnLayerImpl_wrapper_integratedcsr<int64_t, float>(
      csr_rowptr, csr_col_idx, csr_eids, csr_reltypes, hidden, weight, norm,
      ret, true);
  return false;
}

// the referential implementation from seastar
template </*int XPU, */ typename Idx, typename DType>
void _RgcnLayerBackwardImpl_wrapper_integratedcsr(
    // GraphRef graph,
    at::Tensor& transposed_csr_rowptr, at::Tensor& transposed_csr_col_idx,
    at::Tensor& transposed_csr_eids, at::Tensor& transposed_csr_reltypes,
    at::Tensor& hidden, at::Tensor& weight, at::Tensor& norm,
    at::Tensor& grad_out, at::Tensor& grad_hidden, at::Tensor& grad_weight,
    at::Tensor& ret, bool layer1_flag) {
  // assert(csr.IsSortedByEdgeType_CPU());
  // cudaDeviceSynchronize();
  // auto t1 = std::chrono::steady_clock::now();
  // typedef int32_t Idx;
  // typedef float DType;
  // auto csr = graph->GetCsrSortedByEdgeType(true);
  // auto ranges = csr[0];
  // auto ids = csr[1];
  // auto eids = csr[2];
  // auto type_ids = csr[3];
  auto range_data = transposed_csr_rowptr.data_ptr<Idx>();
  auto ids_data = transposed_csr_col_idx.data_ptr<Idx>();
  auto eids_data = transposed_csr_eids.data_ptr<Idx>();
  auto typeids_data = transposed_csr_reltypes.data_ptr<Idx>();
  auto hidden_data = hidden.data_ptr<DType>();
  auto weight_data = weight.data_ptr<DType>();
  auto norm_data = norm.data_ptr<DType>();
  auto grad_out_data = grad_out.data_ptr<DType>();
  auto grad_hidden_data = grad_hidden.data_ptr<DType>();
  auto grad_weight_data = grad_weight.data_ptr<DType>();
  auto ret_data = ret.data_ptr<DType>();
  // print_dims(hidden);
  // print_dims(weight);
  // print_dims(norm);
  // print_dims(grad_out);
  // print_dims(grad_hidden);
  // print_dims(grad_weight);
  // Idx num_nodes = ranges->shape[0] - 1;
  // Idx num_edges = eids->shape[0];
  // Idx ntypes = weight->shape[0];
  // Idx feat_len_y = weight->shape[1];
  // Idx feat_len_x = weight->shape[2];
  Idx num_nodes = transposed_csr_rowptr.numel() - 1;
  Idx num_edges = transposed_csr_col_idx.numel();
  int nblks = num_nodes;
  // auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  // cuda_err_chk(cudaDeviceSynchronize());
  if (layer1_flag) {
    Idx ntypes = weight.size(0);
    Idx feat_len_y = weight.size(1);
    Idx feat_len_x = weight.size(2);
    int nthrs = feat_len_y * feat_len_x;
    RgcnLayer1BackwardKernelImpl<<<nblks, nthrs /*, 0, thr_entry->stream*/>>>(
        range_data, ids_data, eids_data, typeids_data, hidden_data, weight_data,
        norm_data, grad_out_data, grad_hidden_data, grad_weight_data, num_nodes,
        feat_len_y, feat_len_x, ntypes);
  } else {
    Idx ntypes = weight.size(1);
    Idx feat_len = ret.size(2);
    int nthrs = feat_len;
    RgcnLayer0BackwardKernelImpl<<<nblks, nthrs /*, 0, thr_entry->stream*/>>>(
        range_data, ids_data, eids_data, typeids_data, grad_out_data, norm_data,
        ret_data, num_nodes, feat_len, ntypes);
  }
  // cudaDeviceSynchronize();
  // auto t2 = std::chrono::steady_clock::now();
  // LOG(INFO) << "layer 1 backward kernel takes:" <<
  // std::chrono::duration_cast<std::chrono::milliseconds>(t2
  // -t1).count()/1000.0 << " s";
  // cuda_err_chk(cudaPeekAtLastError());
  // cuda_err_chk(cudaDeviceSynchronize());
}

// template </*int XPU, */ typename Idx, typename DType>
bool RgcnLayer0BackwardImpl_wrapper_integratedcsr(
    // GraphRef graph,
    at::Tensor& transposed_csr_rowptr, at::Tensor& transposed_csr_col_idx,
    at::Tensor& transposed_csr_eids, at::Tensor& transposed_csr_reltypes,
    at::Tensor& grad_out, at::Tensor& norm, at::Tensor& ret) {
  // NB: graphiler, seastar by default uses int64_t
  _RgcnLayerBackwardImpl_wrapper_integratedcsr<int64_t, float>(
      transposed_csr_rowptr, transposed_csr_col_idx, transposed_csr_eids,
      transposed_csr_reltypes, /*hidden_dummy*/ ret, /*weight_dummy*/ ret, norm,
      grad_out,
      /*grad_hidden_dummy*/ ret, /*grad_weight_dummy*/ ret, ret, false);
  return false;
}

// template </*int XPU, */ typename Idx, typename DType>
bool RgcnLayer1BackwardImpl_wrapper_integratedcsr(
    // GraphRef graph,
    at::Tensor& transposed_csr_rowptr, at::Tensor& transposed_csr_col_idx,
    at::Tensor& transposed_csr_eids, at::Tensor& transposed_csr_reltypes,
    at::Tensor& hidden, at::Tensor& weight, at::Tensor& norm,
    at::Tensor& grad_out, at::Tensor& grad_hidden, at::Tensor& grad_weight) {
  // NB: graphiler, seastar by default uses int64_t
  _RgcnLayerBackwardImpl_wrapper_integratedcsr<int64_t, float>(
      transposed_csr_rowptr, transposed_csr_col_idx, transposed_csr_eids,
      transposed_csr_reltypes, hidden, weight, norm, grad_out, grad_hidden,
      grad_weight, /*ret_dummy*/ grad_weight, true);
  return false;
}

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

void test_argument_takein(std::string str, bool flag) {
  std::cout << "test_string_takein: " << str << std::endl;
  std::cout << "test_bool_takein: " << flag << std::endl;
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