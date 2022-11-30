#pragma once

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>

// TODO: create dummy tensor instead whenever unused field in torch export
// functions

namespace HET {

namespace TorchExport {
namespace RGCN {
namespace FwProp {
namespace IntegratedCSR {
template </*int XPU, */ typename Idx, typename DType, bool HybridAssignmentFlag>
void _LayerImpl(at::Tensor& csr_rowptr, at::Tensor& csr_col_idx,
                at::Tensor& csr_eids, at::Tensor& csr_reltypes,
                at::Tensor& hidden, at::Tensor& weight, at::Tensor& norm,
                at::Tensor& ret, bool layer1_flag,
                int num_blocks_on_blocks_per_node) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto range_data = csr_rowptr.data_ptr<Idx>();
  auto ids_data = csr_col_idx.data_ptr<Idx>();
  auto eids_data = csr_eids.data_ptr<Idx>();
  auto typeids_data = csr_reltypes.data_ptr<Idx>();
  DType* hidden_data = hidden.numel() == 0 ? nullptr : hidden.data_ptr<DType>();
  auto weight_data = weight.data_ptr<DType>();
  auto norm_data = norm.data_ptr<DType>();
  auto ret_data = ret.data_ptr<DType>();

  Idx num_nodes = csr_rowptr.numel() - 1;
  Idx num_edges = csr_eids.numel();
  int nblks = num_nodes;

  if constexpr (HybridAssignmentFlag) {
    assert(num_blocks_on_blocks_per_node >= 0);
  } else {
    assert(num_blocks_on_blocks_per_node == -1);
  }

  if (layer1_flag) {
    Idx ntypes = weight.size(0);
    Idx feat_len_y = weight.size(1);
    Idx feat_len_x = weight.size(2);
    int nthrs = feat_len_y * feat_len_x;
    if constexpr (HybridAssignmentFlag) {
      RgcnLayer1KernelHybridAssignImpl<Idx, DType><<<nblks, nthrs, 0, stream>>>(
          range_data, ids_data, eids_data, typeids_data, hidden_data,
          weight_data, norm_data, ret_data, num_nodes, feat_len_y, feat_len_x,
          ntypes, num_blocks_on_blocks_per_node);
    } else {
      RgcnLayer1KernelImpl<Idx, DType><<<nblks, nthrs, 0, stream>>>(
          range_data, ids_data, eids_data, typeids_data, hidden_data,
          weight_data, norm_data, ret_data, num_nodes, feat_len_y, feat_len_x,
          ntypes);
    }
  } else {
    Idx ntypes = weight.size(1);
    Idx feat_len = weight.size(2);
    int nthrs = feat_len;
    if constexpr (HybridAssignmentFlag) {
      RgcnLayer0KernelHybridAssignImpl<Idx, DType><<<nblks, nthrs, 0, stream>>>(
          range_data, ids_data, eids_data, typeids_data, weight_data, norm_data,
          ret_data, num_nodes, feat_len, ntypes, num_blocks_on_blocks_per_node);
    } else {
      RgcnLayer0KernelImpl<Idx, DType><<<nblks, nthrs, 0, stream>>>(
          range_data, ids_data, eids_data, typeids_data, weight_data, norm_data,
          ret_data, num_nodes, feat_len, ntypes);
    }
  }
}

// template </*int XPU, */ typename Idx, typename DType>
void Layer0Impl(at::Tensor& csr_rowptr, at::Tensor& csr_col_idx,
                at::Tensor& csr_eids, at::Tensor& csr_reltypes,
                at::Tensor& weight, at::Tensor& norm, at::Tensor& ret) {
  // NB: graphiler, seastar by default uses int64_t
  at::Tensor dummy_tensor;
  _LayerImpl<int64_t, float, false>(csr_rowptr, csr_col_idx, csr_eids,
                                    csr_reltypes, /*dummy_hidden*/ dummy_tensor,
                                    weight, norm, ret, false, -1);
}

void Layer0HybridAssignmentImpl(at::Tensor& csr_rowptr, at::Tensor& csr_col_idx,
                                at::Tensor& csr_eids, at::Tensor& csr_reltypes,
                                at::Tensor& weight, at::Tensor& norm,
                                at::Tensor& ret,
                                int64_t num_blocks_on_blocks_per_node) {
  // NB: graphiler, seastar by default uses int64_t
  at::Tensor dummy_tensor;
  _LayerImpl<int64_t, float, true>(csr_rowptr, csr_col_idx, csr_eids,
                                   csr_reltypes, /*dummy_hidden*/ dummy_tensor,
                                   weight, norm, ret, false,
                                   num_blocks_on_blocks_per_node);
}

// template </*int XPU, */ typename Idx, typename DType>
void Layer1Impl(at::Tensor& csr_rowptr, at::Tensor& csr_col_idx,
                at::Tensor& csr_eids, at::Tensor& csr_reltypes,
                at::Tensor& hidden, at::Tensor& weight, at::Tensor& norm,
                at::Tensor& ret) {
  // NB: graphiler, seastar by default uses int64_t
  _LayerImpl<int64_t, float, false>(csr_rowptr, csr_col_idx, csr_eids,
                                    csr_reltypes, hidden, weight, norm, ret,
                                    true, -1);
}

void Layer1HybridAssignmentImpl(at::Tensor& csr_rowptr, at::Tensor& csr_col_idx,
                                at::Tensor& csr_eids, at::Tensor& csr_reltypes,
                                at::Tensor& hidden, at::Tensor& weight,
                                at::Tensor& norm, at::Tensor& ret,
                                int64_t num_blocks_on_blocks_per_node) {
  // NB: graphiler, seastar by default uses int64_t
  _LayerImpl<int64_t, float, true>(csr_rowptr, csr_col_idx, csr_eids,
                                   csr_reltypes, hidden, weight, norm, ret,
                                   true, num_blocks_on_blocks_per_node);
}
}  // namespace IntegratedCSR
}  // namespace FwProp
namespace BckProp {
namespace IntegratedCSR {
// the referential implementation from seastar
template </*int XPU, */ typename Idx, typename DType, bool HybridAssignmentFlag>
void _LayerImpl(
    // GraphRef graph,
    at::Tensor& transposed_csr_rowptr, at::Tensor& transposed_csr_col_idx,
    at::Tensor& transposed_csr_eids, at::Tensor& transposed_csr_reltypes,
    at::Tensor& hidden, at::Tensor& weight, at::Tensor& norm,
    at::Tensor& grad_out, at::Tensor& grad_hidden, at::Tensor& grad_weight,
    at::Tensor& ret, bool layer1_flag, int num_blocks_on_blocks_per_node) {
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
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto range_data = transposed_csr_rowptr.data_ptr<Idx>();
  auto ids_data = transposed_csr_col_idx.data_ptr<Idx>();
  auto eids_data = transposed_csr_eids.data_ptr<Idx>();
  auto typeids_data = transposed_csr_reltypes.data_ptr<Idx>();
  DType* hidden_data = hidden.numel() == 0 ? nullptr : hidden.data_ptr<DType>();
  DType* weight_data = weight.numel() == 0 ? nullptr : weight.data_ptr<DType>();
  auto norm_data = norm.data_ptr<DType>();
  auto grad_out_data = grad_out.data_ptr<DType>();
  DType* grad_hidden_data =
      grad_hidden.numel() == 0 ? nullptr : grad_hidden.data_ptr<DType>();
  DType* grad_weight_data =
      grad_weight.numel() == 0 ? nullptr : grad_weight.data_ptr<DType>();
  DType* ret_data = ret.numel() == 0 ? nullptr : ret.data_ptr<DType>();
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
    if constexpr (HybridAssignmentFlag) {
      RgcnLayer1BackwardKernelHybridAssignImpl<<<nblks, nthrs, 0, stream>>>(
          range_data, ids_data, eids_data, typeids_data, hidden_data,
          weight_data, norm_data, grad_out_data, grad_hidden_data,
          grad_weight_data, num_nodes, feat_len_y, feat_len_x, ntypes,
          num_blocks_on_blocks_per_node);
    } else {
      RgcnLayer1BackwardKernelImpl<<<nblks, nthrs, 0, stream>>>(
          range_data, ids_data, eids_data, typeids_data, hidden_data,
          weight_data, norm_data, grad_out_data, grad_hidden_data,
          grad_weight_data, num_nodes, feat_len_y, feat_len_x, ntypes);
    }
  } else {
    Idx ntypes = weight.size(1);
    Idx feat_len = ret.size(2);
    int nthrs = feat_len;
    if constexpr (HybridAssignmentFlag) {
      RgcnLayer0BackwardKernelHybridAssignImpl<<<nblks, nthrs, 0, stream>>>(
          range_data, ids_data, eids_data, typeids_data, grad_out_data,
          norm_data, ret_data, num_nodes, feat_len, ntypes,
          num_blocks_on_blocks_per_node);
    } else {
      RgcnLayer0BackwardKernelImpl<<<nblks, nthrs, 0, stream>>>(
          range_data, ids_data, eids_data, typeids_data, grad_out_data,
          norm_data, ret_data, num_nodes, feat_len, ntypes);
    }
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
void Layer0Impl(
    // GraphRef graph,
    at::Tensor& transposed_csr_rowptr, at::Tensor& transposed_csr_col_idx,
    at::Tensor& transposed_csr_eids, at::Tensor& transposed_csr_reltypes,
    at::Tensor& grad_out, at::Tensor& norm, at::Tensor& ret) {
  // NB: graphiler, seastar by default uses int64_t
  at::Tensor dummy_tensor;
  _LayerImpl<int64_t, float, false>(
      transposed_csr_rowptr, transposed_csr_col_idx, transposed_csr_eids,
      transposed_csr_reltypes, /*hidden_dummy*/ dummy_tensor,
      /*weight_dummy*/ dummy_tensor, norm, grad_out,
      /*grad_hidden_dummy*/ ret, /*grad_weight_dummy*/ dummy_tensor, ret, false,
      -1);
}

void Layer0HybridAssignmentImpl(
    // GraphRef graph,
    at::Tensor& transposed_csr_rowptr, at::Tensor& transposed_csr_col_idx,
    at::Tensor& transposed_csr_eids, at::Tensor& transposed_csr_reltypes,
    at::Tensor& grad_out, at::Tensor& norm, at::Tensor& ret,
    int64_t num_blocks_on_blocks_per_node) {
  // NB: graphiler, seastar by default uses int64_t
  at::Tensor dummy_tensor;
  _LayerImpl<int64_t, float, true>(
      transposed_csr_rowptr, transposed_csr_col_idx, transposed_csr_eids,
      transposed_csr_reltypes, /*hidden_dummy*/ dummy_tensor,
      /*weight_dummy*/ dummy_tensor, norm, grad_out,
      /*grad_hidden_dummy*/ dummy_tensor, /*grad_weight_dummy*/ dummy_tensor,
      ret, false, num_blocks_on_blocks_per_node);
}

// template </*int XPU, */ typename Idx, typename DType>
void Layer1Impl(
    // GraphRef graph,
    at::Tensor& transposed_csr_rowptr, at::Tensor& transposed_csr_col_idx,
    at::Tensor& transposed_csr_eids, at::Tensor& transposed_csr_reltypes,
    at::Tensor& hidden, at::Tensor& weight, at::Tensor& norm,
    at::Tensor& grad_out, at::Tensor& grad_hidden, at::Tensor& grad_weight) {
  // NB: graphiler, seastar by default uses int64_t
  at::Tensor dummy_tensor;
  _LayerImpl<int64_t, float, false>(
      transposed_csr_rowptr, transposed_csr_col_idx, transposed_csr_eids,
      transposed_csr_reltypes, hidden, weight, norm, grad_out, grad_hidden,
      grad_weight, /*ret_dummy*/ dummy_tensor, true, -1);
}

void Layer1HybridAssignmentImpl(
    // GraphRef graph,
    at::Tensor& transposed_csr_rowptr, at::Tensor& transposed_csr_col_idx,
    at::Tensor& transposed_csr_eids, at::Tensor& transposed_csr_reltypes,
    at::Tensor& hidden, at::Tensor& weight, at::Tensor& norm,
    at::Tensor& grad_out, at::Tensor& grad_hidden, at::Tensor& grad_weight,
    int64_t num_blocks_on_blocks_per_node) {
  // NB: graphiler, seastar by default uses int64_t
  at::Tensor dummy_tensor;
  _LayerImpl<int64_t, float, true>(
      transposed_csr_rowptr, transposed_csr_col_idx, transposed_csr_eids,
      transposed_csr_reltypes, hidden, weight, norm, grad_out, grad_hidden,
      grad_weight, /*ret_dummy*/ dummy_tensor, true,
      num_blocks_on_blocks_per_node);
}

}  // namespace IntegratedCSR
}  // namespace BckProp
}  // namespace RGCN
}  // namespace TorchExport
}  // namespace HET