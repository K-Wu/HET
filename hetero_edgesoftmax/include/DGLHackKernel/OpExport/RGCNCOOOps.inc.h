#pragma once
namespace HET {
namespace TorchExport {
namespace RGCN {
namespace FwProp {
namespace IntegratedCOO {
template </*int XPU, */ typename Idx, typename DType>
void _LayerImpl(at::Tensor& coo_row_idx, at::Tensor& coo_col_idx,
                at::Tensor& coo_eids, at::Tensor& coo_reltypes,
                at::Tensor& hidden, at::Tensor& weight, at::Tensor& norm,
                at::Tensor& ret, bool layer1_flag) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto row_idx_data = coo_row_idx.data_ptr<Idx>();
  auto ids_data = coo_col_idx.data_ptr<Idx>();
  auto eids_data = coo_eids.data_ptr<Idx>();
  auto typeids_data = coo_reltypes.data_ptr<Idx>();
  auto hidden_data = hidden.data_ptr<DType>();
  auto weight_data = weight.data_ptr<DType>();
  auto norm_data = norm.data_ptr<DType>();
  auto ret_data = ret.data_ptr<DType>();

  Idx num_edges = coo_eids.numel();
  // int nblks = num_nodes;

  if (layer1_flag) {
    Idx ntypes = weight.size(0);
    Idx feat_len_y = weight.size(1);
    Idx feat_len_x = weight.size(2);
    // int nthrs = feat_len_y * feat_len_x;
    int nthrs = feat_len_x < 256 ? 256 : feat_len_x;
    assert(nthrs % 32 == 0);
    int nblks =
        ceil_div<>(num_edges, (int64_t)nthrs / 32);  // 32 is the warp size
    RgcnLayer1COOKernelImpl<Idx, DType><<<nblks, nthrs, 0, stream>>>(
        row_idx_data, ids_data, eids_data, typeids_data, hidden_data,
        weight_data, norm_data, ret_data, num_edges, feat_len_y, feat_len_x,
        ntypes);
  } else {
    Idx ntypes = weight.size(1);
    Idx feat_len = weight.size(2);
    int nthrs = feat_len;
    assert(0 && "not implemented");
    // RgcnLayer0KernelImpl<Idx, DType>
    //    <<<nblks, nthrs >>>(
    //        range_data, ids_data, eids_data, typeids_data, weight_data,
    //        norm_data, ret_data, num_nodes, feat_len, ntypes);
  }
}

// template </*int XPU, */ typename Idx, typename DType>
void Layer1Impl(at::Tensor& coo_row_idx, at::Tensor& coo_col_idx,
                at::Tensor& coo_eids, at::Tensor& coo_reltypes,
                at::Tensor& hidden, at::Tensor& weight, at::Tensor& norm,
                at::Tensor& ret) {
  // NB: graphiler, seastar by default uses int64_t
  _LayerImpl<int64_t, float>(coo_row_idx, coo_col_idx, coo_eids, coo_reltypes,
                             hidden, weight, norm, ret, true);
}
}  // namespace IntegratedCOO
}  // namespace FwProp
namespace BckProp {
namespace IntegratedCOO {
// the referential implementation from seastar
template </*int XPU, */ typename Idx, typename DType>
void _LayerBackwardImpl(
    // GraphRef graph,
    at::Tensor& transposed_coo_row_idx, at::Tensor& transposed_coo_col_idx,
    at::Tensor& transposed_coo_eids, at::Tensor& transposed_coo_reltypes,
    at::Tensor& hidden, at::Tensor& weight, at::Tensor& norm,
    at::Tensor& grad_out, at::Tensor& grad_hidden, at::Tensor& grad_weight,
    at::Tensor& ret, bool layer1_flag) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
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
  auto row_idx_data = transposed_coo_row_idx.data_ptr<Idx>();
  auto ids_data = transposed_coo_col_idx.data_ptr<Idx>();
  auto eids_data = transposed_coo_eids.data_ptr<Idx>();
  auto typeids_data = transposed_coo_reltypes.data_ptr<Idx>();
  auto hidden_data = hidden.data_ptr<DType>();
  auto weight_data = weight.data_ptr<DType>();
  auto norm_data = norm.data_ptr<DType>();
  auto grad_out_data = grad_out.data_ptr<DType>();
  auto grad_hidden_data = grad_hidden.data_ptr<DType>();
  auto grad_weight_data = grad_weight.data_ptr<DType>();
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
  Idx num_edges = transposed_coo_col_idx.numel();
  // int nblks = num_nodes;
  // auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  // cuda_err_chk(cudaDeviceSynchronize());
  if (layer1_flag) {
    Idx ntypes = weight.size(0);
    Idx feat_len_y = weight.size(1);
    Idx feat_len_x = weight.size(2);
    // int nthrs = feat_len_y * feat_len_x;
    int nthrs = feat_len_x < 256 ? 256 : feat_len_x;
    assert(nthrs % 32 == 0);
    int nblks =
        ceil_div<>(num_edges, (int64_t)nthrs / 32);  // 32 is the warp size
    RgcnLayer1BackwardCOOKernelImpl<<<nblks, nthrs, 0, stream>>>(
        row_idx_data, ids_data, eids_data, typeids_data, hidden_data,
        weight_data, norm_data, grad_out_data, grad_hidden_data,
        grad_weight_data, num_edges, feat_len_y, feat_len_x, ntypes);
  } else {
    Idx ntypes = weight.size(1);
    Idx feat_len = ret.size(2);
    int nthrs = feat_len;
    assert(0 && "not implemented");
    // RgcnLayer0BackwardKernelImpl<<<nblks, nthrs>>>(
    //    range_data, ids_data, eids_data, typeids_data, grad_out_data,
    //    norm_data, ret_data, num_nodes, feat_len, ntypes);
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
void Layer1BackwardImpl(
    // GraphRef graph,
    at::Tensor& transposed_coo_row_idx, at::Tensor& transposed_coo_col_idx,
    at::Tensor& transposed_coo_eids, at::Tensor& transposed_coo_reltypes,
    at::Tensor& hidden, at::Tensor& weight, at::Tensor& norm,
    at::Tensor& grad_out, at::Tensor& grad_hidden, at::Tensor& grad_weight) {
  // NB: graphiler, seastar by default uses int64_t
  at::Tensor dummy_tensor;
  _LayerBackwardImpl<int64_t, float>(
      transposed_coo_row_idx, transposed_coo_col_idx, transposed_coo_eids,
      transposed_coo_reltypes, hidden, weight, norm, grad_out, grad_hidden,
      grad_weight, /*ret_dummy*/ dummy_tensor, true);
}
}  // namespace IntegratedCOO
}  // namespace BckProp
}  // namespace RGCN
}  // namespace TorchExport
}  // namespace HET