#pragma once

template </*int XPU, */ typename Idx, typename DType>
void _RgcnLayerImpl_wrapper_integratedcsr(
    at::Tensor& csr_rowptr, at::Tensor& csr_col_idx, at::Tensor& csr_eids,
    at::Tensor& csr_reltypes, at::Tensor& hidden, at::Tensor& weight,
    at::Tensor& norm, at::Tensor& ret, bool layer1_flag) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
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
        <<<nblks, nthrs, 0, stream /*, 0, thr_entry->stream*/>>>(
            range_data, ids_data, eids_data, typeids_data, hidden_data,
            weight_data, norm_data, ret_data, num_nodes, feat_len_y, feat_len_x,
            ntypes);
  } else {
    Idx ntypes = weight.size(1);
    Idx feat_len = weight.size(2);
    int nthrs = feat_len;
    RgcnLayer0KernelImpl<Idx, DType>
        <<<nblks, nthrs, 0, stream /*, 0, thr_entry->stream*/>>>(
            range_data, ids_data, eids_data, typeids_data, weight_data,
            norm_data, ret_data, num_nodes, feat_len, ntypes);
  }
}

// template </*int XPU, */ typename Idx, typename DType>
void RgcnLayer0Impl_wrapper_integratedcsr(at::Tensor& csr_rowptr,
                                          at::Tensor& csr_col_idx,
                                          at::Tensor& csr_eids,
                                          at::Tensor& csr_reltypes,
                                          at::Tensor& weight, at::Tensor& norm,
                                          at::Tensor& ret) {
  // NB: graphiler, seastar by default uses int64_t
  _RgcnLayerImpl_wrapper_integratedcsr<int64_t, float>(
      csr_rowptr, csr_col_idx, csr_eids, csr_reltypes, /*dummy_tensor*/ weight,
      weight, norm, ret, false);
}

// template </*int XPU, */ typename Idx, typename DType>
void RgcnLayer1Impl_wrapper_integratedcsr(
    at::Tensor& csr_rowptr, at::Tensor& csr_col_idx, at::Tensor& csr_eids,
    at::Tensor& csr_reltypes, at::Tensor& hidden, at::Tensor& weight,
    at::Tensor& norm, at::Tensor& ret) {
  // NB: graphiler, seastar by default uses int64_t
  _RgcnLayerImpl_wrapper_integratedcsr<int64_t, float>(
      csr_rowptr, csr_col_idx, csr_eids, csr_reltypes, hidden, weight, norm,
      ret, true);
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
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
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
    RgcnLayer1BackwardKernelImpl<<<nblks, nthrs, 0,
                                   stream /*, 0, thr_entry->stream*/>>>(
        range_data, ids_data, eids_data, typeids_data, hidden_data, weight_data,
        norm_data, grad_out_data, grad_hidden_data, grad_weight_data, num_nodes,
        feat_len_y, feat_len_x, ntypes);
  } else {
    Idx ntypes = weight.size(1);
    Idx feat_len = ret.size(2);
    int nthrs = feat_len;
    RgcnLayer0BackwardKernelImpl<<<nblks, nthrs, 0,
                                   stream /*, 0, thr_entry->stream*/>>>(
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
void RgcnLayer0BackwardImpl_wrapper_integratedcsr(
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
}

// template </*int XPU, */ typename Idx, typename DType>
void RgcnLayer1BackwardImpl_wrapper_integratedcsr(
    // GraphRef graph,
    at::Tensor& transposed_csr_rowptr, at::Tensor& transposed_csr_col_idx,
    at::Tensor& transposed_csr_eids, at::Tensor& transposed_csr_reltypes,
    at::Tensor& hidden, at::Tensor& weight, at::Tensor& norm,
    at::Tensor& grad_out, at::Tensor& grad_hidden, at::Tensor& grad_weight) {
  // NB: graphiler, seastar by default uses int64_t
  // TODO: create dummy tensor instead
  _RgcnLayerBackwardImpl_wrapper_integratedcsr<int64_t, float>(
      transposed_csr_rowptr, transposed_csr_col_idx, transposed_csr_eids,
      transposed_csr_reltypes, hidden, weight, norm, grad_out, grad_hidden,
      grad_weight, /*ret_dummy*/ grad_weight, true);
}