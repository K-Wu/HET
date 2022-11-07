#pragma once

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>

int64_t ComputeXLength(at::Tensor& tensor) {
  int64_t ret = 1;
  for (int i = 1; i < tensor.dim(); ++i) {
    ret *= tensor.size(i);
  }
  return ret;
}

template </*int XPU, */ typename Idx, typename DType>
void _FusedGatKernelImpl_wrapper_integratedcsr(at::Tensor& incsr_row_ptr,
                                               at::Tensor& incsr_col_idx,
                                               at::Tensor& incsr_eids,
                                               at::Tensor& feat_src,
                                               at::Tensor& el, at::Tensor& er,
                                               at::Tensor& sum, at::Tensor& exp,
                                               at::Tensor& ret, double slope) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  // As GAT only has 1 type of relationship, we use a specialcase of separateCSR
  // where num releationship is asserted as 1
  // assert(incsr.num_rels == 1);
  // static_assert(XPU==kDLGPU);
  const Idx MAX_NBLKS = 65535;
  const Idx MAX_NTHRS = 1024;
  // zero out ret, and packing feat_src, el, er, ret, graph together into one
  // struct using raw float pointers get csr matrix
  GatFusedData<Idx, DType> gdata;

  int64_t el_xlen = ComputeXLength(el);
  int64_t feat_src_xlen = ComputeXLength(feat_src);
  int64_t ret_len = ComputeXLength(ret);

  gdata.feat_src = feat_src.data_ptr<DType>();
  gdata.el = el.data_ptr<DType>();
  gdata.er = er.data_ptr<DType>();
  gdata.sum = sum.data_ptr<DType>();
  gdata.exp = exp.data_ptr<DType>();
  gdata.ret = ret.data_ptr<DType>();
  gdata.leaky_relu_slope = slope;
  gdata.n = el.numel() / el_xlen;
  gdata.e_xlen = el_xlen;
  gdata.feat_src_xlen = feat_src_xlen;
  gdata.feat_src_hidden = feat_src_xlen / el_xlen;
  gdata.ret_xlen = ret_len;
  // std::vector<IdArray> incsr_elements = graph->GetAdj(0,true, "csr");
  // printf("!!!!!%d, %d, %d\n",graph->NumVertices(0), graph->NumVertexTypes(),
  // graph->NumEdgeTypes()); aten::CSRMatrix incsr(graph->NumVertices(0),
  // graph->NumVertices(0), incsr_elements[0], incsr_elements[1],
  // incsr_elements[2]);

  gdata.eids = incsr_eids.data_ptr<Idx>();

  // write a device function and call it from here
  // LOG(INFO) << "Within Fused Gat Kernel Impl." << "feat_src_dim:" <<
  // feat_src.GetSize()/sizeof(DType)/feat_src_xlen << "*" << feat_src_xlen
  //    <<" el_dim:" << el.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen  <<
  //    " ret_dim:" << ret.GetSize()/sizeof(DType)/ret_len <<"*" << ret_len
  //    <<" sum_dim:" << sum.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen
  //    <<" exp_dim:" << exp.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen
  //    << " graph csr row_offset length:" <<csr.row_offsets.length << " graph
  //    csr column indices length:" << csr.column_indices.length;

  // Configure kernel launch parameters.
  int nthrs_x = 32;
  int nthrs_y = 1;
  int nblks_x = (el_xlen + nthrs_x - 1) / (nthrs_x);
  int nblks_y = std::min(gdata.n, MAX_NBLKS);
  const dim3 nblks(nblks_x, nblks_y);
  const dim3 nthrs(nthrs_x, nthrs_y);
  int64_t incsr_num_rows = incsr_row_ptr.numel() - 1;

  // LOG(INFO) << "kernel1 blk dim:" << nblks_x << "*" <<nblks_y << " thr dim:"
  // <<nthrs_x << "*" << nthrs_y; aten::CSRMatrix incsr =
  // static_pointer_cast<ImmutableGraph*>(graph)->GetInCSR()->ToCSRMatrix();
  // std::vector<IdArray> incsr_elements = graph->GetAdj();
  // aten::CSRMatrix incsr(graph->NumVertices(), graph->NumVertices(),
  // incsr_elements[0], incsr_elements[1], incsr_elements[2]); print_gdata<Idx,
  // DType>(feat_src,el,er,sum,exp,ret,el_xlen, feat_src_xlen,
  // graph->NumVertices(0),incsr_elements[1].NumElements(), incsr_elements[0],
  // incsr_elements[1], incsr_elements[2]); gatExpLeakyReluSumKernel<<<nblks,
  // nthrs, el_xlen*sizeof(DType), thr_entry->stream>>>(gdata, csr);
  // cuda_err_chk(cudaDeviceSynchronize());
  //   std::chrono::high_resolution_clock::time_point t1 =
  //       std::chrono::high_resolution_clock::now();
  gatExpLeakyReluSumKernel<Idx, DType, true, false>
      <<<nblks, nthrs, 0, stream>>>(gdata, incsr_row_ptr.data_ptr<Idx>(),
                                    incsr_col_idx.data_ptr<Idx>(),
                                    incsr_num_rows, nullptr, nullptr);

  // cuda_err_chk(cudaPeekAtLastError());
  // cuda_err_chk(cudaDeviceSynchronize());
  nthrs_x = FindNumThreads(el_xlen, 64);
  nthrs_y = FindNumThreads(gdata.feat_src_hidden, MAX_NTHRS / nthrs_x);
  nblks_x = 1;
  nblks_y = std::min(gdata.n, MAX_NBLKS);
  const dim3 nthrs2(nthrs_x, nthrs_y);
  const dim3 nblks2(nblks_x, nblks_y);
  // LOG(INFO) << "kernel2 blk dim:" << nblks_x << "*" <<nblks_y << " thr dim:"
  // <<nthrs_x << "*" << nthrs_y;
  gatSumProdZipDivKernel<Idx, DType, true, false>
      <<<nblks2, nthrs2, 0, stream>>>(gdata, incsr_row_ptr.data_ptr<Idx>(),
                                      incsr_col_idx.data_ptr<Idx>(), nullptr,
                                      incsr_num_rows, nullptr, nullptr);
  // cuda_err_chk(cudaPeekAtLastError());
  // cuda_err_chk(cudaDeviceSynchronize());
  //   std::chrono::high_resolution_clock::time_point t2 =
  //       std::chrono::high_resolution_clock::now();
  //   std::cout
  //       << "FusedGatKernelImpl fused<" << 0 << "> time: "
  //       << std::chrono::duration_cast<std::chrono::milliseconds>(t2 -
  //       t1).count()
  //       << " ms" << std::endl;

  // LOG(INFO) << "kernel2 blk dim:" << nblks_x << "*" <<nblks_y << " thr dim:"
  // <<nthrs_x << "*" << nthrs_y;
  //    printf("n_rows: %d\n", incsr.num_rows);
  //    printf("e_xlen: %d\n", gdata.e_xlen);
  //    printf("hidden_xlen: %d\n", gdata.feat_src_xlen/gdata.e_xlen);
  //    printf("stride_head: %d\n", nblks_x * nthrs_x);
  //    printf("stride_vid: %d\n", nblks_y);
  //    printf("dst_vid: %d\n", nthrs_y);
}

template </*int XPU, */ typename Idx, typename DType, bool FLAG_KERNEL_FUSED>
void _BackwardFusedGatKernelImpl_wrapper_integratedcsr(
    at::Tensor& outcsr_row_ptr, at::Tensor& outcsr_col_idx,
    at::Tensor& outcsr_eids, at::Tensor& feat_src, at::Tensor& el,
    at::Tensor& er, at::Tensor& sum, at::Tensor& exp, at::Tensor& ret,
    at::Tensor& grad_out, at::Tensor& grad_feat_src, at::Tensor& grad_el,
    at::Tensor& grad_er,
    // MySimpleNDArray<Idx, thrust::device_allocator<Idx>> eids,
    // //thrust::sequence<Idx>(eids.data.begin(),eids.data.end(), 0); TODO:
    // check if it needs a different eid
    double slope) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  // As GAT only has 1 type of relationship, we use a specialcase of separateCSR
  // where num releationship is asserted as 1
  // assert(outcsr.num_rels == 1);
  // typedef int32_t Idx;
  // typedef float DType;
  const Idx MAX_NBLKS = 65535;
  const Idx MAX_NTHRS = 1024;
  // zero out ret, and packing feat_src, el, er, ret, graph together into one
  // struct using raw float pointers get csr matrix
  BackwardGatFusedData<Idx, DType> gdata;
  int64_t el_xlen = ComputeXLength(el);
  int64_t feat_src_xlen = ComputeXLength(feat_src);
  gdata.feat_src = feat_src.data_ptr<DType>();
  gdata.el = el.data_ptr<DType>();
  gdata.er = er.data_ptr<DType>();
  gdata.sum = sum.data_ptr<DType>();
  gdata.exp = exp.data_ptr<DType>();
  gdata.ret = ret.data_ptr<DType>();
  gdata.grad_out = grad_out.data_ptr<DType>();
  gdata.grad_feat_src = grad_feat_src.data_ptr<DType>();
  gdata.grad_el = grad_el.data_ptr<DType>();
  gdata.grad_er = grad_er.data_ptr<DType>();
  gdata.leaky_relu_slope = slope;
  // gdata.n = el.GetSize()/sizeof(DType)/el_xlen;
  gdata.n = el.numel() / el_xlen;
  gdata.e_xlen = el_xlen;
  gdata.feat_src_xlen = feat_src_xlen;
  gdata.feat_src_hidden = feat_src_xlen / el_xlen;
  // auto outcsr = graph.GetOutCSRMatrix();
  // minigun::Csr<Idx> ocsr = utils::CreateCsr<Idx>(outcsr.indptr,
  // outcsr.indices); gdata.eids =
  // eids.Ptr();//static_cast<Idx*>(outcsr.data->data);
  gdata.eids = outcsr_eids.data_ptr<Idx>();
  // write a device function and call it from here
  // LOG(INFO) << "Within Fused Gat Kernel Impl." << "feat_src_dim:" <<
  // feat_src.GetSize()/sizeof(DType)/feat_src_xlen << "*" << feat_src_xlen
  //    <<" el_dim:" << el.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen  <<
  //    " ret_dim:" << ret.GetSize()/sizeof(DType)/ret_len <<"*" << ret_len
  //    <<" sum_dim:" << sum.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen
  //    <<" exp_dim:" << exp.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen
  //    << " graph csr row_offset length:" <<csr.row_offsets.length << " graph
  //    csr column indices length:" << csr.column_indices.length;
  // print_gdata<Idx, DType>(feat_src,el,er,sum,exp,grad_out,ocsr,el_xlen,
  // feat_src_xlen);
  // Configure kernel launch parameters.
  // auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int nthrs_x = FindNumThreads(el_xlen, 64);
  int nthrs_y = FindNumThreads(gdata.feat_src_hidden, MAX_NTHRS / nthrs_x);
  int nblks_x = 1;
  int nblks_y = std::min(gdata.n, MAX_NBLKS);
  int64_t outcsr_num_rows = outcsr_row_ptr.numel() - 1;
  const dim3 nthrs(nthrs_x, nthrs_y);
  const dim3 nblks(nblks_x, nblks_y);
  // LOG(INFO) << "GradFeatSrc kernel blk dim:" << nblks_x << "*" <<nblks_y << "
  // thr dim:" <<nthrs_x << "*" << nthrs_y;
  // cuda_err_chk(cudaDeviceSynchronize());
  //   std::chrono::high_resolution_clock::time_point t1 =
  //       std::chrono::high_resolution_clock::now();
  if constexpr (!FLAG_KERNEL_FUSED) {
    fusedGatBackwardGradFeatSrc<Idx, DType, true, false>
        <<<nblks, nthrs, 0, stream>>>(gdata, outcsr_row_ptr.data_ptr<Idx>(),
                                      outcsr_col_idx.data_ptr<Idx>(), nullptr,
                                      outcsr_num_rows, nullptr, nullptr);
    // const dim3 nthrs3(nthrs_y, nthrs_x);
    // fusedGatBackwardGradElEr4<<<nblks, nthrs3, 0, thr_entry->stream>>>(gdata,
    // ocsr);
    fusedGatBackwardGradElEr<Idx, DType, true, false>
        <<<nblks, nthrs, 0, stream>>>(gdata, outcsr_row_ptr.data_ptr<Idx>(),
                                      outcsr_col_idx.data_ptr<Idx>(), nullptr,
                                      outcsr_num_rows, nullptr, nullptr);
  } else {
    fusedGatBackwardGradElErFeatSrcFused<Idx, DType, true, false>
        <<<nblks, nthrs, 0, stream>>>(gdata, outcsr_row_ptr.data_ptr<Idx>(),
                                      outcsr_col_idx.data_ptr<Idx>(), nullptr,
                                      outcsr_num_rows, nullptr, nullptr);
  }
  // cuda_err_chk(cudaPeekAtLastError());
  // cuda_err_chk(cudaDeviceSynchronize());
  //   std::chrono::high_resolution_clock::time_point t2 =
  //       std::chrono::high_resolution_clock::now();
  //   std::cout
  //       << "BackwardFusedGatKernelImpl fused<" << FLAG_KERNEL_FUSED << ">
  //       time: "
  //       << std::chrono::duration_cast<std::chrono::milliseconds>(t2 -
  //       t1).count()
  //       << " ms" << std::endl;
}

constexpr auto FusedGatKernelImpl_wrapper_integratedcsr =
    _FusedGatKernelImpl_wrapper_integratedcsr<int64_t, float>;

constexpr auto BackwardFusedGatKernelImpl_wrapper_integratedcsr =
    _BackwardFusedGatKernelImpl_wrapper_integratedcsr<int64_t, float, true>;
