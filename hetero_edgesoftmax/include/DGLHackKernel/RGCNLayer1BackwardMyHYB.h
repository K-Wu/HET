#pragma once
#include "DGLHackKernel/DGLHackKernel.h"

template <typename Idx, typename DType, int ELL_logical_width,
          int ELL_physical_width>
__global__ void RgcnLayer1BackwardMyHYBKernelImpl(
    const Idx* ellcolidx_data, const Idx* ellreltype_data,
    const Idx* elleids_data, Idx* ranges, Idx* dst_ids, Idx* eids, Idx* types,
    DType* hidden, DType* weight, DType* norm, DType* grad_out,
    DType* grad_hidden, DType* grad_weight, Idx num_nodes, Idx feat_len_y,
    Idx feat_len_x, Idx ntypes) {
  if (blockIdx.x < num_nodes) {
    // ell portion
    Idx ellbeg = ELL_physical_width * blockIdx.x;
    Idx ellend = ELL_physical_width * blockIdx.x + ELL_logical_width;
    Idx tx = threadIdx.x;
    for (; tx < feat_len_x * feat_len_y; tx += blockDim.x) {
      Idx ty = tx / feat_len_x;
      Idx th = tx % feat_len_x;
      DType h = __ldg(hidden + blockIdx.x * feat_len_y + ty);
      DType agg = 0.;
      for (; ellbeg < ellend; ellbeg++) {
        Idx dst_id = __ldg(ellcolidx_data + ellbeg);
        if (dst_id ==
            MyHyb_NONEXISTENT_ELEMENT)  // TODO: check if in transposed hyb
                                        // dst_id is uninitalized and is
                                        // MyHyb_NONEXISTENT_ELEMENT
          break;
        Idx eid = __ldg(elleids_data + ellbeg);
        Idx type_id = __ldg(ellreltype_data + ellbeg);
        DType g = __ldg(grad_out + dst_id * feat_len_x + th);
        DType w = __ldg(weight + type_id * feat_len_y * feat_len_x + tx);
        DType n = __ldg(norm + eid);
        agg += g * w * n;
        atomicAdd(grad_weight + type_id * feat_len_y * feat_len_x + tx,
                  g * h * n);
      }
      atomicAdd(grad_hidden + blockIdx.x * feat_len_y + ty, agg);
    }

    // csr portion
    Idx beg = __ldg(ranges + blockIdx.x);
    Idx end = __ldg(ranges + blockIdx.x + 1);
    // Idx tx = threadIdx.x;
    for (; tx < feat_len_x * feat_len_y; tx += blockDim.x) {
      Idx ty = tx / feat_len_x;
      Idx th = tx % feat_len_x;
      DType h = __ldg(hidden + blockIdx.x * feat_len_y + ty);
      DType agg = 0.;
      for (; beg < end; beg++) {
        Idx dst_id = __ldg(dst_ids + beg);
        Idx eid = __ldg(eids + beg);
        Idx type_id = __ldg(types + beg);
        DType g = __ldg(grad_out + dst_id * feat_len_x + th);
        DType w = __ldg(weight + type_id * feat_len_y * feat_len_x + tx);
        DType n = __ldg(norm + eid);
        agg += g * w * n;
        atomicAdd(grad_weight + type_id * feat_len_y * feat_len_x + tx,
                  g * h * n);
      }
      atomicAdd(grad_hidden + blockIdx.x * feat_len_y + ty, agg);
    }
  }
}

template </*int XPU, */ typename Idx, typename DType, int ELL_logical_width,
          int ELL_physical_width>
void RgcnLayer1BackwardMyHYBImpl(
    // GraphRef graph,
    // MyHeteroIntegratedCSR<int32_t, thrust::device_allocator<int32_t>>
    // transposed_csr,
    MyHyb<int32_t, thrust::device_allocator<int32_t>,
          MyHeteroIntegratedCSR<int32_t, thrust::device_allocator<int32_t>>>
        transposed_hyb,
    MySimpleNDArray<DType, thrust::device_allocator<DType>>& hidden,
    MySimpleNDArray<DType, thrust::device_allocator<DType>>& weight,
    MySimpleNDArray<DType, thrust::device_allocator<DType>>& norm,
    MySimpleNDArray<DType, thrust::device_allocator<DType>>& grad_out,
    MySimpleNDArray<DType, thrust::device_allocator<DType>>& grad_hidden,
    MySimpleNDArray<DType, thrust::device_allocator<DType>>& grad_weight) {
  // MySimpleNDArray<Idx, thrust::device_allocator<Idx>> eids
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

  assert(ELL_logical_width == transposed_hyb.ELL_logical_width);
  assert(ELL_physical_width == transposed_hyb.ELL_physical_width);

  auto ellcolidx_data = static_cast<Idx*>(
      thrust::raw_pointer_cast(transposed_hyb.ELLColIdx.data()));
  auto ellreltype_data = static_cast<Idx*>(
      thrust::raw_pointer_cast(transposed_hyb.ELLRelType.data()));
  auto elleids_data = static_cast<Idx*>(
      thrust::raw_pointer_cast(transposed_hyb.ELLEids.data()));

  auto range_data = static_cast<Idx*>(
      thrust::raw_pointer_cast(transposed_hyb.csr.row_ptr.data()));
  auto ids_data = static_cast<Idx*>(
      thrust::raw_pointer_cast(transposed_hyb.csr.col_idx.data()));
  // auto eids_data = eids.Ptr();
  auto eids_data = static_cast<Idx*>(
      thrust::raw_pointer_cast(transposed_hyb.csr.eids.data()));
  auto typeids_data = static_cast<Idx*>(
      thrust::raw_pointer_cast(transposed_hyb.csr.rel_type.data()));
  auto hidden_data = hidden.Ptr();
  auto weight_data = weight.Ptr();
  auto norm_data = norm.Ptr();
  auto grad_out_data = grad_out.Ptr();
  auto grad_hidden_data = grad_hidden.Ptr();
  auto grad_weight_data = grad_weight.Ptr();
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
  Idx num_nodes = transposed_hyb.num_rows;
  Idx num_edges = transposed_hyb.total_num_nnzs;
  Idx ntypes = weight.shape[0];
  Idx feat_len_y = weight.shape[1];
  Idx feat_len_x = weight.shape[2];
  int nblks = num_nodes;
  int nthrs = feat_len_y * feat_len_x;
  // auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  RgcnLayer1BackwardMyHYBKernelImpl<Idx, DType, ELL_logical_width,
                                    ELL_physical_width><<<nblks, nthrs>>>(
      ellcolidx_data, ellreltype_data, elleids_data, range_data, ids_data,
      eids_data, typeids_data, hidden_data, weight_data, norm_data,
      grad_out_data, grad_hidden_data, grad_weight_data, num_nodes, feat_len_y,
      feat_len_x, ntypes);
  // cudaDeviceSynchronize();
  // auto t2 = std::chrono::steady_clock::now();
  // LOG(INFO) << "layer 1 backward kernel takes:" <<
  // std::chrono::duration_cast<std::chrono::milliseconds>(t2
  // -t1).count()/1000.0
  // << " s";
  cuda_err_chk(cudaPeekAtLastError());
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::cout
      << "MyHYB RGCN Layer 1 backward time: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << " ms" << std::endl;
}