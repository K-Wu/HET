#pragma once
#include "DGLHackKernel/DGLHackKernel.h"

// bgs:
template <typename Idx, typename DType, int ELL_logical_width,
          int ELL_physical_width>
__global__ void RgcnLayer1MyHYBKernelImpl(
    const Idx* ellcolidx_data, const Idx* ellreltype_data,
    const Idx* elleids_data, const Idx* ranges, const Idx* src_ids,
    const Idx* eids, const Idx* types, const DType* hidden, const DType* weight,
    const DType* norm, DType* ret, Idx num_nodes, Idx feat_len_y,
    Idx feat_len_x, Idx ntypes) {
  // ell portion
  if (blockIdx.x < num_nodes) {
    Idx ell_beg = ELL_physical_width * blockIdx.x;
    Idx ell_end = ELL_physical_width * blockIdx.x + ELL_logical_width;
    Idx tx = threadIdx.x;
    Idx ty = threadIdx.x / feat_len_x;
    Idx th = threadIdx.x % feat_len_x;
    DType agg_val = 0.;
    DType w = 0.;
    Idx cur_type_id = -1;
    for (; ell_beg < ell_end; ell_beg++) {
      Idx src_id = __ldg(ellcolidx_data + ell_beg);
      if (src_id == MyHyb_NONEXISTENT_ELEMENT) break;
      Idx eid = __ldg(elleids_data + ell_beg);
      Idx type_id = __ldg(ellreltype_data + ell_beg);
      if (type_id != cur_type_id) {
        w = __ldg(weight + type_id * feat_len_y * feat_len_x + tx);
      }
      DType h = __ldg(hidden + src_id * feat_len_y + ty);
      DType n = __ldg(norm + eid);
      agg_val += h * w * n;
    }
    // atomicAdd(ret + blockIdx.x*feat_len_x + th, agg_val);
    //}

    // csr portion
    // if (blockIdx.x < num_nodes) {
    Idx beg = __ldg(ranges + blockIdx.x);
    Idx end = __ldg(ranges + blockIdx.x + 1);
    // Idx tx = threadIdx.x;
    // Idx ty = threadIdx.x / feat_len_x;
    // Idx th = threadIdx.x % feat_len_x;
    // DType agg_val = 0.;
    // DType w = 0.;
    // Idx cur_type_id = -1;
    for (; beg < end; beg++) {
      Idx src_id = __ldg(src_ids + beg);
      Idx eid = __ldg(eids + beg);
      Idx type_id = __ldg(types + beg);
      if (type_id != cur_type_id) {
        w = __ldg(weight + type_id * feat_len_y * feat_len_x + tx);
      }
      DType h = __ldg(hidden + src_id * feat_len_y + ty);
      DType n = __ldg(norm + eid);
      agg_val += h * w * n;
    }
    atomicAdd(ret + blockIdx.x * feat_len_x + th, agg_val);
  }
}

template </*int XPU, */ typename Idx, typename DType, int ELL_logical_width,
          int ELL_physical_width>
void RgcnLayer1MyHYBImpl(
    // GraphRef graph,
    // MyHeteroIntegratedCSR<int32_t, thrust::device_allocator<int32_t>> csr,
    MyHyb<int32_t, thrust::device_allocator<int32_t>,
          MyHeteroIntegratedCSR<int32_t, thrust::device_allocator<int32_t>>>
        hyb,
    MySimpleNDArray<DType, thrust::device_allocator<DType>>& hidden,
    MySimpleNDArray<DType, thrust::device_allocator<DType>>& weight,
    MySimpleNDArray<DType, thrust::device_allocator<DType>>& norm,
    MySimpleNDArray<DType, thrust::device_allocator<DType>>& ret) {
  // MySimpleNDArray<Idx, thrust::device_allocator<Idx>> eids
  // LOG(INFO) << "Calling implementation of rgn layer 1 forward";
  // assert(csr.IsSortedByEdgeType_CPU());
  // typedef int32_t Idx;
  // typedef float DType;
  // auto csr = graph->GetCsrSortedByEdgeType(false);
  // auto ranges = csr[0];
  // auto ids = csr[1];
  // auto eids = csr[2];
  // auto type_ids = csr[3];

  assert(ELL_logical_width == hyb.ELL_logical_width);
  assert(ELL_physical_width == hyb.ELL_physical_width);

  auto ellcolidx_data =
      static_cast<Idx*>(thrust::raw_pointer_cast(hyb.ELLColIdx.data()));
  auto ellreltype_data =
      static_cast<Idx*>(thrust::raw_pointer_cast(hyb.ELLRelType.data()));
  auto elleids_data =
      static_cast<Idx*>(thrust::raw_pointer_cast(hyb.ELLEids.data()));

  auto range_data =
      static_cast<Idx*>(thrust::raw_pointer_cast(hyb.csr.row_ptr.data()));
  auto ids_data =
      static_cast<Idx*>(thrust::raw_pointer_cast(hyb.csr.col_idx.data()));
  // auto eids_data = eids.Ptr();
  auto eids_data =
      static_cast<Idx*>(thrust::raw_pointer_cast(hyb.csr.eids.data()));
  auto typeids_data =
      static_cast<Idx*>(thrust::raw_pointer_cast(hyb.csr.rel_type.data()));

  auto hidden_data = hidden.Ptr();
  auto weight_data = weight.Ptr();
  auto norm_data = norm.Ptr();
  auto ret_data = ret.Ptr();
  // print_dims(hidden);
  // print_dims(weight);
  // print_dims(norm);
  // print_dims(ret);
  // Idx num_nodes = ranges->shape[0] - 1;
  // Idx num_edges = eids->shape[0];
  Idx num_nodes = hyb.num_rows;
  Idx num_edges = hyb.total_num_nnzs;
  Idx ntypes = weight.shape[0];
  Idx feat_len_y = weight.shape[1];
  Idx feat_len_x = weight.shape[2];
  int nblks = num_nodes;
  int nthrs = feat_len_y * feat_len_x;
  // auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  RgcnLayer1MyHYBKernelImpl<Idx, DType, ELL_logical_width, ELL_physical_width>
      <<<nblks, nthrs /*, 0, thr_entry->stream*/>>>(
          ellcolidx_data, ellreltype_data, elleids_data, range_data, ids_data,
          eids_data, typeids_data, hidden_data, weight_data, norm_data,
          ret_data, num_nodes, feat_len_y, feat_len_x, ntypes);
  cuda_err_chk(cudaPeekAtLastError());
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::cout
      << "MyHYB RGCN Layer 1 forward time: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << " ms" << std::endl;
}