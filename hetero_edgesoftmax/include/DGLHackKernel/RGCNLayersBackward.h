#pragma once
#include "DGLHackKernel.h"

// TODO: the layer 0 and 1 may ends with bias and activation
// the referential implementation from seastar

template <typename Idx, typename DType>
__global__ void RgcnLayer0BackwardKernelImpl(Idx *ranges, Idx *dst_ids,
                                             Idx *eids, Idx *types,
                                             DType *grad_out, DType *norm,
                                             DType *grad_weight, Idx num_nodes,
                                             Idx feat_len, Idx ntypes) {
  if (blockIdx.x < num_nodes) {
    Idx beg = __ldg(ranges + blockIdx.x);
    Idx end = __ldg(ranges + blockIdx.x + 1);
    Idx tx = threadIdx.x;
    for (; tx < feat_len; tx += blockDim.x) {
      for (; beg < end; beg++) {
        Idx dst_id = __ldg(dst_ids + beg);
        Idx eid = __ldg(eids + beg);
        Idx type_id = __ldg(types + beg);
        DType w = __ldg(grad_out + dst_id * feat_len + tx);
        DType n = __ldg(norm + eid);
        grad_weight[type_id * ntypes * feat_len + blockIdx.x * feat_len + tx] =
            w * n;
      }
    }
  }
}

template <typename Idx, typename DType>
__global__ void RgcnLayer1BackwardKernelImpl(
    Idx *ranges, Idx *dst_ids, Idx *eids, Idx *types, DType *hidden,
    DType *weight, DType *norm, DType *grad_out, DType *grad_hidden,
    DType *grad_weight, Idx num_nodes, Idx feat_len_y, Idx feat_len_x,
    Idx ntypes) {
  if (blockIdx.x < num_nodes) {
    Idx beg = __ldg(ranges + blockIdx.x);
    Idx end = __ldg(ranges + blockIdx.x + 1);
    Idx tx = threadIdx.x;
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

// the referential implementation from seastar
template </*int XPU, */ typename Idx, typename DType>
void _RgcnLayerBackwardImpl(
    // GraphRef graph,
    MyHeteroIntegratedCSR<int32_t, thrust::device_allocator<int32_t>>
        transposed_csr,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &hidden,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &weight,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &norm,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &grad_out,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &grad_hidden,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &grad_weight,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &ret,
    bool layer1_flag) {
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
  auto range_data = static_cast<Idx *>(
      thrust::raw_pointer_cast(transposed_csr.row_ptr.data()));
  auto ids_data = static_cast<Idx *>(
      thrust::raw_pointer_cast(transposed_csr.col_idx.data()));
  // auto eids_data = static_cast<Idx*>(thrust::raw_pointer_cast(eids));
  auto eids_data =
      static_cast<Idx *>(thrust::raw_pointer_cast(transposed_csr.eids.data()));
  auto typeids_data = static_cast<Idx *>(
      thrust::raw_pointer_cast(transposed_csr.rel_type.data()));
  auto hidden_data = hidden.Ptr();
  auto weight_data = weight.Ptr();
  auto norm_data = norm.Ptr();
  auto grad_out_data = grad_out.Ptr();
  auto grad_hidden_data = grad_hidden.Ptr();
  auto grad_weight_data = grad_weight.Ptr();
  auto ret_data = ret.Ptr();
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
  Idx num_nodes = transposed_csr.num_rows;
  Idx num_edges = transposed_csr.col_idx.size();
  int nblks = num_nodes;
  // auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  if (layer1_flag) {
    Idx ntypes = weight.shape[0];
    Idx feat_len_y = weight.shape[1];
    Idx feat_len_x = weight.shape[2];
    int nthrs = feat_len_y * feat_len_x;
    RgcnLayer1BackwardKernelImpl<<<nblks, nthrs /*, 0, thr_entry->stream*/>>>(
        range_data, ids_data, eids_data, typeids_data, hidden_data, weight_data,
        norm_data, grad_out_data, grad_hidden_data, grad_weight_data, num_nodes,
        feat_len_y, feat_len_x, ntypes);
  } else {
    Idx ntypes = weight.shape[1];
    Idx feat_len = ret.shape[2];
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
  cuda_err_chk(cudaPeekAtLastError());
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::cout
      << "RGCN Layer 1 backward time: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << " ms" << std::endl;
}

template </*int XPU, */ typename Idx, typename DType>
void RgcnLayer1BackwardImpl(
    // GraphRef graph,
    MyHeteroIntegratedCSR<int32_t, thrust::device_allocator<int32_t>>
        transposed_csr,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &hidden,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &weight,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &norm,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &grad_out,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &grad_hidden,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &grad_weight) {
  _RgcnLayerBackwardImpl<Idx, DType>(
      transposed_csr, hidden, weight, norm, grad_out, grad_hidden, grad_weight,
      MySimpleNDArray<DType, thrust::device_allocator<DType>>(
          std::vector<int64_t>({})),
      true);
}

template </*int XPU, */ typename Idx, typename DType>
void RgcnLayer0BackwardImpl(
    // GraphRef graph,
    MyHeteroIntegratedCSR<int32_t, thrust::device_allocator<int32_t>>
        transposed_csr,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &grad_out,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &norm,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &ret) {
  _RgcnLayerBackwardImpl<Idx, DType>(
      transposed_csr,
      MySimpleNDArray<DType, thrust::device_allocator<DType>>(
          std::vector<int64_t>({})),
      MySimpleNDArray<DType, thrust::device_allocator<DType>>(
          std::vector<int64_t>({})),
      norm, grad_out,
      MySimpleNDArray<DType, thrust::device_allocator<DType>>(
          std::vector<int64_t>({})),
      MySimpleNDArray<DType, thrust::device_allocator<DType>>(
          std::vector<int64_t>({})),
      ret, false);
}
