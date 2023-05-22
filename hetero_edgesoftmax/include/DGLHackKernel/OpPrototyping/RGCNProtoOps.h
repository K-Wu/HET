#pragma once

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <npy.hpp>
#include "DGLHackKernel/RGCN/RGCNLayer1BackwardMyHYB.h"
#include "DGLHackKernel/RGCN/RGCNLayer1MyHYB.h"
#include "DGLHackKernel/RGCN/SeastarRGCNBackwardKernels.cu.h"
#include "DGLHackKernel/RGCN/SeastarRGCNKernels.cu.h"
#include "MySimpleNDArray/MySimpleNDArray.h"

namespace HET {
namespace OpPrototyping {
// the referential implementation from seastar
template </*int XPU, */ typename Idx, typename DType>
void _RgcnLayerImpl(
    // GraphRef graph,
    MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>> csr,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &hidden,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &weight,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &norm,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &ret,
    bool layer1_flag) {
  // LOG(INFO) << "Calling implementation of rgn layer 1 forward";
  // assert(csr.IsSortedByEdgeType_CPU());
  // typedef int32_t Idx;
  // typedef float DType;
  // auto csr = graph->GetCsrSortedByEdgeType(false);
  // auto ranges = csr[0];
  // auto ids = csr[1];
  // auto eids = csr[2];
  // auto type_ids = csr[3];
  auto range_data =
      static_cast<Idx *>(thrust::raw_pointer_cast(csr.row_ptr.data()));
  auto ids_data =
      static_cast<Idx *>(thrust::raw_pointer_cast(csr.col_idx.data()));
  // auto eids_data = static_cast<Idx*>(thrust::raw_pointer_cast(eids);
  auto eids_data =
      static_cast<Idx *>(thrust::raw_pointer_cast(csr.eids.data()));
  auto typeids_data =
      static_cast<Idx *>(thrust::raw_pointer_cast(csr.rel_type.data()));
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
  Idx num_nodes = csr.num_rows;
  Idx num_edges = csr.col_idx.size();
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
    HET_Seastar_RgcnLayer1KernelImpl<Idx, DType><<<nblks, nthrs>>>(
        range_data, ids_data, eids_data, typeids_data, hidden_data, weight_data,
        norm_data, ret_data, num_nodes, feat_len_y, feat_len_x, ntypes);
  } else {
    Idx ntypes = weight.shape[1];
    Idx feat_len = weight.shape[2];
    int nthrs = feat_len;
    HET_Seastar_RgcnLayer0KernelImpl<Idx, DType><<<nblks, nthrs>>>(
        range_data, ids_data, eids_data, typeids_data, weight_data, norm_data,
        ret_data, num_nodes, feat_len, ntypes);
  }
  cuda_err_chk(cudaPeekAtLastError());
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::cout
      << "RGCN Layer 1 forward time: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << " ms" << std::endl;
}

// TODO: implement score function and bw for both HGTlayers and rgcnlayers
// probably here

template </*int XPU, */ typename Idx, typename DType>
void RgcnLayer0Impl(
    // GraphRef graph,
    MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>> csr,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &weight,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &norm,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &ret) {
  _RgcnLayerImpl<Idx, DType>(
      csr, MySimpleNDArray<DType, thrust::device_allocator<DType>>({}, nullptr),
      weight, norm, ret, false);
}

template </*int XPU, */ typename Idx, typename DType>
void RgcnLayer1Impl(
    // GraphRef graph,
    MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>> csr,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &hidden,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &weight,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &norm,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &ret) {
  _RgcnLayerImpl<Idx, DType>(csr, hidden, weight, norm, ret, true);
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
    HET_Seastar_RgcnLayer1BackwardKernelImpl<<<nblks, nthrs>>>(
        range_data, ids_data, eids_data, typeids_data, hidden_data, weight_data,
        norm_data, grad_out_data, grad_hidden_data, grad_weight_data, num_nodes,
        feat_len_y, feat_len_x, ntypes);
  } else {
    Idx ntypes = weight.shape[1];
    Idx feat_len = ret.shape[2];
    int nthrs = feat_len;
    HET_Seastar_RgcnLayer0BackwardKernelImpl<<<nblks, nthrs>>>(
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
  MySimpleNDArray<DType, thrust::device_allocator<DType>> ret_dummy(
      std::vector<int64_t>({}));
  _RgcnLayerBackwardImpl<Idx, DType>(transposed_csr, hidden, weight, norm,
                                     grad_out, grad_hidden, grad_weight,
                                     ret_dummy, true);
}

template </*int XPU, */ typename Idx, typename DType>
void RgcnLayer0BackwardImpl(
    // GraphRef graph,
    MyHeteroIntegratedCSR<int32_t, thrust::device_allocator<int32_t>>
        transposed_csr,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &grad_out,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &norm,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &ret) {
  MySimpleNDArray<DType, thrust::device_allocator<DType>> hidden_dummy(
      std::vector<int64_t>({}));
  MySimpleNDArray<DType, thrust::device_allocator<DType>> weight_dummy(
      std::vector<int64_t>({}));
  MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_hidden_dummy(
      std::vector<int64_t>({}));
  MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_weight_dummy(
      std::vector<int64_t>({}));

  _RgcnLayerBackwardImpl<Idx, DType>(transposed_csr, hidden_dummy, weight_dummy,
                                     norm, grad_out, grad_hidden_dummy,
                                     grad_weight_dummy, ret, false);
}

// TODO: implement its equivalent in OpExport
template </*int XPU, */ typename Idx, typename DType, int ELL_logical_width,
          int ELL_physical_width>
void RgcnLayer1BackwardMyHYBImpl(
    // GraphRef graph,
    // MyHeteroIntegratedCSR<int32_t, thrust::device_allocator<int32_t>>
    // transposed_csr,
    MyHyb<int32_t, thrust::device_allocator<int32_t>,
          MyHeteroIntegratedCSR<int32_t, thrust::device_allocator<int32_t>>>
        transposed_hyb,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &hidden,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &weight,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &norm,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &grad_out,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &grad_hidden,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &grad_weight) {
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

  auto ellcolidx_data = static_cast<Idx *>(
      thrust::raw_pointer_cast(transposed_hyb.ELLColIdx.data()));
  auto ellreltype_data = static_cast<Idx *>(
      thrust::raw_pointer_cast(transposed_hyb.ELLRelType.data()));
  auto elleids_data = static_cast<Idx *>(
      thrust::raw_pointer_cast(transposed_hyb.ELLEids.data()));

  auto range_data = static_cast<Idx *>(
      thrust::raw_pointer_cast(transposed_hyb.csr.row_ptr.data()));
  auto ids_data = static_cast<Idx *>(
      thrust::raw_pointer_cast(transposed_hyb.csr.col_idx.data()));
  // auto eids_data = eids.Ptr();
  auto eids_data = static_cast<Idx *>(
      thrust::raw_pointer_cast(transposed_hyb.csr.eids.data()));
  auto typeids_data = static_cast<Idx *>(
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
  HET_Seastar_RgcnLayer1BackwardMyHYBKernelImpl<Idx, DType, ELL_logical_width,
                                                ELL_physical_width>
      <<<nblks, nthrs>>>(ellcolidx_data, ellreltype_data, elleids_data,
                         range_data, ids_data, eids_data, typeids_data,
                         hidden_data, weight_data, norm_data, grad_out_data,
                         grad_hidden_data, grad_weight_data, num_nodes,
                         feat_len_y, feat_len_x, ntypes);
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

// TODO: implement its equivalent in OpExport
template </*int XPU, */ typename Idx, typename DType, int ELL_logical_width,
          int ELL_physical_width>
void RgcnLayer1MyHYBImpl(
    // GraphRef graph,
    // MyHeteroIntegratedCSR<int32_t, thrust::device_allocator<int32_t>> csr,
    MyHyb<int32_t, thrust::device_allocator<int32_t>,
          MyHeteroIntegratedCSR<int32_t, thrust::device_allocator<int32_t>>>
        hyb,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &hidden,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &weight,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &norm,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &ret) {
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
      static_cast<Idx *>(thrust::raw_pointer_cast(hyb.ELLColIdx.data()));
  auto ellreltype_data =
      static_cast<Idx *>(thrust::raw_pointer_cast(hyb.ELLRelType.data()));
  auto elleids_data =
      static_cast<Idx *>(thrust::raw_pointer_cast(hyb.ELLEids.data()));

  auto range_data =
      static_cast<Idx *>(thrust::raw_pointer_cast(hyb.csr.row_ptr.data()));
  auto ids_data =
      static_cast<Idx *>(thrust::raw_pointer_cast(hyb.csr.col_idx.data()));
  // auto eids_data = eids.Ptr();
  auto eids_data =
      static_cast<Idx *>(thrust::raw_pointer_cast(hyb.csr.eids.data()));
  auto typeids_data =
      static_cast<Idx *>(thrust::raw_pointer_cast(hyb.csr.rel_type.data()));

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
  HET_Seastar_RgcnLayer1MyHYBKernelImpl<Idx, DType, ELL_logical_width,
                                        ELL_physical_width><<<nblks, nthrs>>>(
      ellcolidx_data, ellreltype_data, elleids_data, range_data, ids_data,
      eids_data, typeids_data, hidden_data, weight_data, norm_data, ret_data,
      num_nodes, feat_len_y, feat_len_x, ntypes);
  cuda_err_chk(cudaPeekAtLastError());
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::cout
      << "MyHYB RGCN Layer 1 forward time: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << " ms" << std::endl;
}

}  // namespace OpPrototyping
}  // namespace HET