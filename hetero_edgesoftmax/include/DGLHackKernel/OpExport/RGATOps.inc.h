#pragma once
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "DGLHackKernel/DGLHackUtils.h"
#include "DGLHackKernel/GAT/FusedGAT.cu.h"
#include "DGLHackKernel/GAT/FusedGATBackward.cu.h"
#include "DGLHackKernel/RGAT/RGATLayersBackwardKernelsSeparateCOO.cu.h"
#include "DGLHackKernel/RGAT/RGATLayersBackwardKernelsSeparateCSR.cu.h"
#include "DGLHackKernel/RGAT/RGATLayersKernelsSeparateCOO.cu.h"
#include "DGLHackKernel/RGAT/RGATLayersKernelsSeparateCSR.cu.h"
#include "DGLHackKernel/mysgemm/my_shmem_sgemm_func.cu.h"
#include "DGLHackKernel/mysgemm/mysgemm_KernelsBlockConfigurations.h"
#include "GATOps.inc.h"

template </*int XPU, */ typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool IntegratedFormatRatherThanSeparateFlag,
          bool CSRRatherThanCOOFlag>
void _RelationalFusedGATKernel_wrapper(
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_rel_ptrs,
    at::Tensor& separate_coo_row_indices, at::Tensor& separate_coo_col_indices,
    at::Tensor& incsr_row_ptr, at::Tensor& incsr_col_idx,
    at::Tensor& incsr_eids, at::Tensor& incsr_reltypes,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, double slope) {
  const Idx MAX_NBLKS = 65535;
  const Idx MAX_NTHRS = 1024;
  GatFusedData<Idx, DType> gdata;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  int64_t el_xlen = SeastarComputeXLength<>(el);
  int64_t feat_src_xlen = SeastarComputeXLength<>(feat_src);
  int64_t ret_len = SeastarComputeXLength<>(ret);
  // NB: in this case gdata.n, calculation is removed since el is now per edge
  // rather than per node
  gdata.feat_src = feat_src.data_ptr<DType>();
  gdata.el = el.data_ptr<DType>();
  gdata.er = er.data_ptr<DType>();
  gdata.sum = sum.data_ptr<DType>();
  gdata.exp = exp.data_ptr<DType>();
  gdata.ret = ret.data_ptr<DType>();
  gdata.leaky_relu_slope = slope;
  // gdata.n = el.numel() / el_xlen;
  gdata.e_xlen = el_xlen;
  gdata.feat_src_xlen = feat_src_xlen;
  gdata.feat_src_hidden = feat_src_xlen / el_xlen;
  gdata.ret_xlen = ret_len;

  if constexpr (IntegratedFormatRatherThanSeparateFlag &&
                CSRRatherThanCOOFlag) {
    // Integrated CSR
    gdata.eids = incsr_eids.data_ptr<Idx>();
    // Configure kernel launch parameters.
    int nthrs_x = 32;
    int nthrs_y = 1;
    int nblks_x = (el_xlen + nthrs_x - 1) / (nthrs_x);
    int64_t incsr_num_rows = incsr_row_ptr.numel() - 1;
    int nblks_y = std::min(incsr_num_rows, MAX_NBLKS);
    const dim3 nblks(nblks_x, nblks_y);
    const dim3 nthrs(nthrs_x, nthrs_y);

    gatExpLeakyReluSumKernel<Idx, DType, CompactAsOfNodeFlag, true>
        <<<nblks, nthrs, 0, stream>>>(
            gdata, incsr_row_ptr.data_ptr<Idx>(), incsr_col_idx.data_ptr<Idx>(),
            incsr_reltypes.data_ptr<Idx>(), incsr_num_rows,
            unique_srcs_and_dests_rel_ptr.data_ptr<Idx>(),
            unique_srcs_and_dests_node_indices.data_ptr<Idx>());

    nthrs_x = SeastarFindNumThreads(el_xlen, 64);
    nthrs_y = SeastarFindNumThreads(gdata.feat_src_hidden, MAX_NTHRS / nthrs_x);
    nblks_x = 1;
    nblks_y = std::min(incsr_num_rows, MAX_NBLKS);
    const dim3 nthrs2(nthrs_x, nthrs_y);
    const dim3 nblks2(nblks_x, nblks_y);

    gatSumProdZipDivKernel<Idx, DType, CompactAsOfNodeFlag, true>
        <<<nblks2, nthrs2, 0, stream>>>(
            gdata, incsr_row_ptr.data_ptr<Idx>(), incsr_col_idx.data_ptr<Idx>(),
            incsr_reltypes.data_ptr<Idx>(), incsr_num_rows,
            unique_srcs_and_dests_rel_ptr.data_ptr<Idx>(),
            unique_srcs_and_dests_node_indices.data_ptr<Idx>());
  } else if constexpr (!IntegratedFormatRatherThanSeparateFlag &&
                       !CSRRatherThanCOOFlag) {
    // separate coo
    gdata.eids = separate_coo_eids.data_ptr<Idx>();
    int64_t num_edges = separate_coo_row_indices.numel();
    int64_t num_relations = separate_coo_rel_ptrs.numel() - 1;
    int nthrs_x = 32;
    int nthrs_y = 1;
    int nblks_x = (el_xlen + nthrs_x - 1) / (nthrs_x);
    int nblks_y = std::min(num_edges, MAX_NBLKS);
    const dim3 nblks(nblks_x, nblks_y);
    const dim3 nthrs(nthrs_x, nthrs_y);

    gatExpLeakyReluSumKernel_relational_separate_coo<Idx, DType,
                                                     CompactAsOfNodeFlag>
        <<<nblks, nthrs, 0, stream>>>(
            gdata, separate_coo_rel_ptrs.data_ptr<Idx>(),
            separate_coo_row_indices.data_ptr<Idx>(),
            separate_coo_col_indices.data_ptr<Idx>(), num_edges,
            unique_srcs_and_dests_rel_ptr.data_ptr<Idx>(),
            unique_srcs_and_dests_node_indices.data_ptr<Idx>(), num_relations);

    nthrs_x = SeastarFindNumThreads(el_xlen, 64);
    nthrs_y = SeastarFindNumThreads(gdata.feat_src_hidden, MAX_NTHRS / nthrs_x);
    nblks_x = 1;
    nblks_y = std::min(num_edges, MAX_NBLKS);
    const dim3 nthrs2(nthrs_x, nthrs_y);
    const dim3 nblks2(nblks_x, nblks_y);
    gatSumProdZipDivKernel_relational_separate_coo<Idx, DType,
                                                   CompactAsOfNodeFlag>
        <<<nblks2, nthrs2, 0, stream>>>(
            gdata, separate_coo_rel_ptrs.data_ptr<Idx>(),
            separate_coo_row_indices.data_ptr<Idx>(),
            separate_coo_col_indices.data_ptr<Idx>(), num_edges,
            unique_srcs_and_dests_rel_ptr.data_ptr<Idx>(),
            unique_srcs_and_dests_node_indices.data_ptr<Idx>(), num_relations);

  } else {
    assert(0 && "Not implemented");
  }
}

template </*int XPU, */ typename Idx, typename DType, bool FLAG_KERNEL_FUSED,
          bool CompactAsOfNodeFlag, bool IntegratedFormatRatherThanSeparateFlag,
          bool CSRRatherThanCOOFlag>
void _BackwardRelationalFusedGATKernel_wrapper(
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_rel_ptrs,
    at::Tensor& separate_coo_row_indices, at::Tensor& separate_coo_col_indices,
    at::Tensor& outcsr_row_ptr, at::Tensor& outcsr_col_idx,
    at::Tensor& outcsr_eids, at::Tensor& outcsr_reltypes,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, at::Tensor& gradout, at::Tensor& grad_feat_src,
    at::Tensor& grad_el, at::Tensor& grad_er, double slope) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const Idx MAX_NBLKS = 65535;
  const Idx MAX_NTHRS = 1024;
  BackwardGatFusedData<Idx, DType> gdata;
  int64_t el_xlen = SeastarComputeXLength<>(el);
  int64_t feat_src_xlen = SeastarComputeXLength<>(feat_src);
  gdata.feat_src = feat_src.data_ptr<DType>();
  gdata.el = el.data_ptr<DType>();
  gdata.er = er.data_ptr<DType>();
  gdata.sum = sum.data_ptr<DType>();
  gdata.exp = exp.data_ptr<DType>();
  gdata.ret = ret.data_ptr<DType>();
  gdata.grad_out = gradout.data_ptr<DType>();
  gdata.grad_feat_src = grad_feat_src.data_ptr<DType>();
  gdata.grad_el = grad_el.data_ptr<DType>();
  gdata.grad_er = grad_er.data_ptr<DType>();
  gdata.leaky_relu_slope = slope;
  // gdata.n = el.numel() / el_xlen;
  gdata.e_xlen = el_xlen;
  gdata.feat_src_xlen = feat_src_xlen;
  gdata.feat_src_hidden = feat_src_xlen / el_xlen;

  if constexpr (IntegratedFormatRatherThanSeparateFlag &&
                CSRRatherThanCOOFlag) {
    // Integrated CSR
    gdata.eids = outcsr_eids.data_ptr<Idx>();
    int nthrs_x = SeastarFindNumThreads(el_xlen, 64);
    int nthrs_y =
        SeastarFindNumThreads(gdata.feat_src_hidden, MAX_NTHRS / nthrs_x);
    int nblks_x = 1;
    int64_t outcsr_num_rows = outcsr_row_ptr.numel() - 1;
    int nblks_y = std::min(outcsr_num_rows, MAX_NBLKS);
    const dim3 nthrs(nthrs_x, nthrs_y);
    const dim3 nblks(nblks_x, nblks_y);
    if constexpr (!FLAG_KERNEL_FUSED) {
      fusedGatBackwardGradFeatSrc<Idx, DType, CompactAsOfNodeFlag, true>
          <<<nblks, nthrs, 0, stream>>>(
              gdata, outcsr_row_ptr.data_ptr<Idx>(),
              outcsr_col_idx.data_ptr<Idx>(), outcsr_reltypes.data_ptr<Idx>(),
              outcsr_num_rows, unique_srcs_and_dests_rel_ptr.data_ptr<Idx>(),
              unique_srcs_and_dests_node_indices.data_ptr<Idx>());
      fusedGatBackwardGradElEr<Idx, DType, CompactAsOfNodeFlag, true>
          <<<nblks, nthrs, 0, stream>>>(
              gdata, outcsr_row_ptr.data_ptr<Idx>(),
              outcsr_col_idx.data_ptr<Idx>(), outcsr_reltypes.data_ptr<Idx>(),
              outcsr_num_rows, unique_srcs_and_dests_rel_ptr.data_ptr<Idx>(),
              unique_srcs_and_dests_node_indices.data_ptr<Idx>());
    } else {
      fusedGatBackwardGradElErFeatSrcFused<Idx, DType, CompactAsOfNodeFlag,
                                           true><<<nblks, nthrs, 0, stream>>>(
          gdata, outcsr_row_ptr.data_ptr<Idx>(), outcsr_col_idx.data_ptr<Idx>(),
          outcsr_reltypes.data_ptr<Idx>(), outcsr_num_rows,
          unique_srcs_and_dests_rel_ptr.data_ptr<Idx>(),
          unique_srcs_and_dests_node_indices.data_ptr<Idx>());
    }
  } else if constexpr (!IntegratedFormatRatherThanSeparateFlag &&
                       !CSRRatherThanCOOFlag) {
    // separate coo
    gdata.eids = separate_coo_eids.data_ptr<Idx>();
    int64_t num_edges = separate_coo_row_indices.numel();
    int64_t num_relations = separate_coo_rel_ptrs.numel() - 1;
    int nthrs_x = SeastarFindNumThreads(el_xlen, 64);
    int nthrs_y =
        SeastarFindNumThreads(gdata.feat_src_hidden, MAX_NTHRS / nthrs_x);
    int nblks_x = 1;
    int nblks_y = std::min(num_edges, MAX_NBLKS);
    int64_t outcsr_num_rows = outcsr_row_ptr.numel() - 1;
    const dim3 nthrs(nthrs_x, nthrs_y);
    const dim3 nblks(nblks_x, nblks_y);
    if constexpr (!FLAG_KERNEL_FUSED) {
      fusedGatBackwardGradFeatSrc_relational_separate_coo<Idx, DType,
                                                          CompactAsOfNodeFlag>
          <<<nblks, nthrs, 0, stream>>>(
              gdata, separate_coo_rel_ptrs.data_ptr<Idx>(),
              separate_coo_row_indices.data_ptr<Idx>(),
              separate_coo_col_indices.data_ptr<Idx>(), num_edges,
              unique_srcs_and_dests_rel_ptr.data_ptr<Idx>(),
              unique_srcs_and_dests_node_indices.data_ptr<Idx>(),
              num_relations);
      fusedGatBackwardGradElEr_relational_separate_coo<Idx, DType,
                                                       CompactAsOfNodeFlag>
          <<<nblks, nthrs, 0, stream>>>(
              gdata, separate_coo_rel_ptrs.data_ptr<Idx>(),
              separate_coo_row_indices.data_ptr<Idx>(),
              separate_coo_col_indices.data_ptr<Idx>(), num_edges,
              unique_srcs_and_dests_rel_ptr.data_ptr<Idx>(),
              unique_srcs_and_dests_node_indices.data_ptr<Idx>(),
              num_relations);
    } else {
      fusedGatBackwardGradElErFeatSrcFused_relational_separate_coo<
          Idx, DType, CompactAsOfNodeFlag><<<nblks, nthrs, 0, stream>>>(
          gdata, separate_coo_rel_ptrs.data_ptr<Idx>(),
          separate_coo_row_indices.data_ptr<Idx>(),
          separate_coo_col_indices.data_ptr<Idx>(), num_edges,
          unique_srcs_and_dests_rel_ptr.data_ptr<Idx>(),
          unique_srcs_and_dests_node_indices.data_ptr<Idx>(), num_relations);
    }
  } else {
    assert(0 && "Not implemented");
  }
}

void RelationalFusedGATKernel_wrapper_integratedcsr(
    at::Tensor& incsr_row_ptr, at::Tensor& incsr_col_idx,
    at::Tensor& incsr_eids, at::Tensor& incsr_reltypes,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, double slope) {
  at::Tensor dummy_tensor;
  _RelationalFusedGATKernel_wrapper<int64_t, float, false, true, true>(
      dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, incsr_row_ptr,
      incsr_col_idx, incsr_eids, incsr_reltypes, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices, feat_src, el, er, sum, exp, ret,
      slope);
}
void BackwardRelationalFusedGATKernel_wrapper_integratedcsr(
    at::Tensor& outcsr_row_ptr, at::Tensor& outcsr_col_idx,
    at::Tensor& outcsr_eids, at::Tensor& outcsr_reltypes,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, at::Tensor& gradout, at::Tensor& grad_feat_src,
    at::Tensor& grad_el, at::Tensor& grad_er, double slope) {
  at::Tensor dummy_tensor;
  _BackwardRelationalFusedGATKernel_wrapper<int64_t, float, true, false, true,
                                            true>(
      dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, outcsr_row_ptr,
      outcsr_col_idx, outcsr_eids, outcsr_reltypes,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
      feat_src, el, er, sum, exp, ret, gradout, grad_feat_src, grad_el, grad_er,
      slope);
}
void RGATRelationalFusedGATKernelCompactAsOfNode_wrapper_integratedcsr(
    at::Tensor& incsr_row_ptr, at::Tensor& incsr_col_idx,
    at::Tensor& incsr_eids, at::Tensor& incsr_reltypes,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, double slope) {
  at::Tensor dummy_tensor;
  _RelationalFusedGATKernel_wrapper<int64_t, float, true, true, true>(
      dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, incsr_row_ptr,
      incsr_col_idx, incsr_eids, incsr_reltypes, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices, feat_src, el, er, sum, exp, ret,
      slope);
}
void BackwardRGATRelationalFusedGATKernelCompactAsOfNode_wrapper_integratedcsr(
    at::Tensor& outcsr_row_ptr, at::Tensor& outcsr_col_idx,
    at::Tensor& outcsr_eids, at::Tensor& outcsr_reltypes,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, at::Tensor& gradout, at::Tensor& grad_feat_src,
    at::Tensor& grad_el, at::Tensor& grad_er, double slope) {
  at::Tensor dummy_tensor;
  _BackwardRelationalFusedGATKernel_wrapper<int64_t, float, true, true, true,
                                            true>(
      dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, outcsr_row_ptr,
      outcsr_col_idx, outcsr_eids, outcsr_reltypes,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
      feat_src, el, er, sum, exp, ret, gradout, grad_feat_src, grad_el, grad_er,
      slope);
}

void BackwardRGATRelationalFusedGATKernelCompactAsOfNode_wrapper_edge_parallel_separatecoo(
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_rel_ptrs,
    at::Tensor& separate_coo_row_indices, at::Tensor& separate_coo_col_indices,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, at::Tensor& gradout, at::Tensor& grad_feat_src,
    at::Tensor& grad_el, at::Tensor& grad_er, double slope) {
  at::Tensor dummy_tensor;
  _BackwardRelationalFusedGATKernel_wrapper<int64_t, float, true, true, false,
                                            false>(
      separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
      separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
      dummy_tensor, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices, feat_src, el, er, sum, exp, ret,
      gradout, grad_feat_src, grad_el, grad_er, slope);
}

void RGATRelationalFusedGATKernelCompactAsOfNode_wrapper_edge_parallel_separatecoo(
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_rel_ptrs,
    at::Tensor& separate_coo_row_indices, at::Tensor& separate_coo_col_indices,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, double slope) {
  at::Tensor dummy_tensor;
  _RelationalFusedGATKernel_wrapper<int64_t, float, true, false, false>(
      separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
      separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
      dummy_tensor, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices, feat_src, el, er, sum, exp, ret,
      slope);
}

void RelationalFusedGATKernel_wrapper_edge_parallel_separatecoo(
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_rel_ptrs,
    at::Tensor& separate_coo_row_indices, at::Tensor& separate_coo_col_indices,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, double slope) {
  at::Tensor dummy_tensor;
  _RelationalFusedGATKernel_wrapper<int64_t, float, false, false, false>(
      separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
      separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
      dummy_tensor, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices, feat_src, el, er, sum, exp, ret,
      slope);
}

void BackwardRelationalFusedGATKernel_wrapper_edge_parallel_separatecoo(
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_rel_ptrs,
    at::Tensor& separate_coo_row_indices, at::Tensor& separate_coo_col_indices,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, at::Tensor& gradout, at::Tensor& grad_feat_src,
    at::Tensor& grad_el, at::Tensor& grad_er, double slope) {
  at::Tensor dummy_tensor;
  _BackwardRelationalFusedGATKernel_wrapper<int64_t, float, false, false, false,
                                            false>(
      separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
      separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
      dummy_tensor, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices, feat_src, el, er, sum, exp, ret,
      gradout, grad_feat_src, grad_el, grad_er, slope);
}

template <int BLOCK_SIZE, bool CompactAsOfNodeFlag,
          bool ACGatherScatterListIdenticalFlag>
void _RGATRelationalMatMul_wrapper_separatecoo(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_node_indices,
    at::Tensor& separate_coo_eids, at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& weights,
    at::Tensor& input, at::Tensor& ret) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int64_t num_relations = separate_coo_relptrs.numel() - 1;
  const int64_t num_heads = weights.size(1);
  const int64_t num_input_dim = weights.size(2);
  const int64_t num_output_per_head_dim =
      weights.size(3);  // weight shape (num_relations, n_heads,
                        // in_feat, out_feat // n_heads)

  int64_t num_edges;
  if constexpr (CompactAsOfNodeFlag) {
    num_edges = unique_srcs_and_dests_node_indices.numel();
  } else {
    num_edges = separate_coo_eids.numel();
  }
  int grid_dim_y = std::min(
      ceil_div<>(num_edges, (int64_t)BLOCK_SIZE),
      (int64_t)32768);  // using 32768 instead of 65535 to leave some space in
                        // case the total number of blocks is slightly larger
                        // due to relationship with very few workloads
  std::vector<int> num_blocks_assignment_for_same_relation_vect,
      num_blocks_assignment_for_all_prev_relation_vect;
  if constexpr (CompactAsOfNodeFlag) {
    at::Tensor unique_srcs_and_dests_rel_ptr_cpu_contiguous =
        unique_srcs_and_dests_rel_ptr.cpu().contiguous();
    std::tie(num_blocks_assignment_for_same_relation_vect,
             num_blocks_assignment_for_all_prev_relation_vect) =
        get_schedule_by_relation_kernel_launch_metadata<false, false, int64_t*>(
            grid_dim_y, num_relations, BLOCK_SIZE,
            unique_srcs_and_dests_rel_ptr_cpu_contiguous.data_ptr<int64_t>(),
            unique_srcs_and_dests_rel_ptr_cpu_contiguous.data_ptr<int64_t>() +
                num_relations);
  } else {
    at::Tensor separate_coo_relptrs_cpu_contiguous =
        separate_coo_relptrs.cpu().contiguous();
    std::tie(num_blocks_assignment_for_same_relation_vect,
             num_blocks_assignment_for_all_prev_relation_vect) =
        get_schedule_by_relation_kernel_launch_metadata<false, false, int64_t*>(
            grid_dim_y, num_relations, BLOCK_SIZE,
            separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>(),
            separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>() +
                num_relations);
  }
  grid_dim_y = num_blocks_assignment_for_all_prev_relation_vect.back();

  thrust::device_vector<int> dev_num_blocks_assignment_for_same_relation_vect(
      num_blocks_assignment_for_same_relation_vect.begin(),
      num_blocks_assignment_for_same_relation_vect.end());
  thrust::device_vector<int>
      dev_num_blocks_assignment_for_all_prev_relation_vect(
          num_blocks_assignment_for_all_prev_relation_vect.begin(),
          num_blocks_assignment_for_all_prev_relation_vect.end());

  if constexpr (CompactAsOfNodeFlag) {
    if constexpr (ACGatherScatterListIdenticalFlag) {
      CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
          CompactAsOfNodeFlag && ACGatherScatterListIdenticalFlag,
          "CompactAsOfNodeFlag && ACGatherScatterListIdenticalFlag");
    }
    const dim3 nblks(ceil_div<>(num_output_per_head_dim, (long)BLOCK_SIZE),
                     grid_dim_y, num_heads);
    const dim3 nthrs(BLOCK_SIZE, BLOCK_SIZE);
    std::cout << "nblks.x: " << nblks.x << " nblks.y: " << nblks.y
              << " nblks.z: " << nblks.z << std::endl;
    RGNNFeatCompactFWProp<BLOCK_SIZE, int64_t, int64_t*>
        <<<nblks, nthrs, 0, stream>>>(
            input.data_ptr<float>(), weights.data_ptr<float>(),
            ret.data_ptr<float>(),
            unique_srcs_and_dests_rel_ptr.data_ptr<int64_t>(),
            unique_srcs_and_dests_node_indices.data_ptr<int64_t>(),
            num_input_dim, num_output_per_head_dim, num_heads,
            thrust::raw_pointer_cast(
                dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
            num_relations);
  } else {
    const dim3 nblks(ceil_div<>(num_output_per_head_dim, (long)BLOCK_SIZE),
                     grid_dim_y, num_heads);
    const dim3 nthrs(BLOCK_SIZE, BLOCK_SIZE);
    std::cout << "nblks.x: " << nblks.x << " nblks.y: " << nblks.y
              << " nblks.z: " << nblks.z << std::endl;
    if constexpr (ACGatherScatterListIdenticalFlag) {
      RGNNFeatPerEdgeFWPropACGatherScatterListIdentical<BLOCK_SIZE, int64_t,
                                                        int64_t*>
          <<<nblks, nthrs, 0, stream>>>(
              input.data_ptr<float>(), weights.data_ptr<float>(),
              ret.data_ptr<float>(), separate_coo_relptrs.data_ptr<int64_t>(),
              separate_coo_eids.data_ptr<int64_t>(), num_input_dim,
              num_output_per_head_dim, num_heads,
              thrust::raw_pointer_cast(
                  dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
              num_relations);
    } else {
      RGNNFeatPerEdgeFWProp<BLOCK_SIZE, int64_t, int64_t*>
          <<<nblks, nthrs, 0, stream>>>(
              input.data_ptr<float>(), weights.data_ptr<float>(),
              ret.data_ptr<float>(),
              separate_coo_node_indices.data_ptr<int64_t>(),
              separate_coo_relptrs.data_ptr<int64_t>(),
              separate_coo_eids.data_ptr<int64_t>(), num_input_dim,
              num_output_per_head_dim, num_heads,
              thrust::raw_pointer_cast(
                  dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
              num_relations);
    }
  }
}

void RGATRelationalMatMul_wrapper_separatecoo(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_node_indices,
    at::Tensor& separate_coo_eids, at::Tensor& weights, at::Tensor& input,
    at::Tensor& ret) {
  at::Tensor dummy_tensor;
  _RGATRelationalMatMul_wrapper_separatecoo<16, false, false>(
      separate_coo_relptrs, separate_coo_node_indices, separate_coo_eids,
      dummy_tensor, dummy_tensor, weights, input, ret);
}

void RGATRelationalMatMul_ACGatherScatterListIdentical_wrapper_separatecoo(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_eids,
    at::Tensor& weights, at::Tensor& input, at::Tensor& ret) {
  at::Tensor dummy_tensor;
  _RGATRelationalMatMul_wrapper_separatecoo<16, false, true>(
      separate_coo_relptrs, dummy_tensor, separate_coo_eids, dummy_tensor,
      dummy_tensor, weights, input, ret);
}

void RGATRelationalMatMulCompactAsOfNode_wrapper_unique_rel_node_indices(
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& weight,
    at::Tensor& node_feat, at::Tensor& ret) {
  at::Tensor dummy_tensor;
  _RGATRelationalMatMul_wrapper_separatecoo<16, true, false>(
      dummy_tensor, dummy_tensor, dummy_tensor, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices, weight, node_feat, ret);
}

template <int BLOCK_SIZE, bool CompactAsOfNodeFlag,
          bool ACGatherScatterListIdenticalFlag>
void _BackwardRGATRelationalMatMul_wrapper_separatecoo(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_node_indices,
    at::Tensor& separate_coo_eids, at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices,
    at::Tensor& weights_transposed, at::Tensor& input, at::Tensor& gradout,
    at::Tensor& grad_input, at::Tensor& grad_weights) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int64_t num_relations = separate_coo_relptrs.numel() - 1;
  const int64_t num_heads = weights_transposed.size(1);
  const int64_t num_input_dim = weights_transposed.size(3);
  const int64_t num_output_per_head_dim =
      weights_transposed.size(2);  // weight shape (num_relations, n_heads,
                                   // in_feat, out_feat // n_heads)
  at::Tensor separate_coo_relptrs_cpu_contiguous =
      separate_coo_relptrs.cpu().contiguous();
  int64_t num_edges;
  if constexpr (CompactAsOfNodeFlag) {
    num_edges = unique_srcs_and_dests_node_indices.numel();
  } else {
    num_edges = separate_coo_eids.numel();
  }
  int grid_dim_y = std::min(
      ceil_div<>(num_edges, (int64_t)BLOCK_SIZE),
      (int64_t)32768);  // using 32768 instead of 65535 to leave some space in
                        // case the total number of blocks is slightly larger
                        // due to relationship with very few workloads
  std::vector<int> num_blocks_assignment_for_same_relation_vect,
      num_blocks_assignment_for_all_prev_relation_vect;
  if constexpr (CompactAsOfNodeFlag) {
    at::Tensor unique_srcs_and_dests_rel_ptr_cpu_contiguous =
        unique_srcs_and_dests_rel_ptr.cpu().contiguous();
    std::tie(num_blocks_assignment_for_same_relation_vect,
             num_blocks_assignment_for_all_prev_relation_vect) =
        get_schedule_by_relation_kernel_launch_metadata<false, false, int64_t*>(
            grid_dim_y, num_relations, BLOCK_SIZE,
            unique_srcs_and_dests_rel_ptr_cpu_contiguous.data_ptr<int64_t>(),
            unique_srcs_and_dests_rel_ptr_cpu_contiguous.data_ptr<int64_t>() +
                num_relations);
  } else {
    at::Tensor separate_coo_relptrs_cpu_contiguous =
        separate_coo_relptrs.cpu().contiguous();
    std::tie(num_blocks_assignment_for_same_relation_vect,
             num_blocks_assignment_for_all_prev_relation_vect) =
        get_schedule_by_relation_kernel_launch_metadata<false, false, int64_t*>(
            grid_dim_y, num_relations, BLOCK_SIZE,
            separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>(),
            separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>() +
                num_relations);
  }
  grid_dim_y = num_blocks_assignment_for_all_prev_relation_vect.back();
  // print num_blocks_assignment_for_all_prev_relation_vect
  // std::cout << "num_blocks_assignment_for_all_prev_relation_vect: [";
  // for (auto i : num_blocks_assignment_for_all_prev_relation_vect) {
  // std::cout << i << " ";
  // }
  // std::cout << "]"<< std::endl;
  thrust::device_vector<int> dev_num_blocks_assignment_for_same_relation_vect(
      num_blocks_assignment_for_same_relation_vect.begin(),
      num_blocks_assignment_for_same_relation_vect.end());
  thrust::device_vector<int>
      dev_num_blocks_assignment_for_all_prev_relation_vect(
          num_blocks_assignment_for_all_prev_relation_vect.begin(),
          num_blocks_assignment_for_all_prev_relation_vect.end());

  if constexpr (CompactAsOfNodeFlag) {
    if constexpr (ACGatherScatterListIdenticalFlag) {
      CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
          CompactAsOfNodeFlag && ACGatherScatterListIdenticalFlag,
          "CompactAsOfNodeFlag && ACGatherScatterListIdenticalFlag");
    }
    const dim3 nblks(ceil_div<>(num_output_per_head_dim, (long)BLOCK_SIZE),
                     grid_dim_y, num_heads);
    const dim3 nblks_outer_product(
        ceil_div<>(num_output_per_head_dim, (long)BLOCK_SIZE), grid_dim_y,
        num_heads * 128);

    const dim3 nthrs(BLOCK_SIZE, BLOCK_SIZE);
    RGNNDeltaNodeFeatInputCompactBWProp<BLOCK_SIZE, int64_t, int64_t*>
        <<<nblks, nthrs, 0, stream>>>(
            gradout.data_ptr<float>(), weights_transposed.data_ptr<float>(),
            grad_input.data_ptr<float>(),
            unique_srcs_and_dests_rel_ptr.data_ptr<int64_t>(),
            unique_srcs_and_dests_node_indices.data_ptr<int64_t>(), num_edges,
            num_output_per_head_dim, num_input_dim, num_heads,
            thrust::raw_pointer_cast(
                dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
            num_relations);
    RGNNDeltaWeightCompactBWProp<BLOCK_SIZE, int64_t, int64_t*>
        <<<nblks_outer_product, nthrs, 0, stream>>>(
            gradout.data_ptr<float>(), input.data_ptr<float>(),
            grad_weights.data_ptr<float>(),
            unique_srcs_and_dests_rel_ptr.data_ptr<int64_t>(),
            unique_srcs_and_dests_node_indices.data_ptr<int64_t>(), num_edges,
            num_output_per_head_dim, num_input_dim, num_heads,
            thrust::raw_pointer_cast(
                dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
            num_relations);
  } else {
    const dim3 nblks(ceil_div<>(num_output_per_head_dim, (long)BLOCK_SIZE),
                     grid_dim_y, num_heads);
    const dim3 nblks_outer_product(
        ceil_div<>(num_output_per_head_dim, (long)BLOCK_SIZE), grid_dim_y,
        num_heads * 128);
    const dim3 nthrs(BLOCK_SIZE, BLOCK_SIZE);
    if constexpr (ACGatherScatterListIdenticalFlag) {
      RGNNDeltaNodeFeatInputBWPropACGatherScatterListIdentical<
          BLOCK_SIZE, int64_t, int64_t*><<<nblks, nthrs, 0, stream>>>(
          gradout.data_ptr<float>(), weights_transposed.data_ptr<float>(),
          grad_input.data_ptr<float>(), separate_coo_eids.data_ptr<int64_t>(),
          separate_coo_relptrs.data_ptr<int64_t>(), num_output_per_head_dim,
          num_input_dim, num_heads,
          thrust::raw_pointer_cast(
              dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
          num_relations);
      RGNNDeltaWeightBWPropACGatherScatterListIdentical<BLOCK_SIZE, int64_t,
                                                        int64_t*>
          <<<nblks_outer_product, nthrs, 0, stream>>>(
              input.data_ptr<float>(), gradout.data_ptr<float>(),
              grad_weights.data_ptr<float>(),
              separate_coo_eids.data_ptr<int64_t>(),
              separate_coo_relptrs.data_ptr<int64_t>(), num_output_per_head_dim,
              num_input_dim, num_heads,
              thrust::raw_pointer_cast(
                  dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
              num_relations);
    } else {
      RGNNDeltaNodeFeatInputBWProp<BLOCK_SIZE, int64_t, int64_t*>
          <<<nblks, nthrs, 0, stream>>>(
              gradout.data_ptr<float>(), weights_transposed.data_ptr<float>(),
              grad_input.data_ptr<float>(),
              separate_coo_eids.data_ptr<int64_t>(),
              separate_coo_relptrs.data_ptr<int64_t>(),
              separate_coo_node_indices.data_ptr<int64_t>(),
              num_output_per_head_dim, num_input_dim, num_heads,
              thrust::raw_pointer_cast(
                  dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
              num_relations);
      RGNNDeltaWeightBWProp<BLOCK_SIZE, int64_t, int64_t*>
          <<<nblks_outer_product, nthrs, 0, stream>>>(
              input.data_ptr<float>(), gradout.data_ptr<float>(),
              grad_weights.data_ptr<float>(),
              separate_coo_eids.data_ptr<int64_t>(),
              separate_coo_relptrs.data_ptr<int64_t>(),
              separate_coo_node_indices.data_ptr<int64_t>(),
              num_output_per_head_dim, num_input_dim, num_heads,
              thrust::raw_pointer_cast(
                  dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
              num_relations);
    }
  }
}

void BackwardRGATRelationalMatMulCompactAsOfNode_wrapper_unique_rel_node_indices(
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices,
    at::Tensor& weights_transposed, at::Tensor& input, at::Tensor& gradout,
    at::Tensor& grad_input, at::Tensor& grad_weights) {
  at::Tensor dummy_tensor;
  _BackwardRGATRelationalMatMul_wrapper_separatecoo<16, true, false>(
      dummy_tensor, dummy_tensor, dummy_tensor, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices, weights_transposed, input, gradout,
      grad_input, grad_weights);
}

void BackwardRGATRelationalMatMul_wrapper_separatecoo(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_node_indices,
    at::Tensor& separate_coo_eids, at::Tensor& weights_transposed,
    at::Tensor& input, at::Tensor& gradout, at::Tensor& grad_input,
    at::Tensor& grad_weights) {
  at::Tensor dummy_tensor;
  _BackwardRGATRelationalMatMul_wrapper_separatecoo<16, false, false>(
      separate_coo_relptrs, separate_coo_node_indices, separate_coo_eids,
      dummy_tensor, dummy_tensor, weights_transposed, input, gradout,
      grad_input, grad_weights);
}

void BackwardRGATRelationalMatMul_ACGatherScatterListIdentical_wrapper_separatecoo(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_eids,
    at::Tensor& weights_transposed, at::Tensor& input, at::Tensor& gradout,
    at::Tensor& grad_input, at::Tensor& grad_weights) {
  at::Tensor dummy_tensor;
  _BackwardRGATRelationalMatMul_wrapper_separatecoo<16, false, true>(
      separate_coo_relptrs, dummy_tensor, separate_coo_eids, dummy_tensor,
      dummy_tensor, weights_transposed, input, gradout, grad_input,
      grad_weights);
}