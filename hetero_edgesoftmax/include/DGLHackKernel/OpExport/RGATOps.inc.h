#pragma once
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "DGLHackKernel/DGLHackUtils.h"
#include "DGLHackKernel/GAT/FusedGAT.cu.h"
#include "DGLHackKernel/GAT/FusedGATBackward.cu.h"
#include "DGLHackKernel/RGAT/RGATBackwardKernelsSeparateCOO.cu.h"
#include "DGLHackKernel/RGAT/RGATBackwardKernelsSeparateCSR.cu.h"
#include "DGLHackKernel/RGAT/RGATKernelsSeparateCOO.cu.h"
#include "DGLHackKernel/RGAT/RGATKernelsSeparateCSR.cu.h"
#include "GATOps.inc.h"

namespace HET {
namespace TorchExport {
namespace RGAT {
namespace FwProp {
template </*int XPU, */ typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool IntegratedFormatRatherThanSeparateFlag,
          bool CSRRatherThanCOOFlag, bool DualUniqueNodeList>
void _RelationalFusedGATKernel(
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_rel_ptrs,
    at::Tensor& separate_coo_row_indices, at::Tensor& separate_coo_col_indices,
    at::Tensor& incsr_row_ptr, at::Tensor& incsr_col_idx,
    at::Tensor& incsr_eids, at::Tensor& incsr_reltypes,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_rel_ptr_col,
    at::Tensor& unique_srcs_and_dests_node_indices,
    at::Tensor& unique_srcs_and_dests_node_indices_col, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, double slope) {
  const Idx MAX_NBLKS = 65535;
  const Idx MAX_NTHRS = 1024;
  GatFusedData<Idx, DType> gdata;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  gdata.num_heads = SeastarComputeXLength<>(el);
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
  gdata.feat_src_xlen = feat_src_xlen;
  // gdata.feat_src_hidden = feat_src_xlen / el_xlen;
  // gdata.ret_xlen = ret_len;

  if constexpr (IntegratedFormatRatherThanSeparateFlag &&
                CSRRatherThanCOOFlag) {
    CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(IntegratedFormatRatherThanSeparateFlag &&
                                          CSRRatherThanCOOFlag &&
                                          DualUniqueNodeList,
                                      "not implemented");
    // Integrated CSR
    gdata.eids = incsr_eids.data_ptr<Idx>();
    // Configure kernel launch parameters.
    // NB: Type 1 schedule addresses that we can safely reshape (nthrs_x,
    // nthrs_y) to assign more y dimension to rows as usually n_head is smaller
    // than 32

    // NB: updated to Type 1 Schedule:
    // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-069c3c2c5a9041df2c9a0b01c9f28044c4d519d86c5ed2f859d0d74282967062L232-R233
    // head -> blockIdx.x * blockDim.x + threadIdx.x;
    // edge|node -> blockIdx.y * blockDim.y + threadIdx.y;
    int nthrs_x = 1;
    int nthrs_y = 32;
    int nblks_x = (gdata.num_heads + nthrs_x - 1) / (nthrs_x);
    int64_t incsr_num_rows = incsr_row_ptr.numel() - 1;
    int nblks_y =
        std::min(ceil_div(incsr_num_rows, (int64_t)nthrs_y), MAX_NBLKS);
    const dim3 nblks(nblks_x, nblks_y);
    const dim3 nthrs(nthrs_x, nthrs_y);

    HET_gatExpLeakyReluSumKernel<Idx, DType, CompactAsOfNodeFlag, true>
        <<<nblks, nthrs, 0, stream>>>(
            gdata, incsr_row_ptr.data_ptr<Idx>(), incsr_col_idx.data_ptr<Idx>(),
            incsr_reltypes.data_ptr<Idx>(), incsr_num_rows,
            CompactAsOfNodeFlag ? unique_srcs_and_dests_rel_ptr.data_ptr<Idx>()
                                : nullptr,
            CompactAsOfNodeFlag
                ? unique_srcs_and_dests_node_indices.data_ptr<Idx>()
                : nullptr);

    // NB: updated to Type 2 Schedule:
    // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
    // head -> threadIdx.y
    // node -> blockIdx.y
    // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x
    nthrs_y = SeastarFindNumThreads(gdata.num_heads, 64);
    nthrs_x = SeastarFindNumThreads(feat_src_xlen / gdata.num_heads,
                                    MAX_NTHRS / nthrs_y);
    nblks_x = 1;
    nblks_y = std::min(incsr_num_rows, MAX_NBLKS);
    const dim3 nthrs2(nthrs_x, nthrs_y);
    const dim3 nblks2(nblks_x, nblks_y);

    HET_gatSumProdZipDivKernel<Idx, DType, CompactAsOfNodeFlag, true>
        <<<nblks2, nthrs2, 0, stream>>>(
            gdata, incsr_row_ptr.data_ptr<Idx>(), incsr_col_idx.data_ptr<Idx>(),
            incsr_reltypes.data_ptr<Idx>(), incsr_num_rows,
            CompactAsOfNodeFlag ? unique_srcs_and_dests_rel_ptr.data_ptr<Idx>()
                                : nullptr,
            CompactAsOfNodeFlag
                ? unique_srcs_and_dests_node_indices.data_ptr<Idx>()
                : nullptr);
  } else if constexpr (!IntegratedFormatRatherThanSeparateFlag &&
                       !CSRRatherThanCOOFlag) {
    // separate coo
    gdata.eids = separate_coo_eids.data_ptr<Idx>();
    int64_t num_edges = separate_coo_row_indices.numel();
    int64_t num_relations = separate_coo_rel_ptrs.numel() - 1;
    // TODO: we can safely reshape (nthrs_x, nthrs_y) to assign more y dimension
    // to edges as usually n_head is smaller than 32
    // NB: updated to Type 1 Schedule:
    // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-069c3c2c5a9041df2c9a0b01c9f28044c4d519d86c5ed2f859d0d74282967062L232-R233
    // head -> blockIdx.x * blockDim.x + threadIdx.x;
    // edge|node -> blockIdx.y * blockDim.y + threadIdx.y;
    int nthrs_x = 1;
    int nthrs_y = 32;
    int nblks_x = (gdata.num_heads + nthrs_x - 1) / (nthrs_x);
    int nblks_y = std::min(ceil_div(num_edges, (int64_t)nthrs_y), MAX_NBLKS);
    const dim3 nblks(nblks_x, nblks_y);
    const dim3 nthrs(nthrs_x, nthrs_y);

    HET_gatExpLeakyReluSumKernel_relational_separate_coo<
        Idx, DType, CompactAsOfNodeFlag, DualUniqueNodeList>
        <<<nblks, nthrs, 0, stream>>>(
            gdata, separate_coo_rel_ptrs.data_ptr<Idx>(),
            separate_coo_row_indices.data_ptr<Idx>(),
            separate_coo_col_indices.data_ptr<Idx>(), num_edges,
            CompactAsOfNodeFlag ? unique_srcs_and_dests_rel_ptr.data_ptr<Idx>()
                                : nullptr,
            DualUniqueNodeList
                ? unique_srcs_and_dests_rel_ptr_col.data_ptr<Idx>()
                : nullptr,
            CompactAsOfNodeFlag
                ? unique_srcs_and_dests_node_indices.data_ptr<Idx>()
                : nullptr,
            DualUniqueNodeList
                ? unique_srcs_and_dests_node_indices_col.data_ptr<Idx>()
                : nullptr,
            num_relations);

    // NB: updated to Type 2 Schedule:
    // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
    // head -> threadIdx.y
    // node -> blockIdx.y
    // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x
    nthrs_y = SeastarFindNumThreads(gdata.num_heads, 64);
    nthrs_x = SeastarFindNumThreads(feat_src_xlen / gdata.num_heads,
                                    MAX_NTHRS / nthrs_y);
    nblks_x = 1;
    nblks_y = std::min(num_edges, MAX_NBLKS);
    const dim3 nthrs2(nthrs_x, nthrs_y);
    const dim3 nblks2(nblks_x, nblks_y);
    HET_gatSumProdZipDivKernel_relational_separate_coo<
        Idx, DType, CompactAsOfNodeFlag, DualUniqueNodeList>
        <<<nblks2, nthrs2, 0, stream>>>(
            gdata, separate_coo_rel_ptrs.data_ptr<Idx>(),
            separate_coo_row_indices.data_ptr<Idx>(),
            separate_coo_col_indices.data_ptr<Idx>(), num_edges,
            CompactAsOfNodeFlag ? unique_srcs_and_dests_rel_ptr.data_ptr<Idx>()
                                : nullptr,
            DualUniqueNodeList
                ? unique_srcs_and_dests_rel_ptr_col.data_ptr<Idx>()
                : nullptr,
            CompactAsOfNodeFlag
                ? unique_srcs_and_dests_node_indices.data_ptr<Idx>()
                : nullptr,
            DualUniqueNodeList
                ? unique_srcs_and_dests_node_indices_col.data_ptr<Idx>()
                : nullptr,
            num_relations);

  } else {
    assert(0 && "Not implemented");
  }
}

void RelationalFusedGATKernelCompactAsOfNode_edge_parallel_separatecoo_dual_unique_node_list(
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_rel_ptrs,
    at::Tensor& separate_coo_row_indices, at::Tensor& separate_coo_col_indices,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_rel_ptr_col,
    at::Tensor& unique_srcs_and_dests_node_indices_row,
    at::Tensor& unique_srcs_and_dests_node_indices_col, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, double slope) {
  at::Tensor dummy_tensor;
  _RelationalFusedGATKernel<int64_t, float, true, false, false, true>(
      separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
      separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
      dummy_tensor, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_rel_ptr_col, unique_srcs_and_dests_node_indices_row,
      unique_srcs_and_dests_node_indices_col, feat_src, el, er, sum, exp, ret,
      slope);
}

void RelationalFusedGATKernelCompactAsOfNode_edge_parallel_separatecoo(
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_rel_ptrs,
    at::Tensor& separate_coo_row_indices, at::Tensor& separate_coo_col_indices,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, double slope) {
  at::Tensor dummy_tensor;
  _RelationalFusedGATKernel<int64_t, float, true, false, false, false>(
      separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
      separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
      dummy_tensor, unique_srcs_and_dests_rel_ptr, dummy_tensor,
      unique_srcs_and_dests_node_indices, dummy_tensor, feat_src, el, er, sum,
      exp, ret, slope);
}

void RelationalFusedGATKernel_edge_parallel_separatecoo(
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_rel_ptrs,
    at::Tensor& separate_coo_row_indices, at::Tensor& separate_coo_col_indices,
    at::Tensor& feat_src, at::Tensor& el, at::Tensor& er, at::Tensor& sum,
    at::Tensor& exp, at::Tensor& ret, double slope) {
  at::Tensor dummy_tensor;
  _RelationalFusedGATKernel<int64_t, float, false, false, false, false>(
      separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
      separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
      dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor,
      feat_src, el, er, sum, exp, ret, slope);
}

void RelationalFusedGATKernel_integratedcsr(
    at::Tensor& incsr_row_ptr, at::Tensor& incsr_col_idx,
    at::Tensor& incsr_eids, at::Tensor& incsr_reltypes,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, double slope) {
  at::Tensor dummy_tensor;
  _RelationalFusedGATKernel<int64_t, float, false, true, true, false>(
      dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, incsr_row_ptr,
      incsr_col_idx, incsr_eids, incsr_reltypes, unique_srcs_and_dests_rel_ptr,
      dummy_tensor, unique_srcs_and_dests_node_indices, dummy_tensor, feat_src,
      el, er, sum, exp, ret, slope);
}

void RelationalFusedGATKernelCompactAsOfNode_integratedcsr(
    at::Tensor& incsr_row_ptr, at::Tensor& incsr_col_idx,
    at::Tensor& incsr_eids, at::Tensor& incsr_reltypes,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, double slope) {
  at::Tensor dummy_tensor;
  _RelationalFusedGATKernel<int64_t, float, true, true, true, false>(
      dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, incsr_row_ptr,
      incsr_col_idx, incsr_eids, incsr_reltypes, unique_srcs_and_dests_rel_ptr,
      dummy_tensor, unique_srcs_and_dests_node_indices, dummy_tensor, feat_src,
      el, er, sum, exp, ret, slope);
}
}  // namespace FwProp
namespace BckProp {
template </*int XPU, */ typename Idx, typename DType, bool FLAG_KERNEL_FUSED,
          bool CompactAsOfNodeFlag, bool IntegratedFormatRatherThanSeparateFlag,
          bool CSRRatherThanCOOFlag, bool DualUniqueNodeList>
void _RelationalFusedGATKernel(
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_rel_ptrs,
    at::Tensor& separate_coo_row_indices, at::Tensor& separate_coo_col_indices,
    at::Tensor& outcsr_row_ptr, at::Tensor& outcsr_col_idx,
    at::Tensor& outcsr_eids, at::Tensor& outcsr_reltypes,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_rel_ptr_col,
    at::Tensor& unique_srcs_and_dests_node_indices,
    at::Tensor& unique_srcs_and_dests_node_indices_col, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, at::Tensor& gradout, at::Tensor& grad_feat_src,
    at::Tensor& grad_el, at::Tensor& grad_er, double slope) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const Idx MAX_NBLKS = 65535;
  const Idx MAX_NTHRS = 1024;
  BackwardGatFusedData<Idx, DType> gdata;
  gdata.num_heads = SeastarComputeXLength<>(el);
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
  gdata.feat_src_xlen = feat_src_xlen;
  // gdata.feat_src_hidden = feat_src_xlen / el_xlen;

  if constexpr (IntegratedFormatRatherThanSeparateFlag &&
                CSRRatherThanCOOFlag) {
    CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
        DualUniqueNodeList && IntegratedFormatRatherThanSeparateFlag &&
            CSRRatherThanCOOFlag,
        "DualUniqueNodeList && IntegratedFormatRatherThanSeparateFlag&& "
        "CSRRatherThanCOOFlag");
    // Integrated CSR
    gdata.eids = outcsr_eids.data_ptr<Idx>();
    // NB: updated to follow Type 2 Schedule:
    // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
    // head -> threadIdx.y
    // node -> blockIdx.y
    // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x
    int nthrs_y = SeastarFindNumThreads(gdata.num_heads, 64);
    int nthrs_x = SeastarFindNumThreads(feat_src_xlen / gdata.num_heads,
                                        MAX_NTHRS / nthrs_y);
    int nblks_x = 1;
    int64_t outcsr_num_rows = outcsr_row_ptr.numel() - 1;
    int nblks_y = std::min(outcsr_num_rows, MAX_NBLKS);
    const dim3 nthrs(nthrs_x, nthrs_y);
    const dim3 nblks(nblks_x, nblks_y);
    if constexpr (!FLAG_KERNEL_FUSED) {
      HET_fusedGatBackwardGradFeatSrc<Idx, DType, CompactAsOfNodeFlag, true>
          <<<nblks, nthrs, 0, stream>>>(
              gdata, outcsr_row_ptr.data_ptr<Idx>(),
              outcsr_col_idx.data_ptr<Idx>(), outcsr_reltypes.data_ptr<Idx>(),
              outcsr_num_rows,
              CompactAsOfNodeFlag
                  ? unique_srcs_and_dests_rel_ptr.data_ptr<Idx>()
                  : nullptr,
              CompactAsOfNodeFlag
                  ? unique_srcs_and_dests_node_indices.data_ptr<Idx>()
                  : nullptr);
      HET_fusedGatBackwardGradElEr<Idx, DType, CompactAsOfNodeFlag, true>
          <<<nblks, nthrs, 0, stream>>>(
              gdata, outcsr_row_ptr.data_ptr<Idx>(),
              outcsr_col_idx.data_ptr<Idx>(), outcsr_reltypes.data_ptr<Idx>(),
              outcsr_num_rows,
              CompactAsOfNodeFlag
                  ? unique_srcs_and_dests_rel_ptr.data_ptr<Idx>()
                  : nullptr,
              CompactAsOfNodeFlag
                  ? unique_srcs_and_dests_node_indices.data_ptr<Idx>()
                  : nullptr);
    } else {
      HET_fusedGatBackwardGradElErFeatSrcFused<
          Idx, DType, CompactAsOfNodeFlag, true><<<nblks, nthrs, 0, stream>>>(
          gdata, outcsr_row_ptr.data_ptr<Idx>(), outcsr_col_idx.data_ptr<Idx>(),
          outcsr_reltypes.data_ptr<Idx>(), outcsr_num_rows,
          CompactAsOfNodeFlag ? unique_srcs_and_dests_rel_ptr.data_ptr<Idx>()
                              : nullptr,
          CompactAsOfNodeFlag
              ? unique_srcs_and_dests_node_indices.data_ptr<Idx>()
              : nullptr);
    }
  } else if constexpr (!IntegratedFormatRatherThanSeparateFlag &&
                       !CSRRatherThanCOOFlag) {
    // separate coo
    gdata.eids = separate_coo_eids.data_ptr<Idx>();
    int64_t num_edges = separate_coo_row_indices.numel();
    int64_t num_relations = separate_coo_rel_ptrs.numel() - 1;

    // NB: updated to Type 2 Schedule:
    // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
    // head -> threadIdx.y
    // node -> blockIdx.y
    // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x
    int nthrs_y = SeastarFindNumThreads(gdata.num_heads, 64);
    int nthrs_x = SeastarFindNumThreads(feat_src_xlen / gdata.num_heads,
                                        MAX_NTHRS / nthrs_y);
    int nblks_x = 1;
    int nblks_y = std::min(num_edges, MAX_NBLKS);
    int64_t outcsr_num_rows = outcsr_row_ptr.numel() - 1;
    const dim3 nthrs(nthrs_x, nthrs_y);
    const dim3 nblks(nblks_x, nblks_y);
    if constexpr (!FLAG_KERNEL_FUSED) {
      HET_fusedGatBackwardGradFeatSrc_relational_separate_coo<
          Idx, DType, CompactAsOfNodeFlag, DualUniqueNodeList>
          <<<nblks, nthrs, 0, stream>>>(
              gdata, separate_coo_rel_ptrs.data_ptr<Idx>(),
              separate_coo_row_indices.data_ptr<Idx>(),
              separate_coo_col_indices.data_ptr<Idx>(), num_edges,
              CompactAsOfNodeFlag
                  ? unique_srcs_and_dests_rel_ptr.data_ptr<Idx>()
                  : nullptr,
              DualUniqueNodeList
                  ? unique_srcs_and_dests_rel_ptr_col.data_ptr<Idx>()
                  : nullptr,
              CompactAsOfNodeFlag
                  ? unique_srcs_and_dests_node_indices.data_ptr<Idx>()
                  : nullptr,
              DualUniqueNodeList
                  ? unique_srcs_and_dests_node_indices_col.data_ptr<Idx>()
                  : nullptr,
              num_relations);
      HET_fusedGatBackwardGradElEr_relational_separate_coo<
          Idx, DType, CompactAsOfNodeFlag, DualUniqueNodeList>
          <<<nblks, nthrs, 0, stream>>>(
              gdata, separate_coo_rel_ptrs.data_ptr<Idx>(),
              separate_coo_row_indices.data_ptr<Idx>(),
              separate_coo_col_indices.data_ptr<Idx>(), num_edges,
              CompactAsOfNodeFlag
                  ? unique_srcs_and_dests_rel_ptr.data_ptr<Idx>()
                  : nullptr,
              DualUniqueNodeList
                  ? unique_srcs_and_dests_rel_ptr_col.data_ptr<Idx>()
                  : nullptr,
              CompactAsOfNodeFlag
                  ? unique_srcs_and_dests_node_indices.data_ptr<Idx>()
                  : nullptr,
              DualUniqueNodeList
                  ? unique_srcs_and_dests_node_indices_col.data_ptr<Idx>()
                  : nullptr,
              num_relations);
    } else {
      HET_fusedGatBackwardGradElErFeatSrcFused_relational_separate_coo<
          Idx, DType, CompactAsOfNodeFlag, DualUniqueNodeList>
          <<<nblks, nthrs, 0, stream>>>(
              gdata, separate_coo_rel_ptrs.data_ptr<Idx>(),
              separate_coo_row_indices.data_ptr<Idx>(),
              separate_coo_col_indices.data_ptr<Idx>(), num_edges,
              CompactAsOfNodeFlag
                  ? unique_srcs_and_dests_rel_ptr.data_ptr<Idx>()
                  : nullptr,
              DualUniqueNodeList
                  ? unique_srcs_and_dests_rel_ptr_col.data_ptr<Idx>()
                  : nullptr,
              CompactAsOfNodeFlag
                  ? unique_srcs_and_dests_node_indices.data_ptr<Idx>()
                  : nullptr,
              DualUniqueNodeList
                  ? unique_srcs_and_dests_node_indices_col.data_ptr<Idx>()
                  : nullptr,
              num_relations);
    }
  } else {
    assert(0 && "Not implemented");
  }
}

void RelationalFusedGATKernel_integratedcsr(
    at::Tensor& outcsr_row_ptr, at::Tensor& outcsr_col_idx,
    at::Tensor& outcsr_eids, at::Tensor& outcsr_reltypes,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, at::Tensor& gradout, at::Tensor& grad_feat_src,
    at::Tensor& grad_el, at::Tensor& grad_er, double slope) {
  at::Tensor dummy_tensor;
  _RelationalFusedGATKernel<int64_t, float, true, false, true, true, false>(
      dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, outcsr_row_ptr,
      outcsr_col_idx, outcsr_eids, outcsr_reltypes,
      unique_srcs_and_dests_rel_ptr, dummy_tensor,
      unique_srcs_and_dests_node_indices, dummy_tensor, feat_src, el, er, sum,
      exp, ret, gradout, grad_feat_src, grad_el, grad_er, slope);
}

void RelationalFusedGATKernelCompactAsOfNode_integratedcsr(
    at::Tensor& outcsr_row_ptr, at::Tensor& outcsr_col_idx,
    at::Tensor& outcsr_eids, at::Tensor& outcsr_reltypes,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, at::Tensor& gradout, at::Tensor& grad_feat_src,
    at::Tensor& grad_el, at::Tensor& grad_er, double slope) {
  at::Tensor dummy_tensor;
  _RelationalFusedGATKernel<int64_t, float, true, true, true, true, false>(
      dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, outcsr_row_ptr,
      outcsr_col_idx, outcsr_eids, outcsr_reltypes,
      unique_srcs_and_dests_rel_ptr, dummy_tensor,
      unique_srcs_and_dests_node_indices, dummy_tensor, feat_src, el, er, sum,
      exp, ret, gradout, grad_feat_src, grad_el, grad_er, slope);
}

void RelationalFusedGATKernelCompactAsOfNode_edge_parallel_separatecoo_dual_unique_node_list(
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_rel_ptrs,
    at::Tensor& separate_coo_row_indices, at::Tensor& separate_coo_col_indices,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_rel_ptr_col,
    at::Tensor& unique_srcs_and_dests_node_indices,
    at::Tensor& unique_srcs_and_dests_node_indices_col, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, at::Tensor& gradout, at::Tensor& grad_feat_src,
    at::Tensor& grad_el, at::Tensor& grad_er, double slope) {
  at::Tensor dummy_tensor;
  _RelationalFusedGATKernel<int64_t, float, true, true, false, false, true>(
      separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
      separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
      dummy_tensor, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_rel_ptr_col, unique_srcs_and_dests_node_indices,
      unique_srcs_and_dests_node_indices_col, feat_src, el, er, sum, exp, ret,
      gradout, grad_feat_src, grad_el, grad_er, slope);
}

void RelationalFusedGATKernelCompactAsOfNode_edge_parallel_separatecoo(
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_rel_ptrs,
    at::Tensor& separate_coo_row_indices, at::Tensor& separate_coo_col_indices,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, at::Tensor& gradout, at::Tensor& grad_feat_src,
    at::Tensor& grad_el, at::Tensor& grad_er, double slope) {
  at::Tensor dummy_tensor;
  _RelationalFusedGATKernel<int64_t, float, true, true, false, false, false>(
      separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
      separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
      dummy_tensor, unique_srcs_and_dests_rel_ptr, dummy_tensor,
      unique_srcs_and_dests_node_indices, dummy_tensor, feat_src, el, er, sum,
      exp, ret, gradout, grad_feat_src, grad_el, grad_er, slope);
}

void RelationalFusedGATKernel_edge_parallel_separatecoo(
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_rel_ptrs,
    at::Tensor& separate_coo_row_indices, at::Tensor& separate_coo_col_indices,
    at::Tensor& feat_src, at::Tensor& el, at::Tensor& er, at::Tensor& sum,
    at::Tensor& exp, at::Tensor& ret, at::Tensor& gradout,
    at::Tensor& grad_feat_src, at::Tensor& grad_el, at::Tensor& grad_er,
    double slope) {
  at::Tensor dummy_tensor;
  _RelationalFusedGATKernel<int64_t, float, false, false, false, false, false>(
      separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
      separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
      dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor,
      feat_src, el, er, sum, exp, ret, gradout, grad_feat_src, grad_el, grad_er,
      slope);
}

}  // namespace BckProp
}  // namespace RGAT
}  // namespace TorchExport
}  // namespace HET

using namespace HET::TorchExport;
TORCH_LIBRARY_FRAGMENT(torch_hetero_edgesoftmax, m) {
  // RGAT Declaration
  // RGAT Relational SpMM
  m.def("backward_rgat_relational_fused_gat_compact_as_of_node_csr",
        RGAT::BckProp::RelationalFusedGATKernelCompactAsOfNode_integratedcsr);
  m.def("rgat_relational_fused_gat_compact_as_of_node_csr",
        RGAT::FwProp::RelationalFusedGATKernelCompactAsOfNode_integratedcsr);
  m.def("relational_fused_gat_kernel_csr",
        RGAT::FwProp::RelationalFusedGATKernel_integratedcsr);
  m.def("backward_relational_fused_gat_csr",
        RGAT::BckProp::RelationalFusedGATKernel_integratedcsr);
  m.def("relational_fused_gat_kernel_separate_coo",
        RGAT::FwProp::RelationalFusedGATKernel_edge_parallel_separatecoo);
  m.def("backward_relational_fused_gat_separate_coo",
        RGAT::BckProp::RelationalFusedGATKernel_edge_parallel_separatecoo);
  m.def(
      "relational_fused_gat_kernel_compact_as_of_node_separate_coo_dual_unique_"
      "node_list",
      RGAT::FwProp::
          RelationalFusedGATKernelCompactAsOfNode_edge_parallel_separatecoo_dual_unique_node_list);
  m.def(
      "backward_relational_fused_gat_compact_as_of_node_separate_coo_dual_"
      "unique_node_list",
      RGAT::BckProp::
          RelationalFusedGATKernelCompactAsOfNode_edge_parallel_separatecoo_dual_unique_node_list);
  m.def("relational_fused_gat_kernel_compact_as_of_node_separate_coo",
        RGAT::FwProp::
            RelationalFusedGATKernelCompactAsOfNode_edge_parallel_separatecoo);
  m.def("backward_relational_fused_gat_compact_as_of_node_separate_coo",
        RGAT::BckProp::
            RelationalFusedGATKernelCompactAsOfNode_edge_parallel_separatecoo);
}
