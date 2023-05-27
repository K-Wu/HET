#pragma once
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "DGLHackKernel/DGLHackUtils.h"
#include "DGLHackKernel/GAT/FusedGAT.cu.h"
#include "DGLHackKernel/GAT/FusedGATBackward.cu.h"
#include "DGLHackKernel/RGAT/RGATBackwardKernelsSeparateCOO.cu.h"
#include "DGLHackKernel/RGAT/RGATKernelsSeparateCOO.cu.h"
#include "ThreadingGridsBlocksSchedules.h"
#include "kernel_enums.h"

namespace HET {
namespace TorchExport {
namespace RGAT {
namespace FwProp {
template <
    /*int XPU, */ typename Idx, typename DType, CompactAsOfNodeKind CompactKind,
    bool IntegratedFormatRatherThanSeparateFlag, bool CSRRatherThanCOOFlag>
void _RelationalFusedGATKernel(
    at::Tensor &separate_coo_eids, at::Tensor &separate_coo_rel_ptrs,
    at::Tensor &separate_coo_row_indices, at::Tensor &separate_coo_col_indices,
    at::Tensor &incsr_row_ptr, at::Tensor &incsr_col_indices,
    at::Tensor &incsr_eids, at::Tensor &incsr_reltypes,
    at::Tensor &unique_srcs_and_dests_rel_ptrs,
    at::Tensor &unique_srcs_and_dests_rel_ptrs_col,
    at::Tensor &unique_srcs_and_dests_node_indices,
    at::Tensor &unique_srcs_and_dests_node_indices_col, at::Tensor &feat_src,
    at::Tensor &el, at::Tensor &er, at::Tensor &sum, at::Tensor &exp,
    at::Tensor &ret, double slope) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  // NB: in this case gdata.n, calculation is removed since el is now per edge
  // rather than per node
  constexpr bool CompactAsOfNodeFlag = IsCompact(CompactKind);
  constexpr bool DualUniqueNodeList = IsCompactWithDualList(CompactKind);
  GatFusedData<Idx, DType> gdata{
      .feat_src_xlen = SeastarComputeXLength<>(feat_src),
      .num_heads = SeastarComputeXLength<>(el),
      .eids = nullptr,  // to be assigned later in if branches
      .leaky_relu_slope = slope,
      .feat_src = feat_src.data_ptr<DType>(),
      .el = el.data_ptr<DType>(),
      .er = er.data_ptr<DType>(),
      .sum = sum.data_ptr<DType>(),
      .exp = exp.data_ptr<DType>(),
      .ret = ret.data_ptr<DType>()};

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
    int64_t incsr_num_rows = incsr_row_ptr.numel() - 1;
    auto [nblks, nthrs] = get_type1_schedule(gdata.num_heads, incsr_num_rows);

    HET_gatExpLeakyReluSumKernel<Idx, DType, CompactKind, true>
        <<<nblks, nthrs, 0, stream>>>(
            gdata, incsr_row_ptr.data_ptr<Idx>(),
            incsr_col_indices.data_ptr<Idx>(), incsr_reltypes.data_ptr<Idx>(),
            incsr_num_rows,
            CompactAsOfNodeFlag ? unique_srcs_and_dests_rel_ptrs.data_ptr<Idx>()
                                : nullptr,
            CompactAsOfNodeFlag
                ? unique_srcs_and_dests_node_indices.data_ptr<Idx>()
                : nullptr);

    // NB: updated to Type 2 Schedule:
    // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
    // head -> threadIdx.y
    // node -> blockIdx.y
    // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x
    auto [nblks2, nthrs2] = get_type2_schedule(
        gdata.num_heads, gdata.feat_src_xlen, incsr_num_rows);

    HET_gatSumProdZipDivKernel<Idx, DType, CompactKind, true>
        <<<nblks2, nthrs2, 0, stream>>>(
            gdata, incsr_row_ptr.data_ptr<Idx>(),
            incsr_col_indices.data_ptr<Idx>(), incsr_reltypes.data_ptr<Idx>(),
            incsr_num_rows,
            CompactAsOfNodeFlag ? unique_srcs_and_dests_rel_ptrs.data_ptr<Idx>()
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
    auto [nblks, nthrs] = get_type1_schedule(gdata.num_heads, num_edges);

    HET_gatExpLeakyReluSumKernel_relational_separate_coo<
        Idx, DType, CompactKind><<<nblks, nthrs, 0, stream>>>(
        gdata, separate_coo_rel_ptrs.data_ptr<Idx>(),
        separate_coo_row_indices.data_ptr<Idx>(),
        separate_coo_col_indices.data_ptr<Idx>(), num_edges,
        CompactAsOfNodeFlag ? unique_srcs_and_dests_rel_ptrs.data_ptr<Idx>()
                            : nullptr,
        DualUniqueNodeList ? unique_srcs_and_dests_rel_ptrs_col.data_ptr<Idx>()
                           : nullptr,
        CompactAsOfNodeFlag ? unique_srcs_and_dests_node_indices.data_ptr<Idx>()
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
    auto [nblks2, nthrs2] =
        get_type2_schedule(gdata.num_heads, gdata.feat_src_xlen, num_edges);
    HET_gatSumProdZipDivKernel_relational_separate_coo<Idx, DType, CompactKind>
        <<<nblks2, nthrs2, 0, stream>>>(
            gdata, separate_coo_rel_ptrs.data_ptr<Idx>(),
            separate_coo_row_indices.data_ptr<Idx>(),
            separate_coo_col_indices.data_ptr<Idx>(), num_edges,
            CompactAsOfNodeFlag ? unique_srcs_and_dests_rel_ptrs.data_ptr<Idx>()
                                : nullptr,
            DualUniqueNodeList
                ? unique_srcs_and_dests_rel_ptrs_col.data_ptr<Idx>()
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
    at::Tensor &separate_coo_eids, at::Tensor &separate_coo_rel_ptrs,
    at::Tensor &separate_coo_row_indices, at::Tensor &separate_coo_col_indices,
    at::Tensor &unique_srcs_and_dests_rel_ptrs,
    at::Tensor &unique_srcs_and_dests_rel_ptrs_col,
    at::Tensor &unique_srcs_and_dests_node_indices_row,
    at::Tensor &unique_srcs_and_dests_node_indices_col, at::Tensor &feat_src,
    at::Tensor &el, at::Tensor &er, at::Tensor &sum, at::Tensor &exp,
    at::Tensor &ret, double slope) {
  at::Tensor dummy_tensor;
  _RelationalFusedGATKernel<
      int64_t, float, CompactAsOfNodeKind::EnabledWithDualList, false, false>(
      separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
      separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
      dummy_tensor, unique_srcs_and_dests_rel_ptrs,
      unique_srcs_and_dests_rel_ptrs_col,
      unique_srcs_and_dests_node_indices_row,
      unique_srcs_and_dests_node_indices_col, feat_src, el, er, sum, exp, ret,
      slope);
}

void RelationalFusedGATKernel_edge_parallel_separatecoo(
    at::Tensor &separate_coo_eids, at::Tensor &separate_coo_rel_ptrs,
    at::Tensor &separate_coo_row_indices, at::Tensor &separate_coo_col_indices,
    bool CompactAsOfNodeFlag,
    torch::Dict<std::string, at::Tensor> unique_srcs_and_dests,
    // at::Tensor &unique_srcs_and_dests_rel_ptr,
    // at::Tensor &unique_srcs_and_dests_node_indices,
    at::Tensor &feat_src, at::Tensor &el, at::Tensor &er, at::Tensor &sum,
    at::Tensor &exp, at::Tensor &ret, double slope) {
  at::Tensor dummy_tensor;
  if (CompactAsOfNodeFlag) {
    // CompactAsOfNode
    at::Tensor unique_srcs_and_dests_rel_ptrs =
        unique_srcs_and_dests.at("rel_ptrs");
    at::Tensor unique_srcs_and_dests_node_indices =
        unique_srcs_and_dests.at("node_indices");
    _RelationalFusedGATKernel<int64_t, float, CompactAsOfNodeKind::Enabled,
                              false, false>(
        separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
        separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
        dummy_tensor, unique_srcs_and_dests_rel_ptrs, dummy_tensor,
        unique_srcs_and_dests_node_indices, dummy_tensor, feat_src, el, er, sum,
        exp, ret, slope);
  } else {
    _RelationalFusedGATKernel<int64_t, float, CompactAsOfNodeKind::Disabled,
                              false, false>(
        separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
        separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
        dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor,
        feat_src, el, er, sum, exp, ret, slope);
  }
}

void RelationalFusedGATKernel_integratedcsr(
    at::Tensor &incsr_row_ptr, at::Tensor &incsr_col_indices,
    at::Tensor &incsr_eids, at::Tensor &incsr_reltypes,
    at::Tensor &unique_srcs_and_dests_rel_ptrs,
    at::Tensor &unique_srcs_and_dests_node_indices, at::Tensor &feat_src,
    at::Tensor &el, at::Tensor &er, at::Tensor &sum, at::Tensor &exp,
    at::Tensor &ret, double slope, bool CompactAsOfNodeFlag) {
  at::Tensor dummy_tensor;
  if (CompactAsOfNodeFlag) {
    _RelationalFusedGATKernel<int64_t, float, CompactAsOfNodeKind::Enabled,
                              true, true>(
        dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, incsr_row_ptr,
        incsr_col_indices, incsr_eids, incsr_reltypes,
        unique_srcs_and_dests_rel_ptrs, dummy_tensor,
        unique_srcs_and_dests_node_indices, dummy_tensor, feat_src, el, er, sum,
        exp, ret, slope);
  } else {
    _RelationalFusedGATKernel<int64_t, float, CompactAsOfNodeKind::Disabled,
                              true, true>(
        dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, incsr_row_ptr,
        incsr_col_indices, incsr_eids, incsr_reltypes,
        unique_srcs_and_dests_rel_ptrs, dummy_tensor,
        unique_srcs_and_dests_node_indices, dummy_tensor, feat_src, el, er, sum,
        exp, ret, slope);
  }
}
}  // namespace FwProp
namespace BckProp {
template </*int XPU, */ typename Idx, typename DType, bool FLAG_KERNEL_FUSED,
          CompactAsOfNodeKind CompactKind,
          bool IntegratedFormatRatherThanSeparateFlag,
          bool CSRRatherThanCOOFlag>
void _RelationalFusedGATKernel(
    at::Tensor &separate_coo_eids, at::Tensor &separate_coo_rel_ptrs,
    at::Tensor &separate_coo_row_indices, at::Tensor &separate_coo_col_indices,
    at::Tensor &outcsr_row_ptr, at::Tensor &outcsr_col_indices,
    at::Tensor &outcsr_eids, at::Tensor &outcsr_reltypes,
    at::Tensor &unique_srcs_and_dests_rel_ptrs,
    at::Tensor &unique_srcs_and_dests_rel_ptrs_col,
    at::Tensor &unique_srcs_and_dests_node_indices,
    at::Tensor &unique_srcs_and_dests_node_indices_col, at::Tensor &feat_src,
    at::Tensor &el, at::Tensor &er, at::Tensor &sum, at::Tensor &exp,
    at::Tensor &ret, at::Tensor &gradout, at::Tensor &grad_feat_src,
    at::Tensor &grad_el, at::Tensor &grad_er, double slope) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  constexpr bool CompactAsOfNodeFlag = IsCompact(CompactKind);
  constexpr bool DualUniqueNodeList = IsCompactWithDualList(CompactKind);

  BackwardGatFusedData<Idx, DType> gdata{
      .feat_src_xlen = SeastarComputeXLength<>(feat_src),
      .num_heads = SeastarComputeXLength<>(el),
      .eids = nullptr,  // to be assigned later in if branches
      .leaky_relu_slope = slope,
      .feat_src = feat_src.data_ptr<DType>(),
      .el = el.data_ptr<DType>(),
      .er = er.data_ptr<DType>(),
      .sum = sum.data_ptr<DType>(),
      .exp = exp.data_ptr<DType>(),
      .ret = ret.data_ptr<DType>(),
      .grad_out = gradout.data_ptr<DType>(),
      .grad_feat_src = grad_feat_src.data_ptr<DType>(),
      .grad_el = grad_el.data_ptr<DType>(),
      .grad_er = grad_er.data_ptr<DType>()};

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
    int64_t outcsr_num_rows = outcsr_row_ptr.numel() - 1;
    auto [nblks, nthrs] = get_type2_schedule(
        gdata.num_heads, gdata.feat_src_xlen, outcsr_num_rows);
    if constexpr (!FLAG_KERNEL_FUSED) {
      HET_fusedGatBackwardGradFeatSrc<Idx, DType, CompactKind, true>
          <<<nblks, nthrs, 0, stream>>>(
              gdata, outcsr_row_ptr.data_ptr<Idx>(),
              outcsr_col_indices.data_ptr<Idx>(),
              outcsr_reltypes.data_ptr<Idx>(), outcsr_num_rows,
              CompactAsOfNodeFlag
                  ? unique_srcs_and_dests_rel_ptrs.data_ptr<Idx>()
                  : nullptr,
              CompactAsOfNodeFlag
                  ? unique_srcs_and_dests_node_indices.data_ptr<Idx>()
                  : nullptr);
      HET_fusedGatBackwardGradElEr<Idx, DType, CompactKind, true>
          <<<nblks, nthrs, 0, stream>>>(
              gdata, outcsr_row_ptr.data_ptr<Idx>(),
              outcsr_col_indices.data_ptr<Idx>(),
              outcsr_reltypes.data_ptr<Idx>(), outcsr_num_rows,
              CompactAsOfNodeFlag
                  ? unique_srcs_and_dests_rel_ptrs.data_ptr<Idx>()
                  : nullptr,
              CompactAsOfNodeFlag
                  ? unique_srcs_and_dests_node_indices.data_ptr<Idx>()
                  : nullptr);
    } else {
      HET_fusedGatBackwardGradElErFeatSrcFused<Idx, DType, CompactKind, true>
          <<<nblks, nthrs, 0, stream>>>(
              gdata, outcsr_row_ptr.data_ptr<Idx>(),
              outcsr_col_indices.data_ptr<Idx>(),
              outcsr_reltypes.data_ptr<Idx>(), outcsr_num_rows,
              CompactAsOfNodeFlag
                  ? unique_srcs_and_dests_rel_ptrs.data_ptr<Idx>()
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
    auto [nblks, nthrs] =
        get_type2_schedule(gdata.num_heads, gdata.feat_src_xlen, num_edges);
    if constexpr (!FLAG_KERNEL_FUSED) {
      HET_fusedGatBackwardGradFeatSrc_relational_separate_coo<Idx, DType,
                                                              CompactKind>
          <<<nblks, nthrs, 0, stream>>>(
              gdata, separate_coo_rel_ptrs.data_ptr<Idx>(),
              separate_coo_row_indices.data_ptr<Idx>(),
              separate_coo_col_indices.data_ptr<Idx>(), num_edges,
              CompactAsOfNodeFlag
                  ? unique_srcs_and_dests_rel_ptrs.data_ptr<Idx>()
                  : nullptr,
              DualUniqueNodeList
                  ? unique_srcs_and_dests_rel_ptrs_col.data_ptr<Idx>()
                  : nullptr,
              CompactAsOfNodeFlag
                  ? unique_srcs_and_dests_node_indices.data_ptr<Idx>()
                  : nullptr,
              DualUniqueNodeList
                  ? unique_srcs_and_dests_node_indices_col.data_ptr<Idx>()
                  : nullptr,
              num_relations);
      HET_fusedGatBackwardGradElEr_relational_separate_coo<Idx, DType,
                                                           CompactKind>
          <<<nblks, nthrs, 0, stream>>>(
              gdata, separate_coo_rel_ptrs.data_ptr<Idx>(),
              separate_coo_row_indices.data_ptr<Idx>(),
              separate_coo_col_indices.data_ptr<Idx>(), num_edges,
              CompactAsOfNodeFlag
                  ? unique_srcs_and_dests_rel_ptrs.data_ptr<Idx>()
                  : nullptr,
              DualUniqueNodeList
                  ? unique_srcs_and_dests_rel_ptrs_col.data_ptr<Idx>()
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
          Idx, DType, CompactKind><<<nblks, nthrs, 0, stream>>>(
          gdata, separate_coo_rel_ptrs.data_ptr<Idx>(),
          separate_coo_row_indices.data_ptr<Idx>(),
          separate_coo_col_indices.data_ptr<Idx>(), num_edges,
          CompactAsOfNodeFlag ? unique_srcs_and_dests_rel_ptrs.data_ptr<Idx>()
                              : nullptr,
          DualUniqueNodeList
              ? unique_srcs_and_dests_rel_ptrs_col.data_ptr<Idx>()
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
    at::Tensor &outcsr_row_ptr, at::Tensor &outcsr_col_indices,
    at::Tensor &outcsr_eids, at::Tensor &outcsr_reltypes,
    at::Tensor &unique_srcs_and_dests_rel_ptrs,
    at::Tensor &unique_srcs_and_dests_node_indices, at::Tensor &feat_src,
    at::Tensor &el, at::Tensor &er, at::Tensor &sum, at::Tensor &exp,
    at::Tensor &ret, at::Tensor &gradout, at::Tensor &grad_feat_src,
    at::Tensor &grad_el, at::Tensor &grad_er, double slope,
    bool CompactAsOfNodeFlag) {
  at::Tensor dummy_tensor;
  if (CompactAsOfNodeFlag) {
    _RelationalFusedGATKernel<int64_t, float, true,
                              CompactAsOfNodeKind::Disabled, true, true>(
        dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, outcsr_row_ptr,
        outcsr_col_indices, outcsr_eids, outcsr_reltypes,
        unique_srcs_and_dests_rel_ptrs, dummy_tensor,
        unique_srcs_and_dests_node_indices, dummy_tensor, feat_src, el, er, sum,
        exp, ret, gradout, grad_feat_src, grad_el, grad_er, slope);
  } else {
    _RelationalFusedGATKernel<int64_t, float, true,
                              CompactAsOfNodeKind::Enabled, true, true>(
        dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, outcsr_row_ptr,
        outcsr_col_indices, outcsr_eids, outcsr_reltypes,
        unique_srcs_and_dests_rel_ptrs, dummy_tensor,
        unique_srcs_and_dests_node_indices, dummy_tensor, feat_src, el, er, sum,
        exp, ret, gradout, grad_feat_src, grad_el, grad_er, slope);
  }
}

void RelationalFusedGATKernel_edge_parallel_separatecoo(
    at::Tensor &separate_coo_eids, at::Tensor &separate_coo_rel_ptrs,
    at::Tensor &separate_coo_row_indices, at::Tensor &separate_coo_col_indices,
    int64_t IntKind, torch::Dict<std::string, at::Tensor> unique_srcs_and_dests,
    // at::Tensor &unique_srcs_and_dests_rel_ptr,
    // at::Tensor &unique_srcs_and_dests_node_indices,
    at::Tensor &feat_src, at::Tensor &el, at::Tensor &er, at::Tensor &sum,
    at::Tensor &exp, at::Tensor &ret, at::Tensor &gradout,
    at::Tensor &grad_feat_src, at::Tensor &grad_el, at::Tensor &grad_er,
    double slope) {
  at::Tensor dummy_tensor;
  auto Kind = static_cast<CompactAsOfNodeKind>(IntKind);
  if (Kind == CompactAsOfNodeKind::EnabledWithDualList) {
    at::Tensor unique_srcs_and_dests_rel_ptrs =
        unique_srcs_and_dests.at("unique_srcs_and_dests_rel_ptrs");
    at::Tensor unique_srcs_and_dests_rel_ptrs_col =
        unique_srcs_and_dests.at("unique_srcs_and_dests_rel_col");
    at::Tensor unique_srcs_and_dests_node_indices =
        unique_srcs_and_dests.at("unique_srcs_and_dests_node_indices_row");
    at::Tensor unique_srcs_and_dests_node_indices_col =
        unique_srcs_and_dests.at("unique_srcs_and_dests_node_indices_col");

    _RelationalFusedGATKernel<int64_t, float, true,
                              CompactAsOfNodeKind::EnabledWithDualList, false,
                              false>(
        separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
        separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
        dummy_tensor, unique_srcs_and_dests_rel_ptrs,
        unique_srcs_and_dests_rel_ptrs_col, unique_srcs_and_dests_node_indices,
        unique_srcs_and_dests_node_indices_col, feat_src, el, er, sum, exp, ret,
        gradout, grad_feat_src, grad_el, grad_er, slope);
  } else if (Kind == CompactAsOfNodeKind::Enabled) {
    at::Tensor unique_srcs_and_dests_rel_ptrs =
        unique_srcs_and_dests.at("unique_srcs_and_dests_rel_ptrs");
    at::Tensor unique_srcs_and_dests_node_indices =
        unique_srcs_and_dests.at("unique_srcs_and_dests_node_indices");
    _RelationalFusedGATKernel<int64_t, float, true,
                              CompactAsOfNodeKind::Enabled, false, false>(
        separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
        separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
        dummy_tensor, unique_srcs_and_dests_rel_ptrs, dummy_tensor,
        unique_srcs_and_dests_node_indices, dummy_tensor, feat_src, el, er, sum,
        exp, ret, gradout, grad_feat_src, grad_el, grad_er, slope);
  } else if (Kind == CompactAsOfNodeKind::Disabled) {
    _RelationalFusedGATKernel<int64_t, float, false,
                              CompactAsOfNodeKind::Disabled, false, false>(
        separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
        separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
        dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor,
        feat_src, el, er, sum, exp, ret, gradout, grad_feat_src, grad_el,
        grad_er, slope);
  } else {
    printf("%ld\n", IntKind);
    throw std::runtime_error("Invalid CompactAsOfNodeKind");
  }
}

}  // namespace BckProp
}  // namespace RGAT
}  // namespace TorchExport
}  // namespace HET

using namespace HET::TorchExport;
TORCH_LIBRARY_FRAGMENT(torch_hetero_edgesoftmax, m) {
  // RGAT Declaration
  // RGAT Relational SpMM
  m.def("relational_fused_gat_kernel_csr",
        RGAT::FwProp::RelationalFusedGATKernel_integratedcsr);
  m.def("backward_relational_fused_gat_csr",
        RGAT::BckProp::RelationalFusedGATKernel_integratedcsr);
  m.def("relational_fused_gat_separate_coo",
        RGAT::FwProp::RelationalFusedGATKernel_edge_parallel_separatecoo);
  m.def("backward_relational_fused_gat_separate_coo",
        RGAT::BckProp::RelationalFusedGATKernel_edge_parallel_separatecoo);
  // clang-format off
  m.def(
      "relational_fused_gat_kernel_compact_as_of_node_separate_coo_dual_unique_node_list",
      RGAT::FwProp::
          RelationalFusedGATKernelCompactAsOfNode_edge_parallel_separatecoo_dual_unique_node_list);
  // clang-format on
}
