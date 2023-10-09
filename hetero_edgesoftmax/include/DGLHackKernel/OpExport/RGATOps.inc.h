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
#include "ThreadingGridsBlocksSchedules.h"
#include "kernel_enums.h"

namespace HET {
namespace TorchExport {
namespace RGAT {
namespace FwProp {
template <
    /*int XPU, */ typename Idx, typename DType, CompactAsOfNodeKind CompactKind,
    bool IntegratedFormatRatherThanSeparateFlag, bool CSRRatherThanCOOFlag>
void _RelationalFusedGAT(
    at::Tensor &separate_coo_eids, at::Tensor &separate_coo_rel_ptrs,
    at::Tensor &separate_coo_row_indices, at::Tensor &separate_coo_col_indices,
    at::Tensor &incsr_row_ptr, at::Tensor &incsr_col_indices,
    at::Tensor &incsr_eids, at::Tensor &incsr_reltypes,
    at::Tensor &unique_srcs_and_dests_rel_ptrs,
    at::Tensor &unique_srcs_and_dests_rel_ptrs_col,
    at::Tensor &unique_srcs_and_dests_node_indices,
    at::Tensor &unique_srcs_and_dests_node_indices_col,
    at::Tensor &edata_idx_to_inverse_idx,
    at::Tensor &edata_idx_to_inverse_idx_col, at::Tensor &feat_src,
    at::Tensor &el, at::Tensor &er, at::Tensor &sum, at::Tensor &exp,
    at::Tensor &ret, double slope) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  // NB: in this case gdata.n, calculation is removed since el is now per edge
  // rather than per node
  GatFusedData<Idx, DType> gdata{
      .feat_src_xlen = SeastarComputeXLength<>(feat_src),
      .num_heads = SeastarComputeXLength<>(el),
      .eids = nullptr,  // to be assigned later in if branches
      .leaky_relu_slope = static_cast<float>(slope),
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
                                          IsCompactWithDualList(CompactKind),
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

    ETypeMapperData<Idx, CompactKind> etype_mapper_data;
    if constexpr (IsCompact(CompactKind)) {
      if constexpr (CompactKind == CompactAsOfNodeKind::Enabled) {
        etype_mapper_data.unique_srcs_and_dests_rel_ptrs =
            unique_srcs_and_dests_rel_ptrs.data_ptr<Idx>();
        etype_mapper_data.unique_srcs_and_dests_node_indices =
            unique_srcs_and_dests_node_indices.data_ptr<Idx>();
      } else {
        assert(0 && "unrecognized compact kind");
      }
    }
    ETypeData<Idx, false> etype_data{.etypes = incsr_reltypes.data_ptr<Idx>()};

    HET_gatExpLeakyReluSumKernel<Idx, DType, CompactKind, true>
        <<<nblks, nthrs, 0, stream>>>(gdata, incsr_row_ptr.data_ptr<Idx>(),
                                      incsr_col_indices.data_ptr<Idx>(),
                                      etype_data, incsr_num_rows,
                                      etype_mapper_data);

    // NB: updated to Type 2 Schedule:
    // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
    // head -> threadIdx.y
    // node -> blockIdx.y
    // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x
    auto [nblks2, nthrs2] = get_type2_schedule(
        gdata.num_heads, gdata.feat_src_xlen, incsr_num_rows);

    HET_gatSumProdZipDivKernel<Idx, DType, CompactKind, true>
        <<<nblks2, nthrs2, 0, stream>>>(gdata, incsr_row_ptr.data_ptr<Idx>(),
                                        incsr_col_indices.data_ptr<Idx>(),
                                        etype_data, incsr_num_rows,
                                        etype_mapper_data);
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

    ETypeMapperData<Idx, CompactKind> etype_mapper_data;
    if constexpr (IsCompact(CompactKind)) {
      if constexpr (IsBinarySearch(CompactKind)) {
        etype_mapper_data.unique_srcs_and_dests_rel_ptrs =
            unique_srcs_and_dests_rel_ptrs.data_ptr<Idx>();
        etype_mapper_data.unique_srcs_and_dests_node_indices =
            unique_srcs_and_dests_node_indices.data_ptr<Idx>();
      } else {
        etype_mapper_data.edata_idx_to_inverse_idx =
            edata_idx_to_inverse_idx.data_ptr<Idx>();
      }
    }
    ETypeMapperData<Idx, CompactKind> etype_mapper_data_col;
    if constexpr (IsCompactWithDualList(CompactKind)) {
      if constexpr (IsBinarySearch(CompactKind)) {
        etype_mapper_data_col.unique_srcs_and_dests_rel_ptrs =
            unique_srcs_and_dests_rel_ptrs_col.data_ptr<Idx>();
        etype_mapper_data_col.unique_srcs_and_dests_node_indices =
            unique_srcs_and_dests_node_indices_col.data_ptr<Idx>();
      } else {
        etype_mapper_data_col.edata_idx_to_inverse_idx =
            edata_idx_to_inverse_idx_col.data_ptr<Idx>();
      }
    }
    ETypeData<Idx, true> etype_data{
        .etypes = separate_coo_rel_ptrs.data_ptr<Idx>(),
        .num_relations = num_relations};

    HET_gatExpLeakyReluSumKernel_relational_separate_coo<Idx, DType,
                                                         CompactKind>
        <<<nblks, nthrs, 0, stream>>>(
            gdata, etype_data, separate_coo_row_indices.data_ptr<Idx>(),
            separate_coo_col_indices.data_ptr<Idx>(), num_edges,
            etype_mapper_data, etype_mapper_data_col);

    // NB: updated to Type 2 Schedule:
    // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
    // head -> threadIdx.y
    // node -> blockIdx.y
    // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x
    auto [nblks2, nthrs2] =
        get_type2_schedule(gdata.num_heads, gdata.feat_src_xlen, num_edges);
    HET_gatSumProdZipDivKernel_relational_separate_coo<Idx, DType, CompactKind>
        <<<nblks2, nthrs2, 0, stream>>>(
            gdata, etype_data, separate_coo_row_indices.data_ptr<Idx>(),
            separate_coo_col_indices.data_ptr<Idx>(), num_edges,
            etype_mapper_data, etype_mapper_data_col);

  } else {
    assert(0 && "Not implemented");
  }
}
namespace SeparateCOO {
namespace EdgeParallel {
void RelationalFusedGAT(at::Tensor &separate_coo_eids,
                        at::Tensor &separate_coo_rel_ptrs,
                        at::Tensor &separate_coo_row_indices,
                        at::Tensor &separate_coo_col_indices, int64_t IntKind,
                        torch::Dict<std::string, at::Tensor> args_tensor_dict,
                        at::Tensor &feat_src, at::Tensor &el, at::Tensor &er,
                        at::Tensor &sum, at::Tensor &exp, at::Tensor &ret,
                        double slope) {
  at::Tensor dummy_tensor;
  auto CompactKind = static_cast<CompactAsOfNodeKind>(IntKind);
  if (CompactKind ==
      CompactAsOfNodeKind::EnabledWithDualListWithDirectIndexing) {
    at::Tensor edata_idx_to_inverse_idx_row =
        args_tensor_dict.at("edata_idx_to_inverse_idx_row");
    at::Tensor edata_idx_to_inverse_idx_col =
        args_tensor_dict.at("edata_idx_to_inverse_idx_col");
    _RelationalFusedGAT<
        int64_t, float,
        CompactAsOfNodeKind::EnabledWithDualListWithDirectIndexing, false,
        false>(separate_coo_eids, separate_coo_rel_ptrs,
               separate_coo_row_indices, separate_coo_col_indices, dummy_tensor,
               dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor,
               dummy_tensor, dummy_tensor, dummy_tensor,
               edata_idx_to_inverse_idx_row, edata_idx_to_inverse_idx_col,
               feat_src, el, er, sum, exp, ret, slope);
  } else if (CompactKind == CompactAsOfNodeKind::EnabledWithDirectIndexing) {
    at::Tensor edata_idx_to_inverse_idx =
        args_tensor_dict.at("edata_idx_to_inverse_idx");
    _RelationalFusedGAT<int64_t, float,
                        CompactAsOfNodeKind::EnabledWithDirectIndexing, false,
                        false>(
        separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
        separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
        dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor,
        edata_idx_to_inverse_idx, dummy_tensor, feat_src, el, er, sum, exp, ret,
        slope);
  } else if (CompactKind == CompactAsOfNodeKind::EnabledWithDualList) {
    at::Tensor unique_srcs_and_dests_rel_ptrs =
        args_tensor_dict.at("unique_srcs_and_dests_rel_ptrs");
    at::Tensor unique_srcs_and_dests_rel_ptrs_col =
        args_tensor_dict.at("unique_srcs_and_dests_rel_ptrs_col");
    at::Tensor unique_srcs_and_dests_node_indices_row =
        args_tensor_dict.at("unique_srcs_and_dests_node_indices_row");
    at::Tensor unique_srcs_and_dests_node_indices_col =
        args_tensor_dict.at("unique_srcs_and_dests_node_indices_col");
    _RelationalFusedGAT<int64_t, float,
                        CompactAsOfNodeKind::EnabledWithDualList, false, false>(
        separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
        separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
        dummy_tensor, unique_srcs_and_dests_rel_ptrs,
        unique_srcs_and_dests_rel_ptrs_col,
        unique_srcs_and_dests_node_indices_row,
        unique_srcs_and_dests_node_indices_col, dummy_tensor, dummy_tensor,
        feat_src, el, er, sum, exp, ret, slope);
  } else if (CompactKind == CompactAsOfNodeKind::Enabled) {
    // CompactAsOfNode
    at::Tensor unique_srcs_and_dests_rel_ptrs =
        args_tensor_dict.at("unique_srcs_and_dests_rel_ptrs");
    at::Tensor unique_srcs_and_dests_node_indices =
        args_tensor_dict.at("unique_srcs_and_dests_node_indices");
    _RelationalFusedGAT<int64_t, float, CompactAsOfNodeKind::Enabled, false,
                        false>(
        separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
        separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
        dummy_tensor, unique_srcs_and_dests_rel_ptrs, dummy_tensor,
        unique_srcs_and_dests_node_indices, dummy_tensor, dummy_tensor,
        dummy_tensor, feat_src, el, er, sum, exp, ret, slope);
  } else {
    _RelationalFusedGAT<int64_t, float, CompactAsOfNodeKind::Disabled, false,
                        false>(
        separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
        separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
        dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor,
        dummy_tensor, dummy_tensor, feat_src, el, er, sum, exp, ret, slope);
  }
}

}  // namespace EdgeParallel
}  // namespace SeparateCOO
namespace IntegratedCSR {
// TODO: pass in direct indexing
void RelationalFusedGAT(at::Tensor &incsr_row_ptr,
                        at::Tensor &incsr_col_indices, at::Tensor &incsr_eids,
                        at::Tensor &incsr_reltypes,
                        at::Tensor &unique_srcs_and_dests_rel_ptrs,
                        at::Tensor &unique_srcs_and_dests_node_indices,
                        at::Tensor &feat_src, at::Tensor &el, at::Tensor &er,
                        at::Tensor &sum, at::Tensor &exp, at::Tensor &ret,
                        double slope, bool CompactAsOfNodeFlag) {
  at::Tensor dummy_tensor;
  if (CompactAsOfNodeFlag) {
    _RelationalFusedGAT<int64_t, float, CompactAsOfNodeKind::Enabled, true,
                        true>(
        dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, incsr_row_ptr,
        incsr_col_indices, incsr_eids, incsr_reltypes,
        unique_srcs_and_dests_rel_ptrs, dummy_tensor,
        unique_srcs_and_dests_node_indices, dummy_tensor, dummy_tensor,
        dummy_tensor, feat_src, el, er, sum, exp, ret, slope);
  } else {
    _RelationalFusedGAT<int64_t, float, CompactAsOfNodeKind::Disabled, true,
                        true>(
        dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, incsr_row_ptr,
        incsr_col_indices, incsr_eids, incsr_reltypes,
        unique_srcs_and_dests_rel_ptrs, dummy_tensor,
        unique_srcs_and_dests_node_indices, dummy_tensor, dummy_tensor,
        dummy_tensor, feat_src, el, er, sum, exp, ret, slope);
  }
}
}  // namespace IntegratedCSR
}  // namespace FwProp
namespace BckProp {
template </*int XPU, */ typename Idx, typename DType, bool FLAG_KERNEL_FUSED,
          CompactAsOfNodeKind CompactKind,
          bool IntegratedFormatRatherThanSeparateFlag, //want this to be false
          bool CSRRatherThanCOOFlag> //want this to be true
void _RelationalFusedGAT(
    at::Tensor &separate_coo_eids, at::Tensor &separate_coo_rel_ptrs,
    at::Tensor &separate_coo_row_indices, at::Tensor &separate_coo_col_indices,
    at::Tensor &outcsr_row_ptr, at::Tensor &outcsr_col_indices,
    at::Tensor &outcsr_eids, at::Tensor &outcsr_reltypes,
    at::Tensor &unique_srcs_and_dests_rel_ptrs,
    at::Tensor &unique_srcs_and_dests_rel_ptrs_col,
    at::Tensor &unique_srcs_and_dests_node_indices,
    at::Tensor &unique_srcs_and_dests_node_indices_col,
    at::Tensor &edata_idx_to_inverse_idx,
    at::Tensor &edata_idx_to_inverse_idx_col, at::Tensor &feat_src,
    at::Tensor &el, at::Tensor &er, at::Tensor &sum, at::Tensor &exp,
    at::Tensor &ret, at::Tensor &gradout, at::Tensor &grad_feat_src,
    at::Tensor &grad_el, at::Tensor &grad_er, double slope) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  BackwardGatFusedData<Idx, DType> gdata{
      .feat_src_xlen = SeastarComputeXLength<>(feat_src),
      .num_heads = SeastarComputeXLength<>(el),
      .eids = nullptr,  // to be assigned later in if branches
      .leaky_relu_slope = static_cast<float>(slope),
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
        IsCompactWithDualList(CompactKind) &&
            IntegratedFormatRatherThanSeparateFlag && CSRRatherThanCOOFlag,
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
    ETypeMapperData<Idx, CompactKind> etype_mapper_data;
    if constexpr (IsCompact(CompactKind)) {
      etype_mapper_data.unique_srcs_and_dests_rel_ptrs =
          unique_srcs_and_dests_rel_ptrs.data_ptr<Idx>();
      etype_mapper_data.unique_srcs_and_dests_node_indices =
          unique_srcs_and_dests_node_indices.data_ptr<Idx>();
    }
    ETypeData<Idx, false> etype_data{
        .etypes = outcsr_reltypes.data_ptr<Idx>(),
    };
    if constexpr (!FLAG_KERNEL_FUSED) {
      HET_fusedGatBackwardGradFeatSrc<Idx, DType, CompactKind, true>
          <<<nblks, nthrs, 0, stream>>>(gdata, outcsr_row_ptr.data_ptr<Idx>(),
                                        outcsr_col_indices.data_ptr<Idx>(),
                                        etype_data, outcsr_num_rows,
                                        etype_mapper_data);
      HET_fusedGatBackwardGradElEr<Idx, DType, CompactKind, true>
          <<<nblks, nthrs, 0, stream>>>(gdata, outcsr_row_ptr.data_ptr<Idx>(),
                                        outcsr_col_indices.data_ptr<Idx>(),
                                        etype_data, outcsr_num_rows,
                                        etype_mapper_data);
    } else {
      HET_fusedGatBackwardGradElErFeatSrcFused<Idx, DType, CompactKind, true>
          <<<nblks, nthrs, 0, stream>>>(gdata, outcsr_row_ptr.data_ptr<Idx>(),
                                        outcsr_col_indices.data_ptr<Idx>(),
                                        etype_data, outcsr_num_rows,
                                        etype_mapper_data);
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
        get_type2_schedule(gdata.num_heads, gdata.feat_src_xlen, num_edges); // last arg used to be num_edges
    ETypeMapperData<Idx, CompactKind> etype_mapper_data;
    if constexpr (IsCompact(CompactKind)) {
      if constexpr (IsBinarySearch(CompactKind)) {
        etype_mapper_data.unique_srcs_and_dests_rel_ptrs =
            unique_srcs_and_dests_rel_ptrs.data_ptr<Idx>();
        etype_mapper_data.unique_srcs_and_dests_node_indices =
            unique_srcs_and_dests_node_indices.data_ptr<Idx>();
      } else {
        etype_mapper_data.edata_idx_to_inverse_idx =
            edata_idx_to_inverse_idx.data_ptr<Idx>();
      }
    }
    ETypeMapperData<Idx, CompactKind> etype_mapper_data_col;
    if constexpr (IsCompactWithDualList(CompactKind)) {
      if constexpr (IsBinarySearch(CompactKind)) {
        etype_mapper_data_col.unique_srcs_and_dests_rel_ptrs =
            unique_srcs_and_dests_rel_ptrs_col.data_ptr<Idx>();
        etype_mapper_data_col.unique_srcs_and_dests_node_indices =
            unique_srcs_and_dests_node_indices_col.data_ptr<Idx>();
      } else {
        etype_mapper_data_col.edata_idx_to_inverse_idx =
            edata_idx_to_inverse_idx_col.data_ptr<Idx>();
      }
    }
    ETypeData<Idx, true> etype_data{
        .etypes = separate_coo_rel_ptrs.data_ptr<Idx>(),
        .num_relations = num_relations};
    if constexpr (!FLAG_KERNEL_FUSED) {
      HET_fusedGatBackwardGradFeatSrc_relational_separate_coo<Idx, DType,
                                                              CompactKind>
          <<<nblks, nthrs, 0, stream>>>(
              gdata, etype_data, separate_coo_row_indices.data_ptr<Idx>(),
              separate_coo_col_indices.data_ptr<Idx>(), num_edges,
              etype_mapper_data, etype_mapper_data_col);
      HET_fusedGatBackwardGradElEr_relational_separate_coo<Idx, DType,
                                                           CompactKind>
          <<<nblks, nthrs, 0, stream>>>(
              gdata, etype_data, separate_coo_row_indices.data_ptr<Idx>(),
              separate_coo_col_indices.data_ptr<Idx>(), num_edges,
              etype_mapper_data, etype_mapper_data_col);
    } else {
      HET_fusedGatBackwardGradElErFeatSrcFused_relational_separate_coo<
          Idx, DType, CompactKind><<<nblks, nthrs, 0, stream>>>(
          gdata, etype_data, separate_coo_row_indices.data_ptr<Idx>(),
          separate_coo_col_indices.data_ptr<Idx>(), num_edges,
          etype_mapper_data, etype_mapper_data_col);
    }
  } else if constexpr (!IntegratedFormatRatherThanSeparateFlag &&
                       CSRRatherThanCOOFlag) {
    // separate CSR
    
      gdata.eids = outcsr_eids.data_ptr<Idx>();
    int64_t outcsr_num_rows = outcsr_row_ptr.numel() - 1;
    auto [nblks, nthrs] = get_type2_schedule(
        gdata.num_heads, gdata.feat_src_xlen, outcsr_num_rows);
    ETypeMapperData<Idx, CompactKind> etype_mapper_data;
    if constexpr (IsCompact(CompactKind)) {
      etype_mapper_data.unique_srcs_and_dests_rel_ptrs =
          unique_srcs_and_dests_rel_ptrs.data_ptr<Idx>();
      etype_mapper_data.unique_srcs_and_dests_node_indices =
          unique_srcs_and_dests_node_indices.data_ptr<Idx>();
    }
    ETypeData<Idx, false> etype_data{
        .etypes = outcsr_reltypes.data_ptr<Idx>(),
    };
    if constexpr (!FLAG_KERNEL_FUSED) {
      HET_fusedGatBackwardGradFeatSrc_relational_separate_csr_vertex_parallel<Idx, DType, CompactKind>
          <<<nblks, nthrs, 0, stream>>>(gdata, etype_data ,outcsr_row_ptr.data_ptr<Idx>(), 
                                        outcsr_col_indices.data_ptr<Idx>(), outcsr_num_rows,
                                        etype_mapper_data, outcsr_reltypes.numel() - 1);

      HET_fusedGatBackwardGradElEr_relational_separate_csr_vertex_parallel<Idx, DType, CompactKind>
          <<<nblks, nthrs, 0, stream>>>(gdata, etype_data ,outcsr_row_ptr.data_ptr<Idx>(), 
                                        outcsr_col_indices.data_ptr<Idx>(), outcsr_num_rows,
                                        etype_mapper_data, outcsr_reltypes.numel() - 1);
    } else {
      HET_fusedGatBackwardGradElErFeatSrcFused_relational_separate_csr_vertex_parallel<
             Idx, DType, CompactKind><<<nblks, nthrs, 0, stream>>>(
             gdata, etype_data ,outcsr_row_ptr.data_ptr<Idx>(), 
             outcsr_col_indices.data_ptr<Idx>(), outcsr_num_rows,
             etype_mapper_data, outcsr_reltypes.numel() - 1);
    }
                        
  } else {
    assert(0 && "Not implemented");
  }
}

namespace IntegratedCSR {
// TODO: pass in direct indexing
void RelationalFusedGAT(at::Tensor &outcsr_row_ptr,
                        at::Tensor &outcsr_col_indices, at::Tensor &outcsr_eids,
                        at::Tensor &outcsr_reltypes,
                        at::Tensor &unique_srcs_and_dests_rel_ptrs,
                        at::Tensor &unique_srcs_and_dests_node_indices,
                        at::Tensor &feat_src, at::Tensor &el, at::Tensor &er,
                        at::Tensor &sum, at::Tensor &exp, at::Tensor &ret,
                        at::Tensor &gradout, at::Tensor &grad_feat_src,
                        at::Tensor &grad_el, at::Tensor &grad_er, double slope,
                        bool CompactAsOfNodeFlag) {
  at::Tensor dummy_tensor;
  if (!CompactAsOfNodeFlag) {
    _RelationalFusedGAT<int64_t, float, true, CompactAsOfNodeKind::Disabled,
                        true, true>(
        dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, outcsr_row_ptr,
        outcsr_col_indices, outcsr_eids, outcsr_reltypes,
        unique_srcs_and_dests_rel_ptrs, dummy_tensor,
        unique_srcs_and_dests_node_indices, dummy_tensor, dummy_tensor,
        dummy_tensor, feat_src, el, er, sum, exp, ret, gradout, grad_feat_src,
        grad_el, grad_er, slope);
  } else {
    _RelationalFusedGAT<int64_t, float, true, CompactAsOfNodeKind::Enabled,
                        true, true>(
        dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, outcsr_row_ptr,
        outcsr_col_indices, outcsr_eids, outcsr_reltypes,
        unique_srcs_and_dests_rel_ptrs, dummy_tensor,
        unique_srcs_and_dests_node_indices, dummy_tensor, dummy_tensor,
        dummy_tensor, feat_src, el, er, sum, exp, ret, gradout, grad_feat_src,
        grad_el, grad_er, slope);
  }
}
}  // namespace IntegratedCSR

namespace SeparateCOO {
namespace EdgeParallel {
void RelationalFusedGAT(at::Tensor &separate_coo_eids,
                        at::Tensor &separate_coo_rel_ptrs,
                        at::Tensor &separate_coo_row_indices,
                        at::Tensor &separate_coo_col_indices, int64_t IntKind,
                        torch::Dict<std::string, at::Tensor> args_tensor_dict,
                        at::Tensor &feat_src, at::Tensor &el, at::Tensor &er,
                        at::Tensor &sum, at::Tensor &exp, at::Tensor &ret,
                        at::Tensor &gradout, at::Tensor &grad_feat_src,
                        at::Tensor &grad_el, at::Tensor &grad_er,
                        double slope) {
  at::Tensor dummy_tensor;
  auto Kind = static_cast<CompactAsOfNodeKind>(IntKind);
  if (Kind == CompactAsOfNodeKind::EnabledWithDirectIndexing) {
    at::Tensor edata_idx_to_inverse_idx =
        args_tensor_dict.at("edata_idx_to_inverse_idx");

    _RelationalFusedGAT<int64_t, float, true,
                        CompactAsOfNodeKind::EnabledWithDirectIndexing, false,
                        false>(
        separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
        separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
        dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor,
        edata_idx_to_inverse_idx, dummy_tensor, feat_src, el, er, sum, exp, ret,
        gradout, grad_feat_src, grad_el, grad_er, slope);

  } else if (Kind ==
             CompactAsOfNodeKind::EnabledWithDualListWithDirectIndexing) {
    at::Tensor edata_idx_to_inverse_idx_row =
        args_tensor_dict.at("edata_idx_to_inverse_idx_row");
    at::Tensor edata_idx_to_inverse_idx_col =
        args_tensor_dict.at("edata_idx_to_inverse_idx_col");

    _RelationalFusedGAT<
        int64_t, float, true,
        CompactAsOfNodeKind::EnabledWithDualListWithDirectIndexing, false,
        false>(
        separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
        separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
        dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor,
        edata_idx_to_inverse_idx_row, edata_idx_to_inverse_idx_col, feat_src,
        el, er, sum, exp, ret, gradout, grad_feat_src, grad_el, grad_er, slope);

  } else if (Kind == CompactAsOfNodeKind::EnabledWithDualList) {
    at::Tensor unique_srcs_and_dests_rel_ptrs =
        args_tensor_dict.at("unique_srcs_and_dests_rel_ptrs");
    at::Tensor unique_srcs_and_dests_rel_ptrs_col =
        args_tensor_dict.at("unique_srcs_and_dests_rel_col");
    at::Tensor unique_srcs_and_dests_node_indices =
        args_tensor_dict.at("unique_srcs_and_dests_node_indices_row");
    at::Tensor unique_srcs_and_dests_node_indices_col =
        args_tensor_dict.at("unique_srcs_and_dests_node_indices_col");

    _RelationalFusedGAT<int64_t, float, true,
                        CompactAsOfNodeKind::EnabledWithDualList, false, false>(
        separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
        separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
        dummy_tensor, unique_srcs_and_dests_rel_ptrs,
        unique_srcs_and_dests_rel_ptrs_col, unique_srcs_and_dests_node_indices,
        unique_srcs_and_dests_node_indices_col, dummy_tensor, dummy_tensor,
        feat_src, el, er, sum, exp, ret, gradout, grad_feat_src, grad_el,
        grad_er, slope);
  } else if (Kind == CompactAsOfNodeKind::Enabled) {
    at::Tensor unique_srcs_and_dests_rel_ptrs =
        args_tensor_dict.at("unique_srcs_and_dests_rel_ptrs");
    at::Tensor unique_srcs_and_dests_node_indices =
        args_tensor_dict.at("unique_srcs_and_dests_node_indices");
    _RelationalFusedGAT<int64_t, float, true, CompactAsOfNodeKind::Enabled,
                        false, false>(
        separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
        separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
        dummy_tensor, unique_srcs_and_dests_rel_ptrs, dummy_tensor,
        unique_srcs_and_dests_node_indices, dummy_tensor, dummy_tensor,
        dummy_tensor, feat_src, el, er, sum, exp, ret, gradout, grad_feat_src,
        grad_el, grad_er, slope);
  } else if (Kind == CompactAsOfNodeKind::Disabled) {
    _RelationalFusedGAT<int64_t, float, true, CompactAsOfNodeKind::Disabled,
                        false, false>(
        separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
        separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
        dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor,
        dummy_tensor, dummy_tensor, feat_src, el, er, sum, exp, ret, gradout,
        grad_feat_src, grad_el, grad_er, slope);
  } else {
    printf("%ld\n", IntKind);
    throw std::runtime_error("Invalid CompactAsOfNodeKind");
  }
}
}  // namespace EdgeParallel
}  // namespace SeparateCOO

namespace SeparateCSR
{
  void RelationalFusedGAT(at::Tensor &outcsr_row_ptr,
                        at::Tensor &outcsr_col_indices, at::Tensor &outcsr_eids,
                        at::Tensor &outcsr_reltypes,
                        at::Tensor &unique_srcs_and_dests_rel_ptrs,
                        at::Tensor &unique_srcs_and_dests_node_indices,
                        at::Tensor &feat_src, at::Tensor &el, at::Tensor &er,
                        at::Tensor &sum, at::Tensor &exp, at::Tensor &ret,
                        at::Tensor &gradout, at::Tensor &grad_feat_src,
                        at::Tensor &grad_el, at::Tensor &grad_er, double slope,
                        bool CompactAsOfNodeFlag) {
  at::Tensor dummy_tensor;
  if (!CompactAsOfNodeFlag) {
    _RelationalFusedGAT<int64_t, float, true, CompactAsOfNodeKind::Disabled,
                        false, true>( // changed last to second templated bool to false
        dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, outcsr_row_ptr,
        outcsr_col_indices, outcsr_eids, outcsr_reltypes,
        unique_srcs_and_dests_rel_ptrs, dummy_tensor,
        unique_srcs_and_dests_node_indices, dummy_tensor, dummy_tensor,
        dummy_tensor, feat_src, el, er, sum, exp, ret, gradout, grad_feat_src,
        grad_el, grad_er, slope);
  } else {
    _RelationalFusedGAT<int64_t, float, true, CompactAsOfNodeKind::Enabled,
                        false, true>( // changed last to second templated bool to false
        dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, outcsr_row_ptr,
        outcsr_col_indices, outcsr_eids, outcsr_reltypes,
        unique_srcs_and_dests_rel_ptrs, dummy_tensor,
        unique_srcs_and_dests_node_indices, dummy_tensor, dummy_tensor,
        dummy_tensor, feat_src, el, er, sum, exp, ret, gradout, grad_feat_src,
        grad_el, grad_er, slope);
  }
}
  
} // namespace SeparateCSR

}  // namespace BckProp
}  // namespace RGAT
}  // namespace TorchExport
}  // namespace HET

using namespace HET::TorchExport;
TORCH_LIBRARY_FRAGMENT(torch_hetero_edgesoftmax, m) {
  // RGAT Declaration
  // RGAT Relational SpMM
  m.def("relational_fused_gat_csr",
        RGAT::FwProp::IntegratedCSR::RelationalFusedGAT);
  m.def("backward_relational_fused_gat_csr",
        RGAT::BckProp::IntegratedCSR::RelationalFusedGAT);
  m.def("relational_fused_gat_separate_coo",
        RGAT::FwProp::SeparateCOO::EdgeParallel::RelationalFusedGAT);
  m.def("backward_relational_fused_gat_separate_coo",
        RGAT::BckProp::SeparateCOO::EdgeParallel::RelationalFusedGAT);
  m.def("backward_relational_fused_gat_separate_csr",
        RGAT::BckProp::SeparateCSR::RelationalFusedGAT);
}
