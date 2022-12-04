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
          bool CSRRatherThanCOOFlag>
void _RelationalFusedGATKernel(
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
  gdata.num_heads = el_xlen;
  gdata.feat_src_xlen = feat_src_xlen;
  // gdata.feat_src_hidden = feat_src_xlen / el_xlen;
  // gdata.ret_xlen = ret_len;

  if constexpr (IntegratedFormatRatherThanSeparateFlag &&
                CSRRatherThanCOOFlag) {
    // Integrated CSR
    gdata.eids = incsr_eids.data_ptr<Idx>();
    // Configure kernel launch parameters.
    // TODO: we can safely reshape (nthrs_x, nthrs_y) to assign more y dimension
    // to rows as usually n_head is smaller than 32
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
    nthrs_y =
        SeastarFindNumThreads(feat_src_xlen / el_xlen, MAX_NTHRS / nthrs_x);
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
    // TODO: we can safely reshape (nthrs_x, nthrs_y) to assign more y dimension
    // to edges as usually n_head is smaller than 32
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
    nthrs_y =
        SeastarFindNumThreads(feat_src_xlen / el_xlen, MAX_NTHRS / nthrs_x);
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

void RelationalFusedGATKernelCompactAsOfNode_edge_parallel_separatecoo(
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_rel_ptrs,
    at::Tensor& separate_coo_row_indices, at::Tensor& separate_coo_col_indices,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, double slope) {
  at::Tensor dummy_tensor;
  _RelationalFusedGATKernel<int64_t, float, true, false, false>(
      separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
      separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
      dummy_tensor, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices, feat_src, el, er, sum, exp, ret,
      slope);
}

void RelationalFusedGATKernel_edge_parallel_separatecoo(
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_rel_ptrs,
    at::Tensor& separate_coo_row_indices, at::Tensor& separate_coo_col_indices,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, double slope) {
  at::Tensor dummy_tensor;
  _RelationalFusedGATKernel<int64_t, float, false, false, false>(
      separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
      separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
      dummy_tensor, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices, feat_src, el, er, sum, exp, ret,
      slope);
}

void RelationalFusedGATKernel_integratedcsr(
    at::Tensor& incsr_row_ptr, at::Tensor& incsr_col_idx,
    at::Tensor& incsr_eids, at::Tensor& incsr_reltypes,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, double slope) {
  at::Tensor dummy_tensor;
  _RelationalFusedGATKernel<int64_t, float, false, true, true>(
      dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, incsr_row_ptr,
      incsr_col_idx, incsr_eids, incsr_reltypes, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices, feat_src, el, er, sum, exp, ret,
      slope);
}

void RelationalFusedGATKernelCompactAsOfNode_integratedcsr(
    at::Tensor& incsr_row_ptr, at::Tensor& incsr_col_idx,
    at::Tensor& incsr_eids, at::Tensor& incsr_reltypes,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, double slope) {
  at::Tensor dummy_tensor;
  _RelationalFusedGATKernel<int64_t, float, true, true, true>(
      dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, incsr_row_ptr,
      incsr_col_idx, incsr_eids, incsr_reltypes, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices, feat_src, el, er, sum, exp, ret,
      slope);
}
}  // namespace FwProp
namespace BckProp {
template </*int XPU, */ typename Idx, typename DType, bool FLAG_KERNEL_FUSED,
          bool CompactAsOfNodeFlag, bool IntegratedFormatRatherThanSeparateFlag,
          bool CSRRatherThanCOOFlag>
void _RelationalFusedGATKernel(
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
  gdata.num_heads = el_xlen;
  gdata.feat_src_xlen = feat_src_xlen;
  // gdata.feat_src_hidden = feat_src_xlen / el_xlen;

  if constexpr (IntegratedFormatRatherThanSeparateFlag &&
                CSRRatherThanCOOFlag) {
    // Integrated CSR
    gdata.eids = outcsr_eids.data_ptr<Idx>();
    int nthrs_x = SeastarFindNumThreads(el_xlen, 64);
    int nthrs_y =
        SeastarFindNumThreads(feat_src_xlen / el_xlen, MAX_NTHRS / nthrs_x);
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
        SeastarFindNumThreads(feat_src_xlen / el_xlen, MAX_NTHRS / nthrs_x);
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

void RelationalFusedGATKernel_integratedcsr(
    at::Tensor& outcsr_row_ptr, at::Tensor& outcsr_col_idx,
    at::Tensor& outcsr_eids, at::Tensor& outcsr_reltypes,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, at::Tensor& gradout, at::Tensor& grad_feat_src,
    at::Tensor& grad_el, at::Tensor& grad_er, double slope) {
  at::Tensor dummy_tensor;
  _RelationalFusedGATKernel<int64_t, float, true, false, true, true>(
      dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, outcsr_row_ptr,
      outcsr_col_idx, outcsr_eids, outcsr_reltypes,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
      feat_src, el, er, sum, exp, ret, gradout, grad_feat_src, grad_el, grad_er,
      slope);
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
  _RelationalFusedGATKernel<int64_t, float, true, true, true, true>(
      dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, outcsr_row_ptr,
      outcsr_col_idx, outcsr_eids, outcsr_reltypes,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
      feat_src, el, er, sum, exp, ret, gradout, grad_feat_src, grad_el, grad_er,
      slope);
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
  _RelationalFusedGATKernel<int64_t, float, true, true, false, false>(
      separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
      separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
      dummy_tensor, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices, feat_src, el, er, sum, exp, ret,
      gradout, grad_feat_src, grad_el, grad_er, slope);
}

void RelationalFusedGATKernel_edge_parallel_separatecoo(
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_rel_ptrs,
    at::Tensor& separate_coo_row_indices, at::Tensor& separate_coo_col_indices,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, at::Tensor& gradout, at::Tensor& grad_feat_src,
    at::Tensor& grad_el, at::Tensor& grad_er, double slope) {
  at::Tensor dummy_tensor;
  _RelationalFusedGATKernel<int64_t, float, false, false, false, false>(
      separate_coo_eids, separate_coo_rel_ptrs, separate_coo_row_indices,
      separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
      dummy_tensor, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices, feat_src, el, er, sum, exp, ret,
      gradout, grad_feat_src, grad_el, grad_er, slope);
}

}  // namespace BckProp
}  // namespace RGAT
}  // namespace TorchExport
}  // namespace HET
