#pragma once
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "DGLHackKernel/GAT/FusedGAT.cu.h"
#include "DGLHackKernel/GAT/FusedGATBackward.cu.h"
#include "DGLHackKernel/RGAT/RGATLayersBackwardKernels.cu.h"
#include "DGLHackKernel/RGAT/RGATLayersKernels.cu.h"
#include "DGLHackKernel/mysgemm/my_shmem_sgemm_func.cu.h"
#include "DGLHackKernel/mysgemm/mysgemm_KernelsBlockConfigurations.h"
#include "GATOps.inc.h"

template </*int XPU, */ typename Idx, typename DType, bool CompactAsOfNodeFlag>
void _RelationalFusedGATKernel_wrapper_integratedcsr(
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
  gdata.eids = incsr_eids.data_ptr<Idx>();

  // Configure kernel launch parameters.
  int nthrs_x = 32;
  int nthrs_y = 1;
  int nblks_x = (el_xlen + nthrs_x - 1) / (nthrs_x);
  int nblks_y = std::min(gdata.n, MAX_NBLKS);
  const dim3 nblks(nblks_x, nblks_y);
  const dim3 nthrs(nthrs_x, nthrs_y);
  int64_t incsr_num_rows = incsr_row_ptr.numel() - 1;

  gatExpLeakyReluSumKernel<Idx, DType, CompactAsOfNodeFlag, true>
      <<<nblks, nthrs, 0, stream>>>(
          gdata, incsr_row_ptr.data_ptr<Idx>(), incsr_col_idx.data_ptr<Idx>(),
          nullptr, incsr_num_rows,
          unique_srcs_and_dests_rel_ptr.data_ptr<Idx>(),
          unique_srcs_and_dests_node_indices.data_ptr<Idx>());

  nthrs_x = FindNumThreads(el_xlen, 64);
  nthrs_y = FindNumThreads(gdata.feat_src_hidden, MAX_NTHRS / nthrs_x);
  nblks_x = 1;
  nblks_y = std::min(gdata.n, MAX_NBLKS);
  const dim3 nthrs2(nthrs_x, nthrs_y);
  const dim3 nblks2(nblks_x, nblks_y);

  gatSumProdZipDivKernel<Idx, DType, CompactAsOfNodeFlag, true>
      <<<nblks2, nthrs2, 0, stream>>>(
          gdata, incsr_row_ptr.data_ptr<Idx>(), incsr_col_idx.data_ptr<Idx>(),
          incsr_reltypes.data_ptr<Idx>(), incsr_num_rows,
          unique_srcs_and_dests_rel_ptr.data_ptr<Idx>(),
          unique_srcs_and_dests_node_indices.data_ptr<Idx>());
}

template </*int XPU, */ typename Idx, typename DType, bool FLAG_KERNEL_FUSED,
          bool CompactAsOfNodeFlag>
void _BackwardRelationalFusedGATKernel_wrapper_integratedcsr(
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
  int64_t el_xlen = ComputeXLength(el);
  int64_t feat_src_xlen = ComputeXLength(feat_src);
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
  gdata.n = el.numel() / el_xlen;
  gdata.e_xlen = el_xlen;
  gdata.feat_src_xlen = feat_src_xlen;
  gdata.feat_src_hidden = feat_src_xlen / el_xlen;
  gdata.eids = outcsr_eids.data_ptr<Idx>();
  int nthrs_x = FindNumThreads(el_xlen, 64);
  int nthrs_y = FindNumThreads(gdata.feat_src_hidden, MAX_NTHRS / nthrs_x);
  int nblks_x = 1;
  int nblks_y = std::min(gdata.n, MAX_NBLKS);
  int64_t outcsr_num_rows = outcsr_row_ptr.numel() - 1;
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
    fusedGatBackwardGradElErFeatSrcFused<Idx, DType, CompactAsOfNodeFlag, true>
        <<<nblks, nthrs, 0, stream>>>(
            gdata, outcsr_row_ptr.data_ptr<Idx>(),
            outcsr_col_idx.data_ptr<Idx>(), outcsr_reltypes.data_ptr<Idx>(),
            outcsr_num_rows, unique_srcs_and_dests_rel_ptr.data_ptr<Idx>(),
            unique_srcs_and_dests_node_indices.data_ptr<Idx>());
  }
}

constexpr auto RelationalFusedGATKernel_wrapper_integratedcsr =
    _RelationalFusedGATKernel_wrapper_integratedcsr<int64_t, float, false>;
constexpr auto BackwardRelationalFusedGATKernel_wrapper_integratedcsr =
    _BackwardRelationalFusedGATKernel_wrapper_integratedcsr<int64_t, float,
                                                            true, false>;
constexpr auto
    RGATRelationalFusedGATKernelCompactAsOfNode_wrapper_integratedcsr =
        _RelationalFusedGATKernel_wrapper_integratedcsr<int64_t, float, true>;
constexpr auto
    BackwardRGATRelationalFusedGATKernelCompactAsOfNode_wrapper_integratedcsr =
        _BackwardRelationalFusedGATKernel_wrapper_integratedcsr<int64_t, float,
                                                                true, true>;

template <int BLOCK_SIZE, bool CompactAsOfNodeFlag>
void _RGATRelationalMatMul_wrapper_separatecoo(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_node_indices,
    at::Tensor& separate_coo_eids, at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& weights,
    at::Tensor& input, at::Tensor& ret) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int64_t num_relations = separate_coo_relptrs.numel() - 1;
  const int64_t num_edges = separate_coo_eids.numel();
  const int64_t num_heads = weights.size(1);
  const int64_t num_input_dim = weights.size(2);
  const int64_t num_output_dim =
      weights.size(3) * num_heads;  // weight shape (num_relations, n_heads,
                                    // in_feat, out_feat // n_heads)
  auto [num_blocks_assignment_for_same_relation_vect,
        num_blocks_assignment_for_all_prev_relation_vect] =
      get_schedule_by_relation_kernel_launch_metadata<false, false, Idx*>(
          -1, num_relations, BLOCK_SIZE,
          separate_coo_relptrs.data_ptr<int64_t>(),
          separate_coo_relptrs.data_ptr<int64_t>() + num_relations);

  thrust::device<int> dev_num_blocks_assignment_for_same_relation_vect(
      num_blocks_assignment_for_same_relation_vect.data(),
      num_blocks_assignment_for_same_relation_vect.data() + num_relations);
  thrust::device<int> dev_num_blocks_assignment_for_all_prev_relation_vect(
      num_blocks_assignment_for_all_prev_relation_vect.data(),
      num_blocks_assignment_for_all_prev_relation_vect.data() + num_relations);
  const dim3 nblks(ceil_div<>(num_output_dim / num_heads, BLOCK_SIZE),
                   ceil_div(num_edges, BLOCK_SIZE), num_heads);
  const dim3 nthrs(BLOCK_SIZE, BLOCK_SIZE);
  if constexpr (CompactAsOfNodeFlag) {
    RGNNFeatCompactFWProp<BLOCK_SIZE, int64_t, int64_t*>
        <<<nblks, nthrs, 0, stream>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(),
            ret.data_ptr<float>(),
            separate_coo_node_indices.data_ptr<int64_t>(),
            separate_coo_relptrs.data_ptr<int64_t>(),
            unique_srcs_and_dests_rel_ptr.data_ptr<int64_t>(),
            unique_srcs_and_dests_node_indices.data_ptr<int64_t>(), num_edges,
            num_input_dim, num_output_dim, num_heads,
            thrust::raw_pointer_cast(
                dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
            num_relations);
  } else {
    RGNNFeatPerEdgeFWProp<BLOCK_SIZE, int64_t, int64_t*>
        <<<nblks, nthrs, 0, stream>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(),
            ret.data_ptr<float>(),
            separate_coo_node_indices.data_ptr<int64_t>(),
            separate_coo_relptrs.data_ptr<int64_t>(),
            separate_coo_eids.data_ptr<int64_t>(), num_edges, num_input_dim,
            num_output_dim, num_heads,
            thrust::raw_pointer_cast(
                dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
            num_relations);
  }
}

void RGATRelationalMatMul_wrapper_separatecoo(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_node_indices,
    at::Tensor& separate_coo_eids, at::Tensor& weights, at::Tensor& input,
    at::Tensor& ret) {
  _RGATRelationalMatMul_wrapper_separatecoo<16, false>(
      separate_coo_relptrs, separate_coo_node_indices, separate_coo_eids,
      /*dummy*/ separate_coo_eids, /*dummy*/ separate_coo_eids, weights, input,
      ret);
}

void RGATRelationalMatMulCompactAsOfNode_wrapper_unique_rel_node_indices(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_node_indices,
    at::Tensor& separate_coo_eids, at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_idx, at::Tensor& weight,
    at::Tensor& node_feat, at::Tensor& ret) {
  _RGATRelationalMatMul_wrapper_separatecoo<16, true>(
      separate_coo_relptrs, separate_coo_node_indices, separate_coo_eids,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_idx, weight,
      node_feat, ret);
}

template <int BLOCK_SIZE, bool CompactAsOfNodeFlag>
void _BackwardRGATRelationalMatMul_wrapper_separatecoo(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_node_indices,
    at::Tensor& separate_coo_eids, at::Tensor& weights_transposed,
    at::Tensor& input, at::Tensor& gradout, at::Tensor& grad_input,
    at::Tensor& grad_weights) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int64_t num_relations = separate_coo_relptrs.numel() - 1;
  const int64_t num_edges = separate_coo_eids.numel();
  const int64_t num_heads = weights_transposed.size(1);
  const int64_t num_input_dim = weights_transposed.size(3);
  const int64_t num_output_dim =
      weights_transposed.size(2) *
      num_heads;  // weight shape (num_relations, n_heads, in_feat, out_feat //
                  // n_heads)
  auto [num_blocks_assignment_for_same_relation_vect,
        num_blocks_assignment_for_all_prev_relation_vect] =
      get_schedule_by_relation_kernel_launch_metadata<false, false, Idx*>(
          -1, num_relations, BLOCK_SIZE,
          separate_coo_relptrs.data_ptr<int64_t>(),
          separate_coo_relptrs.data_ptr<int64_t>() + num_relations);

  thrust::device<int> dev_num_blocks_assignment_for_same_relation_vect(
      num_blocks_assignment_for_same_relation_vect.data(),
      num_blocks_assignment_for_same_relation_vect.data() + num_relations);
  thrust::device<int> dev_num_blocks_assignment_for_all_prev_relation_vect(
      num_blocks_assignment_for_all_prev_relation_vect.data(),
      num_blocks_assignment_for_all_prev_relation_vect.data() + num_relations);
  const dim3 nblks(ceil_div<>(num_output_dim / num_heads, BLOCK_SIZE),
                   ceil_div(num_edges, BLOCK_SIZE), num_heads);
  const dim3 nthrs(BLOCK_SIZE, BLOCK_SIZE);
}

void BackwardRGATRelationalMatMulCompactAsOfNode_wrapper_unique_rel_node_indices(
    at::Tensor& unqiue_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_idx, at::Tensor& weight,
    at::Tensor& node_feat, at::Tensor& ret, at::Tensor& gradout,
    at::Tensor& grad_weight, at::Tensor& grad_node_feat) {
  //_BackwardRGATRelationalMatMul_wrapper_separatecoo<16, true>()
}

void BackwardRGATRelationalMatMul_wrapper_separatecoo() {
  //_BackwardRGATRelationalMatMul_wrapper_separatecoo<16,
  // false>(separate_coo_relptrs, separate_coo_node_indices);
}