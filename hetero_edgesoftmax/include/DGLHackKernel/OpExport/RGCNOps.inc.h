#pragma once

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "DGLHackKernel/RGCN/RGCNBackwardKernelsEdgeParallel.cu.h"
#include "DGLHackKernel/RGCN/RGCNKernelsEdgeParallel.cu.h"
#include "DGLHackKernel/RGNN/my_shmem_sgemm_func_rgcn_hgt.cu.h"
#include "DGLHackKernel/RGNN/mysgemm_KernelsBlockConfigurations.h"
#include "ThreadingGridsBlocksSchedules.h"

// TODO: create dummy tensor instead whenever unused field in torch export
// functions

namespace HET {

namespace TorchExport {
namespace RGCN {
namespace FwProp {
template <typename Idx, typename DType>
void Layer1_NodeMeanAggregation_CompactAsOfNode_SeparateCOO(
    at::Tensor &separate_coo_eids, at::Tensor &separate_coo_rel_ptrs,
    at::Tensor &separate_coo_row_indices, at::Tensor &separate_coo_col_indices,
    at::Tensor &unique_srcs_and_dests_rel_ptr,
    at::Tensor &unique_srcs_and_dests_node_indices, at::Tensor &feat_src,
    at::Tensor &enorm, at::Tensor &ret) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  RGCNData<Idx, DType> gdata{.feat_src_xlen = SeastarComputeXLength<>(feat_src),
                             .eids = separate_coo_eids.data_ptr<Idx>(),
                             .feat_src = feat_src.data_ptr<DType>(),
                             .enorm = enorm.data_ptr<DType>(),
                             .ret = ret.data_ptr<DType>()};
  // separate coo
  int64_t num_edges = separate_coo_row_indices.numel();
  int64_t num_relations = separate_coo_rel_ptrs.numel() - 1;
  // adapted from launch configuration of
  // HET_gatSumProdZipDivKernel_relational_separate_coo in
  // [[hetero_edgesoftmax/python/backend/rgcn_layers_and_funcs.py]] NB: updated
  // to Type 2 Schedule:
  // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
  // head -> threadIdx.y
  // node -> blockIdx.y
  // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x
  auto [nblks2, nthrs2] =
      get_type2_schedule(num_edges, /*num_heads*/ 1, num_edges);
  HET_rgcnNodeMeanAggregation_edge_parallel<Idx, DType, true>
      <<<nblks2, nthrs2, 0, stream>>>(
          gdata, separate_coo_rel_ptrs.data_ptr<Idx>(),
          separate_coo_row_indices.data_ptr<Idx>(),
          separate_coo_col_indices.data_ptr<Idx>(), num_edges,
          unique_srcs_and_dests_rel_ptr.data_ptr<Idx>(),
          unique_srcs_and_dests_node_indices.data_ptr<Idx>(), num_relations);
}
void Layer1_SeparateCOO(at::Tensor &separate_coo_relptrs,
                        at::Tensor &separate_coo_eids,
                        at::Tensor &separate_coo_row_idx,
                        at::Tensor &separate_coo_col_idx,
                        at::Tensor &node_feat_input, at::Tensor &weights,
                        at::Tensor &edge_norm, at::Tensor &node_feat_output) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  constexpr int WORK_BLOCK_SIZE = 32;
  constexpr bool COARSEN_FACTOR_2_FLAG_X = true;

  constexpr bool COARSEN_FACTOR_2_FLAG_Y = true;
  constexpr int THREADING_BLOCK_SIZE_X =
      COARSEN_FACTOR_2_FLAG_X ? WORK_BLOCK_SIZE / 2 : WORK_BLOCK_SIZE;
  constexpr int THREADING_BLOCK_SIZE_Y =
      COARSEN_FACTOR_2_FLAG_Y ? WORK_BLOCK_SIZE / 2 : WORK_BLOCK_SIZE;
  const int64_t num_relations = (separate_coo_relptrs.numel() - 1);
  const int64_t num_heads = 1;
  const int64_t num_input_dim = weights.size(1);
  const int64_t num_output_dim = weights.size(2);
  int64_t num_edges = separate_coo_eids.numel();
  int grid_dim_y =
      std::min(ceil_div<>(num_edges, (int64_t)WORK_BLOCK_SIZE), (int64_t)32768);
  at::Tensor separate_coo_relptrs_cpu_contiguous =
      separate_coo_relptrs.cpu().contiguous();
  std::vector<int> num_blocks_assignment_for_same_relation_vect,
      num_blocks_assignment_for_all_prev_relation_vect;
  std::tie(num_blocks_assignment_for_same_relation_vect,
           num_blocks_assignment_for_all_prev_relation_vect) =
      get_schedule_by_relation_kernel_launch_metadata<false, false, int64_t *>(
          grid_dim_y, num_relations, WORK_BLOCK_SIZE,
          separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>(),
          separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>() +
              num_relations + 1);
  grid_dim_y = num_blocks_assignment_for_all_prev_relation_vect.back();

  thrust::device_vector<int> dev_num_blocks_assignment_for_same_relation_vect(
      num_blocks_assignment_for_same_relation_vect.begin(),
      num_blocks_assignment_for_same_relation_vect.end());
  thrust::device_vector<int>
      dev_num_blocks_assignment_for_all_prev_relation_vect(
          num_blocks_assignment_for_all_prev_relation_vect.begin(),
          num_blocks_assignment_for_all_prev_relation_vect.end());
  // NB: my shmem sgemm matmul scheme
  const dim3 nblks(ceil_div<>(num_output_dim, (long)WORK_BLOCK_SIZE),
                   grid_dim_y, 1);
  const dim3 nthrs(THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y);
  // TODO: KWU: allow more dtype options in this file
  HET_RGCNMatmulNoScatterGatherListFwProp<COARSEN_FACTOR_2_FLAG_X,
                                          COARSEN_FACTOR_2_FLAG_Y,
                                          WORK_BLOCK_SIZE, int64_t, int64_t *>
      <<<nblks, nthrs, 0, stream>>>(
          node_feat_input.data_ptr<float>(), weights.data_ptr<float>(),
          node_feat_output.data_ptr<float>(), edge_norm.data_ptr<float>(),
          separate_coo_row_idx.data_ptr<int64_t>(),
          separate_coo_col_idx.data_ptr<int64_t>(),
          separate_coo_eids.data_ptr<int64_t>(),
          separate_coo_relptrs.data_ptr<int64_t>(),
          thrust::raw_pointer_cast(
              dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
          num_relations, num_input_dim, num_output_dim);
}

namespace IntegratedCSR {
template </*int XPU, */ typename Idx, typename DType, bool HybridAssignmentFlag>
void _LayerImpl(at::Tensor &csr_rowptr, at::Tensor &csr_col_idx,
                at::Tensor &csr_eids, at::Tensor &csr_reltypes,
                at::Tensor &hidden, at::Tensor &weight, at::Tensor &norm,
                at::Tensor &ret, bool layer1_flag,
                int num_blocks_on_blocks_per_node) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto range_data = csr_rowptr.data_ptr<Idx>();
  auto ids_data = csr_col_idx.data_ptr<Idx>();
  auto eids_data = csr_eids.data_ptr<Idx>();
  auto typeids_data = csr_reltypes.data_ptr<Idx>();
  DType *hidden_data = hidden.numel() == 0 ? nullptr : hidden.data_ptr<DType>();
  auto weight_data = weight.data_ptr<DType>();
  auto norm_data = norm.data_ptr<DType>();
  auto ret_data = ret.data_ptr<DType>();

  Idx num_nodes = csr_rowptr.numel() - 1;
  Idx num_edges = csr_eids.numel();
  int nblks = num_nodes;

  if constexpr (HybridAssignmentFlag) {
    assert(num_blocks_on_blocks_per_node >= 0);
  } else {
    assert(num_blocks_on_blocks_per_node == -1);
  }

  if (layer1_flag) {
    Idx ntypes = weight.size(0);
    Idx feat_len_y = weight.size(1);
    Idx feat_len_x = weight.size(2);
    // int nthrs = feat_len_y * feat_len_x;
    int nthrs = feat_len_x < 512 ? 512 : feat_len_x;
    if constexpr (HybridAssignmentFlag) {
      HET_Seastar_RgcnLayer1KernelHybridAssignImpl<Idx, DType>
          <<<nblks, nthrs, 0, stream>>>(
              range_data, ids_data, eids_data, typeids_data, hidden_data,
              weight_data, norm_data, ret_data, num_nodes, feat_len_y,
              feat_len_x, ntypes, num_blocks_on_blocks_per_node);
    } else {
      HET_Seastar_RgcnLayer1KernelImpl<Idx, DType><<<nblks, nthrs, 0, stream>>>(
          range_data, ids_data, eids_data, typeids_data, hidden_data,
          weight_data, norm_data, ret_data, num_nodes, feat_len_y, feat_len_x,
          ntypes);
    }
  } else {
    Idx ntypes = weight.size(1);
    Idx feat_len = weight.size(2);
    int nthrs = feat_len;
    if constexpr (HybridAssignmentFlag) {
      HET_Seastar_RgcnLayer0KernelHybridAssignImpl<Idx, DType>
          <<<nblks, nthrs, 0, stream>>>(range_data, ids_data, eids_data,
                                        typeids_data, weight_data, norm_data,
                                        ret_data, num_nodes, feat_len, ntypes,
                                        num_blocks_on_blocks_per_node);
    } else {
      HET_Seastar_RgcnLayer0KernelImpl<Idx, DType><<<nblks, nthrs, 0, stream>>>(
          range_data, ids_data, eids_data, typeids_data, weight_data, norm_data,
          ret_data, num_nodes, feat_len, ntypes);
    }
  }
}

// template </*int XPU, */ typename Idx, typename DType>
void Layer0Impl(at::Tensor &csr_rowptr, at::Tensor &csr_col_idx,
                at::Tensor &csr_eids, at::Tensor &csr_reltypes,
                at::Tensor &weight, at::Tensor &norm, at::Tensor &ret) {
  // NB: graphiler, seastar by default uses int64_t
  at::Tensor dummy_tensor;
  _LayerImpl<int64_t, float, false>(csr_rowptr, csr_col_idx, csr_eids,
                                    csr_reltypes, /*dummy_hidden*/ dummy_tensor,
                                    weight, norm, ret, false, -1);
}

void Layer0HybridAssignmentImpl(at::Tensor &csr_rowptr, at::Tensor &csr_col_idx,
                                at::Tensor &csr_eids, at::Tensor &csr_reltypes,
                                at::Tensor &weight, at::Tensor &norm,
                                at::Tensor &ret,
                                int64_t num_blocks_on_blocks_per_node) {
  // NB: graphiler, seastar by default uses int64_t
  at::Tensor dummy_tensor;
  _LayerImpl<int64_t, float, true>(csr_rowptr, csr_col_idx, csr_eids,
                                   csr_reltypes, /*dummy_hidden*/ dummy_tensor,
                                   weight, norm, ret, false,
                                   num_blocks_on_blocks_per_node);
}

// template </*int XPU, */ typename Idx, typename DType>
void Layer1Impl(at::Tensor &csr_rowptr, at::Tensor &csr_col_idx,
                at::Tensor &csr_eids, at::Tensor &csr_reltypes,
                at::Tensor &hidden, at::Tensor &weight, at::Tensor &norm,
                at::Tensor &ret) {
  // NB: graphiler, seastar by default uses int64_t
  _LayerImpl<int64_t, float, false>(csr_rowptr, csr_col_idx, csr_eids,
                                    csr_reltypes, hidden, weight, norm, ret,
                                    true, -1);
}

void Layer1HybridAssignmentImpl(at::Tensor &csr_rowptr, at::Tensor &csr_col_idx,
                                at::Tensor &csr_eids, at::Tensor &csr_reltypes,
                                at::Tensor &hidden, at::Tensor &weight,
                                at::Tensor &norm, at::Tensor &ret,
                                int64_t num_blocks_on_blocks_per_node) {
  // NB: graphiler, seastar by default uses int64_t
  _LayerImpl<int64_t, float, true>(csr_rowptr, csr_col_idx, csr_eids,
                                   csr_reltypes, hidden, weight, norm, ret,
                                   true, num_blocks_on_blocks_per_node);
}
}  // namespace IntegratedCSR

namespace IntegratedCOO {
template </*int XPU, */ typename Idx, typename DType>
void _LayerImpl(at::Tensor &coo_row_idx, at::Tensor &coo_col_idx,
                at::Tensor &coo_eids, at::Tensor &coo_reltypes,
                at::Tensor &hidden, at::Tensor &weight, at::Tensor &norm,
                at::Tensor &ret, bool layer1_flag) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto row_idx_data = coo_row_idx.data_ptr<Idx>();
  auto ids_data = coo_col_idx.data_ptr<Idx>();
  auto eids_data = coo_eids.data_ptr<Idx>();
  auto typeids_data = coo_reltypes.data_ptr<Idx>();
  auto hidden_data = hidden.data_ptr<DType>();
  auto weight_data = weight.data_ptr<DType>();
  auto norm_data = norm.data_ptr<DType>();
  auto ret_data = ret.data_ptr<DType>();

  Idx num_edges = coo_eids.numel();

  if (layer1_flag) {
    Idx ntypes = weight.size(0);
    Idx feat_len_y = weight.size(1);
    Idx feat_len_x = weight.size(2);
    int nthrs = feat_len_x < 256 ? 256 : feat_len_x;
    assert(nthrs % 32 == 0);
    int nblks =
        ceil_div<>(num_edges, (int64_t)nthrs / 32);  // 32 is the warp size
    HET_Seastar_RgcnLayer1COOKernelImpl<Idx, DType>
        <<<nblks, nthrs, 0, stream>>>(row_idx_data, ids_data, eids_data,
                                      typeids_data, hidden_data, weight_data,
                                      norm_data, ret_data, num_edges,
                                      feat_len_y, feat_len_x, ntypes);
  } else {
    Idx ntypes = weight.size(1);
    Idx feat_len = weight.size(2);
    int nthrs = feat_len;
    assert(0 && "not implemented");
    // HET_Seastar_RgcnLayer0KernelImpl<Idx, DType>
    //    <<<nblks, nthrs >>>(
    //        range_data, ids_data, eids_data, typeids_data, weight_data,
    //        norm_data, ret_data, num_nodes, feat_len, ntypes);
  }
}

// template </*int XPU, */ typename Idx, typename DType>
void Layer1Impl(at::Tensor &coo_row_idx, at::Tensor &coo_col_idx,
                at::Tensor &coo_eids, at::Tensor &coo_reltypes,
                at::Tensor &hidden, at::Tensor &weight, at::Tensor &norm,
                at::Tensor &ret) {
  // NB: graphiler, seastar by default uses int64_t
  _LayerImpl<int64_t, float>(coo_row_idx, coo_col_idx, coo_eids, coo_reltypes,
                             hidden, weight, norm, ret, true);
}
}  // namespace IntegratedCOO
}  // namespace FwProp
namespace BckProp {
template <typename Idx, typename DType>
void Layer1_NodeMeanAggregation_CompactAsOfNode_SeparateCOO(
    at::Tensor &separate_coo_eids, at::Tensor &separate_coo_rel_ptrs,
    at::Tensor &separate_coo_row_indices, at::Tensor &separate_coo_col_indices,
    at::Tensor &unique_srcs_and_dests_rel_ptr,
    at::Tensor &unique_srcs_and_dests_node_indices, at::Tensor &feat_src,
    at::Tensor &enorm, at::Tensor &ret, at::Tensor &gradout,
    at::Tensor &grad_feat_src) {
  // adapted from launch configuration of
  // HET_fusedGatBackwardGradFeatSrc_relational_separate_coo in
  // [[hetero_edgesoftmax/include/DGLHackKernel/OpExport/RGATOps.inc.h]]
  // separate coo
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  BackwardRGCNData<Idx, DType> gdata{
      .feat_src_xlen = SeastarComputeXLength<>(feat_src),
      .eids = separate_coo_eids.data_ptr<Idx>(),
      .feat_src = feat_src.data_ptr<DType>(),
      .enorm = enorm.data_ptr<DType>(),
      .ret = ret.data_ptr<DType>(),
      .grad_out = gradout.data_ptr<DType>(),
      .grad_feat_src = grad_feat_src.data_ptr<DType>()};
  int64_t num_edges = separate_coo_row_indices.numel();
  int64_t num_relations = separate_coo_rel_ptrs.numel() - 1;

  // NB: updated to Type 2 Schedule:
  // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
  // head -> threadIdx.y
  // node -> blockIdx.y
  // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x
  auto [nblks, nthrs] = get_type2_schedule(1, gdata.feat_src_xlen, num_edges);

  HET_rgcnBackwardNodeMeanAggregation_edge_parallel<Idx, DType, true>
      <<<nblks, nthrs, 0, stream>>>(
          gdata, separate_coo_rel_ptrs.data_ptr<Idx>(),
          separate_coo_row_indices.data_ptr<Idx>(),
          separate_coo_col_indices.data_ptr<Idx>(), num_edges,
          unique_srcs_and_dests_rel_ptr.data_ptr<Idx>(),
          unique_srcs_and_dests_node_indices.data_ptr<Idx>(), num_relations);
}
void Layer1_SeparateCOO(
    at::Tensor &separate_coo_relptrs, at::Tensor &separate_coo_eids,
    at::Tensor &separate_coo_row_idx, at::Tensor &separate_coo_col_idx,
    at::Tensor &node_feat_input, at::Tensor &weights_transposed,
    at::Tensor &edge_norm, at::Tensor &grad_edge_norm,
    at::Tensor &delta_node_feat_input, at::Tensor &delta_node_feat_output,
    at::Tensor &delta_weights) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  constexpr int WORK_BLOCK_SIZE = 32;
  constexpr bool COARSEN_FACTOR_2_FLAG_X = true;
  constexpr bool COARSEN_FACTOR_2_FLAG_Y = true;
  constexpr int THREADING_BLOCK_SIZE_X =
      COARSEN_FACTOR_2_FLAG_X ? WORK_BLOCK_SIZE / 2 : WORK_BLOCK_SIZE;
  constexpr int THREADING_BLOCK_SIZE_Y =
      COARSEN_FACTOR_2_FLAG_Y ? WORK_BLOCK_SIZE / 2 : WORK_BLOCK_SIZE;
  const int64_t num_relations = (separate_coo_relptrs.numel() - 1);
  const int64_t num_heads = 1;
  const int64_t num_fw_input_dim = weights_transposed.size(2);
  const int64_t num_fw_output_dim = weights_transposed.size(1);
  int64_t num_edges = separate_coo_eids.numel();
  int grid_dim_y =
      std::min(ceil_div<>(num_edges, (int64_t)WORK_BLOCK_SIZE), (int64_t)32768);
  at::Tensor separate_coo_relptrs_cpu_contiguous =
      separate_coo_relptrs.cpu().contiguous();
  std::vector<int> num_blocks_assignment_for_same_relation_vect,
      num_blocks_assignment_for_all_prev_relation_vect;
  std::tie(num_blocks_assignment_for_same_relation_vect,
           num_blocks_assignment_for_all_prev_relation_vect) =
      get_schedule_by_relation_kernel_launch_metadata<false, false, int64_t *>(
          grid_dim_y, num_relations, WORK_BLOCK_SIZE,
          separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>(),
          separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>() +
              num_relations + 1);
  grid_dim_y = num_blocks_assignment_for_all_prev_relation_vect.back();

  thrust::device_vector<int>
      dev_num_blocks_assignment_for_all_prev_relation_vect(
          num_blocks_assignment_for_all_prev_relation_vect.begin(),
          num_blocks_assignment_for_all_prev_relation_vect.end());
  // NB: my shmem sgemm matmul scheme
  const dim3 nblks(ceil_div<>(num_fw_input_dim, (long)WORK_BLOCK_SIZE),
                   grid_dim_y, 1);
  const dim3 nthrs(THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y);

  HET_RGCNMatmulNoScatterGatherListDeltaNodeFeatBckProp<
      COARSEN_FACTOR_2_FLAG_X, COARSEN_FACTOR_2_FLAG_Y, WORK_BLOCK_SIZE,
      int64_t, int64_t *><<<nblks, nthrs, 0, stream>>>(
      delta_node_feat_output.data_ptr<float>(),
      weights_transposed.data_ptr<float>(),
      delta_node_feat_input.data_ptr<float>(), edge_norm.data_ptr<float>(),
      grad_edge_norm.data_ptr<float>(), node_feat_input.data_ptr<float>(),
      separate_coo_row_idx.data_ptr<int64_t>(),
      separate_coo_col_idx.data_ptr<int64_t>(),
      separate_coo_eids.data_ptr<int64_t>(),
      separate_coo_relptrs.data_ptr<int64_t>(),
      thrust::raw_pointer_cast(
          dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
      num_relations, num_fw_output_dim, num_fw_input_dim);

  constexpr int WORK_BLOCK_SIZE_OUTPROD = 32;
  constexpr bool COARSEN_FACTOR_2_FLAG_X_OUTPROD = true;
  constexpr bool COARSEN_FACTOR_2_FLAG_Y_OUTPROD = true;
  constexpr int THREADING_BLOCK_SIZE_X_OUTPROD =
      COARSEN_FACTOR_2_FLAG_X_OUTPROD ? WORK_BLOCK_SIZE_OUTPROD / 2
                                      : WORK_BLOCK_SIZE_OUTPROD;
  constexpr int THREADING_BLOCK_SIZE_Y_OUTPROD =
      COARSEN_FACTOR_2_FLAG_Y_OUTPROD ? WORK_BLOCK_SIZE_OUTPROD / 2
                                      : WORK_BLOCK_SIZE_OUTPROD;
  int grid_dim_y_outprod = std::min(
      ceil_div<>(num_edges, (int64_t)WORK_BLOCK_SIZE_OUTPROD), (int64_t)32768);
  std::vector<int> num_blocks_assignment_for_same_relation_vect_outprod,
      num_blocks_assignment_for_all_prev_relation_vect_outprod;
  std::tie(num_blocks_assignment_for_same_relation_vect_outprod,
           num_blocks_assignment_for_all_prev_relation_vect_outprod) =
      get_schedule_by_relation_kernel_launch_metadata<false, false, int64_t *>(
          grid_dim_y_outprod, num_relations, WORK_BLOCK_SIZE_OUTPROD,
          separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>(),
          separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>() +
              num_relations + 1);
  grid_dim_y_outprod =
      num_blocks_assignment_for_all_prev_relation_vect_outprod.back();

  thrust::device_vector<int>
      dev_num_blocks_assignment_for_all_prev_relation_vect_outprod(
          num_blocks_assignment_for_all_prev_relation_vect_outprod.begin(),
          num_blocks_assignment_for_all_prev_relation_vect_outprod.end());
  const dim3 nblks_outer_product(
      ceil_div<>(num_fw_output_dim, (long)WORK_BLOCK_SIZE_OUTPROD),
      ceil_div<>(num_fw_input_dim, (long)WORK_BLOCK_SIZE_OUTPROD),
      num_heads * grid_dim_y_outprod);
  const dim3 nthrs_outer_product(THREADING_BLOCK_SIZE_X_OUTPROD,
                                 THREADING_BLOCK_SIZE_Y_OUTPROD);
  HET_RGCNMatmulNoScatterGatherListDeltaWeightBckProp<
      COARSEN_FACTOR_2_FLAG_X_OUTPROD, COARSEN_FACTOR_2_FLAG_Y_OUTPROD,
      WORK_BLOCK_SIZE_OUTPROD, int64_t,
      int64_t *><<<nblks_outer_product, nthrs_outer_product, 0, stream>>>(
      node_feat_input.data_ptr<float>(),
      delta_node_feat_output.data_ptr<float>(), delta_weights.data_ptr<float>(),
      edge_norm.data_ptr<float>(), separate_coo_row_idx.data_ptr<int64_t>(),
      separate_coo_col_idx.data_ptr<int64_t>(),
      separate_coo_eids.data_ptr<int64_t>(),
      separate_coo_relptrs.data_ptr<int64_t>(),
      thrust::raw_pointer_cast(
          dev_num_blocks_assignment_for_all_prev_relation_vect_outprod.data()),
      num_relations, num_fw_output_dim, num_fw_input_dim);
}

namespace IntegratedCSR {
// the referential implementation from seastar
template </*int XPU, */ typename Idx, typename DType, bool HybridAssignmentFlag>
void _LayerImpl(
    // GraphRef graph,
    at::Tensor &transposed_csr_rowptr, at::Tensor &transposed_csr_col_idx,
    at::Tensor &transposed_csr_eids, at::Tensor &transposed_csr_reltypes,
    at::Tensor &hidden, at::Tensor &weight, at::Tensor &norm,
    at::Tensor &grad_out, at::Tensor &grad_hidden, at::Tensor &grad_weight,
    at::Tensor &ret, bool layer1_flag, int num_blocks_on_blocks_per_node) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto range_data = transposed_csr_rowptr.data_ptr<Idx>();
  auto ids_data = transposed_csr_col_idx.data_ptr<Idx>();
  auto eids_data = transposed_csr_eids.data_ptr<Idx>();
  auto typeids_data = transposed_csr_reltypes.data_ptr<Idx>();
  DType *hidden_data = hidden.numel() == 0 ? nullptr : hidden.data_ptr<DType>();
  DType *weight_data = weight.numel() == 0 ? nullptr : weight.data_ptr<DType>();
  auto norm_data = norm.data_ptr<DType>();
  auto grad_out_data = grad_out.data_ptr<DType>();
  DType *grad_hidden_data =
      grad_hidden.numel() == 0 ? nullptr : grad_hidden.data_ptr<DType>();
  DType *grad_weight_data =
      grad_weight.numel() == 0 ? nullptr : grad_weight.data_ptr<DType>();
  DType *ret_data = ret.numel() == 0 ? nullptr : ret.data_ptr<DType>();
  Idx num_nodes = transposed_csr_rowptr.numel() - 1;
  Idx num_edges = transposed_csr_col_idx.numel();
  int nblks = num_nodes;
  if (layer1_flag) {
    Idx ntypes = weight.size(0);
    Idx feat_len_y = weight.size(1);
    Idx feat_len_x = weight.size(2);
    int nthrs = feat_len_x < 256 ? 256 : feat_len_x;  // feat_len_y *
                                                      // feat_len_x;
    if constexpr (HybridAssignmentFlag) {
      HET_Seastar_RgcnLayer1BackwardKernelHybridAssignImpl<<<nblks, nthrs, 0,
                                                             stream>>>(
          range_data, ids_data, eids_data, typeids_data, hidden_data,
          weight_data, norm_data, grad_out_data, grad_hidden_data,
          grad_weight_data, num_nodes, feat_len_y, feat_len_x, ntypes,
          num_blocks_on_blocks_per_node);
    } else {
      HET_Seastar_RgcnLayer1BackwardKernelImpl<<<nblks, nthrs, 0, stream>>>(
          range_data, ids_data, eids_data, typeids_data, hidden_data,
          weight_data, norm_data, grad_out_data, grad_hidden_data,
          grad_weight_data, num_nodes, feat_len_y, feat_len_x, ntypes);
    }
  } else {
    Idx ntypes = weight.size(1);
    Idx feat_len = ret.size(2);
    int nthrs = feat_len;
    if constexpr (HybridAssignmentFlag) {
      HET_Seastar_RgcnLayer0BackwardKernelHybridAssignImpl<<<nblks, nthrs, 0,
                                                             stream>>>(
          range_data, ids_data, eids_data, typeids_data, grad_out_data,
          norm_data, ret_data, num_nodes, feat_len, ntypes,
          num_blocks_on_blocks_per_node);
    } else {
      HET_Seastar_RgcnLayer0BackwardKernelImpl<<<nblks, nthrs, 0, stream>>>(
          range_data, ids_data, eids_data, typeids_data, grad_out_data,
          norm_data, ret_data, num_nodes, feat_len, ntypes);
    }
  }
}

// template </*int XPU, */ typename Idx, typename DType>
void Layer0Impl(
    // GraphRef graph,
    at::Tensor &transposed_csr_rowptr, at::Tensor &transposed_csr_col_idx,
    at::Tensor &transposed_csr_eids, at::Tensor &transposed_csr_reltypes,
    at::Tensor &grad_out, at::Tensor &norm, at::Tensor &ret) {
  // NB: graphiler, seastar by default uses int64_t
  at::Tensor dummy_tensor;
  _LayerImpl<int64_t, float, false>(
      transposed_csr_rowptr, transposed_csr_col_idx, transposed_csr_eids,
      transposed_csr_reltypes, /*hidden_dummy*/ dummy_tensor,
      /*weight_dummy*/ dummy_tensor, norm, grad_out,
      /*grad_hidden_dummy*/ ret, /*grad_weight_dummy*/ dummy_tensor, ret, false,
      -1);
}

void Layer0HybridAssignmentImpl(
    // GraphRef graph,
    at::Tensor &transposed_csr_rowptr, at::Tensor &transposed_csr_col_idx,
    at::Tensor &transposed_csr_eids, at::Tensor &transposed_csr_reltypes,
    at::Tensor &grad_out, at::Tensor &norm, at::Tensor &ret,
    int64_t num_blocks_on_blocks_per_node) {
  // NB: graphiler, seastar by default uses int64_t
  at::Tensor dummy_tensor;
  _LayerImpl<int64_t, float, true>(
      transposed_csr_rowptr, transposed_csr_col_idx, transposed_csr_eids,
      transposed_csr_reltypes, /*hidden_dummy*/ dummy_tensor,
      /*weight_dummy*/ dummy_tensor, norm, grad_out,
      /*grad_hidden_dummy*/ dummy_tensor, /*grad_weight_dummy*/ dummy_tensor,
      ret, false, num_blocks_on_blocks_per_node);
}

// template </*int XPU, */ typename Idx, typename DType>
void Layer1Impl(
    // GraphRef graph,
    at::Tensor &transposed_csr_rowptr, at::Tensor &transposed_csr_col_idx,
    at::Tensor &transposed_csr_eids, at::Tensor &transposed_csr_reltypes,
    at::Tensor &hidden, at::Tensor &weight, at::Tensor &norm,
    at::Tensor &grad_out, at::Tensor &grad_hidden, at::Tensor &grad_weight) {
  // NB: graphiler, seastar by default uses int64_t
  at::Tensor dummy_tensor;
  _LayerImpl<int64_t, float, false>(
      transposed_csr_rowptr, transposed_csr_col_idx, transposed_csr_eids,
      transposed_csr_reltypes, hidden, weight, norm, grad_out, grad_hidden,
      grad_weight, /*ret_dummy*/ dummy_tensor, true, -1);
}

void Layer1HybridAssignmentImpl(
    // GraphRef graph,
    at::Tensor &transposed_csr_rowptr, at::Tensor &transposed_csr_col_idx,
    at::Tensor &transposed_csr_eids, at::Tensor &transposed_csr_reltypes,
    at::Tensor &hidden, at::Tensor &weight, at::Tensor &norm,
    at::Tensor &grad_out, at::Tensor &grad_hidden, at::Tensor &grad_weight,
    int64_t num_blocks_on_blocks_per_node) {
  // NB: graphiler, seastar by default uses int64_t
  at::Tensor dummy_tensor;
  _LayerImpl<int64_t, float, true>(
      transposed_csr_rowptr, transposed_csr_col_idx, transposed_csr_eids,
      transposed_csr_reltypes, hidden, weight, norm, grad_out, grad_hidden,
      grad_weight, /*ret_dummy*/ dummy_tensor, true,
      num_blocks_on_blocks_per_node);
}

}  // namespace IntegratedCSR

namespace IntegratedCOO {
// the referential implementation from seastar
template </*int XPU, */ typename Idx, typename DType>
void _LayerBackwardImpl(
    // GraphRef graph,
    at::Tensor &transposed_coo_row_idx, at::Tensor &transposed_coo_col_idx,
    at::Tensor &transposed_coo_eids, at::Tensor &transposed_coo_reltypes,
    at::Tensor &hidden, at::Tensor &weight, at::Tensor &norm,
    at::Tensor &grad_out, at::Tensor &grad_hidden, at::Tensor &grad_weight,
    at::Tensor &ret, bool layer1_flag) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto row_idx_data = transposed_coo_row_idx.data_ptr<Idx>();
  auto ids_data = transposed_coo_col_idx.data_ptr<Idx>();
  auto eids_data = transposed_coo_eids.data_ptr<Idx>();
  auto typeids_data = transposed_coo_reltypes.data_ptr<Idx>();
  auto hidden_data = hidden.data_ptr<DType>();
  auto weight_data = weight.data_ptr<DType>();
  auto norm_data = norm.data_ptr<DType>();
  auto grad_out_data = grad_out.data_ptr<DType>();
  auto grad_hidden_data = grad_hidden.data_ptr<DType>();
  auto grad_weight_data = grad_weight.data_ptr<DType>();
  DType *ret_data = ret.numel() == 0 ? nullptr : ret.data_ptr<DType>();
  Idx num_edges = transposed_coo_col_idx.numel();
  if (layer1_flag) {
    Idx ntypes = weight.size(0);
    Idx feat_len_y = weight.size(1);
    Idx feat_len_x = weight.size(2);
    // int nthrs = feat_len_y * feat_len_x;
    int nthrs = feat_len_x < 256 ? 256 : feat_len_x;
    assert(nthrs % 32 == 0);
    int nblks =
        ceil_div<>(num_edges, (int64_t)nthrs / 32);  // 32 is the warp size
    HET_Seastar_RgcnLayer1BackwardCOOKernelImpl<<<nblks, nthrs, 0, stream>>>(
        row_idx_data, ids_data, eids_data, typeids_data, hidden_data,
        weight_data, norm_data, grad_out_data, grad_hidden_data,
        grad_weight_data, num_edges, feat_len_y, feat_len_x, ntypes);
  } else {
    Idx ntypes = weight.size(1);
    Idx feat_len = ret.size(2);
    int nthrs = feat_len;
    assert(0 && "not implemented");
    // HET_Seastar_RgcnLayer0BackwardKernelImpl<<<nblks, nthrs>>>(
    //    range_data, ids_data, eids_data, typeids_data, grad_out_data,
    //    norm_data, ret_data, num_nodes, feat_len, ntypes);
  }
}

// template </*int XPU, */ typename Idx, typename DType>
void Layer1BackwardImpl(
    // GraphRef graph,
    at::Tensor &transposed_coo_row_idx, at::Tensor &transposed_coo_col_idx,
    at::Tensor &transposed_coo_eids, at::Tensor &transposed_coo_reltypes,
    at::Tensor &hidden, at::Tensor &weight, at::Tensor &norm,
    at::Tensor &grad_out, at::Tensor &grad_hidden, at::Tensor &grad_weight) {
  // NB: graphiler, seastar by default uses int64_t
  at::Tensor dummy_tensor;
  _LayerBackwardImpl<int64_t, float>(
      transposed_coo_row_idx, transposed_coo_col_idx, transposed_coo_eids,
      transposed_coo_reltypes, hidden, weight, norm, grad_out, grad_hidden,
      grad_weight, /*ret_dummy*/ dummy_tensor, true);
}
}  // namespace IntegratedCOO
}  // namespace BckProp
}  // namespace RGCN
}  // namespace TorchExport
}  // namespace HET

using namespace HET::TorchExport;
TORCH_LIBRARY_FRAGMENT(torch_hetero_edgesoftmax, m) {
  // RGCN edge parallel mean aggregation declaration
  m.def("backward_rgcn_node_mean_aggregation_compact_as_of_node_separate_coo",
        RGCN::BckProp::Layer1_NodeMeanAggregation_CompactAsOfNode_SeparateCOO<
            int64_t, float>);
  m.def("rgcn_node_mean_aggregation_compact_as_of_node_separate_coo",
        RGCN::FwProp::Layer1_NodeMeanAggregation_CompactAsOfNode_SeparateCOO<
            int64_t, float>);
  // RGCN separate coo (edge parallel) declaration
  m.def("backward_rgcn_layer1_separate_coo", RGCN::BckProp::Layer1_SeparateCOO);
  m.def("rgcn_layer1_separate_coo", RGCN::FwProp::Layer1_SeparateCOO);

  // RGCN CSR Declaration
  m.def("seastar_rgcn_layer0_csr", RGCN::FwProp::IntegratedCSR::Layer0Impl);
  m.def("seastar_backward_rgcn_layer0_csr",
        RGCN::BckProp::IntegratedCSR::Layer0Impl);
  m.def("seastar_rgcn_layer1_csr", RGCN::FwProp::IntegratedCSR::Layer1Impl);
  m.def("seastar_backward_rgcn_layer1_csr",
        RGCN::BckProp::IntegratedCSR::Layer1Impl);
  // FIXME: hybrid assign layer 0 is unused. Apply it in python backend
  // submodule
  m.def("seastar_rgcn_layer0_csr_hybrid_assign",
        RGCN::FwProp::IntegratedCSR::Layer0HybridAssignmentImpl);
  m.def("seastar_backward_rgcn_layer0_csr_hybrid_assign",
        RGCN::BckProp::IntegratedCSR::Layer0HybridAssignmentImpl);
  m.def("seastar_rgcn_layer1_csr_hybrid_assign",
        RGCN::FwProp::IntegratedCSR::Layer1HybridAssignmentImpl);
  m.def("seastar_backward_rgcn_layer1_csr_hybrid_assign",
        RGCN::BckProp::IntegratedCSR::Layer1HybridAssignmentImpl);
  // RGCN COO Declaration
  m.def("seastar_rgcn_layer1_coo", RGCN::FwProp::IntegratedCOO::Layer1Impl);
  m.def("seastar_backward_rgcn_layer1_coo",
        RGCN::BckProp::IntegratedCOO::Layer1BackwardImpl);
}
