#pragma once
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "DGLHackKernel/DGLHackUtils.h"
#include "DGLHackKernel/RGNN/inner_product.cu.h"
#include "DGLHackKernel/RGNN/inner_product_edge_parallel.cu.h"
#include "DGLHackKernel/RGNN/my_shmem_sgemm_func.cu.h"
#include "DGLHackKernel/RGNN/mysgemm_KernelsBlockConfigurations.h"
#include "ThreadingGridsBlocksSchedules.h"

namespace HET {
namespace TorchExport {
namespace RGNN {
namespace FwProp {

// NB: KWU: use reg tiling here: test fuse attn score vs non-fused
// TODO: remove the unused SingleSidedCompactAsOfNodeFlag and its logic
template <bool CompactAsOfNodeFlag, bool ACGatherScatterListIdenticalFlag,
          bool InputNumHeadOneFlag>
void _RelationalMatMul_separatecoo(
    at::Tensor &separate_coo_relptrs, at::Tensor &separate_coo_node_indices,
    at::Tensor &separate_coo_eids, at::Tensor &unique_srcs_and_dests_rel_ptr,
    at::Tensor &unique_srcs_and_dests_node_indices, at::Tensor &weights,
    at::Tensor &node_feat, at::Tensor &ret) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int64_t num_relations =
      separate_coo_relptrs.numel() == 0
          ? (unique_srcs_and_dests_rel_ptr.numel() - 1)
          : (separate_coo_relptrs.numel() - 1);
  const int64_t num_heads = weights.size(1);
  const int64_t num_input_dim = weights.size(2);
  const int64_t num_output_per_head_dim =
      weights.size(3);  // weight shape (num_relations, n_heads,
                        // in_feat, out_feat // n_heads)
  int64_t num_edges;

  // TODO: KWU: add reg-tiled speicifc configurations by introducing tenary
  // operators

  // NB: configuration specific to shmem-tiled sgemm

  // assuming coarsening in both x and y direction if shmem is used instead of
  // reg tiling
  constexpr bool REG_TILING_FLAG = true;

  constexpr int WORK_BLOCK_SIZE_X = REG_TILING_FLAG ? 64 : 32;
  constexpr int WORK_BLOCK_SIZE_Y = REG_TILING_FLAG ? 16 : 32;
  constexpr int WORK_BLOCK_SIZE_K = REG_TILING_FLAG ? 16 : 32;
  constexpr int THREADING_BLOCK_SIZE_X =
      REG_TILING_FLAG ? WORK_BLOCK_SIZE_X : WORK_BLOCK_SIZE_X / 2;
  constexpr int THREADING_BLOCK_SIZE_Y =
      REG_TILING_FLAG ? 1 : WORK_BLOCK_SIZE_Y / 2;

  if constexpr (CompactAsOfNodeFlag) {
    num_edges = unique_srcs_and_dests_node_indices.numel();
  } else {
    num_edges = separate_coo_eids.numel();
  }
  int grid_dim_y = std::min(
      ceil_div<>(num_edges, (int64_t)WORK_BLOCK_SIZE_Y),
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
        get_schedule_by_relation_kernel_launch_metadata<false, false,
                                                        int64_t *>(
            grid_dim_y, num_relations, WORK_BLOCK_SIZE_Y,
            unique_srcs_and_dests_rel_ptr_cpu_contiguous.data_ptr<int64_t>(),
            unique_srcs_and_dests_rel_ptr_cpu_contiguous.data_ptr<int64_t>() +
                num_relations + 1);
  } else {
    at::Tensor separate_coo_relptrs_cpu_contiguous =
        separate_coo_relptrs.cpu().contiguous();
    std::tie(num_blocks_assignment_for_same_relation_vect,
             num_blocks_assignment_for_all_prev_relation_vect) =
        get_schedule_by_relation_kernel_launch_metadata<false, false,
                                                        int64_t *>(
            grid_dim_y, num_relations, WORK_BLOCK_SIZE_Y,
            separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>(),
            separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>() +
                num_relations + 1);
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
    if constexpr (REG_TILING_FLAG) {
      // not implemented yet
      assert(0 && "not implemented yet");
    }
    if constexpr (ACGatherScatterListIdenticalFlag) {
      CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
          CompactAsOfNodeFlag && ACGatherScatterListIdenticalFlag,
          "CompactAsOfNodeFlag && ACGatherScatterListIdenticalFlag");
    }

    // NB: my shmem sgemm matmul scheme
    const dim3 nblks(
        ceil_div<>(num_output_per_head_dim, (long)WORK_BLOCK_SIZE_X),
        grid_dim_y, num_heads);
    const dim3 nthrs(THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y);
    // TODO: KWU: allow more dtype options in this file
    HET_RGNNFeatCompactFWProp<THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y,
                              WORK_BLOCK_SIZE_X, WORK_BLOCK_SIZE_Y,
                              WORK_BLOCK_SIZE_K, int64_t, int64_t *,
                              InputNumHeadOneFlag><<<nblks, nthrs, 0, stream>>>(
        node_feat.data_ptr<float>(), weights.data_ptr<float>(),
        ret.data_ptr<float>(),
        unique_srcs_and_dests_rel_ptr.data_ptr<int64_t>(),
        unique_srcs_and_dests_node_indices.data_ptr<int64_t>(), num_input_dim,
        num_output_per_head_dim, num_heads,
        thrust::raw_pointer_cast(
            dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
        num_relations);
  } else {
    // NB: my shmem sgemm matmul scheme
    const dim3 nblks(
        ceil_div<>(num_output_per_head_dim, (long)WORK_BLOCK_SIZE_X),
        grid_dim_y, num_heads);
    const dim3 nthrs(THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y);
    if constexpr (ACGatherScatterListIdenticalFlag) {
      HET_RGNNFeatPerEdgeFWPropACGatherScatterListIdentical<
          THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_X,
          WORK_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_K, int64_t, int64_t *,
          InputNumHeadOneFlag><<<nblks, nthrs, 0, stream>>>(
          node_feat.data_ptr<float>(), weights.data_ptr<float>(),
          ret.data_ptr<float>(), separate_coo_relptrs.data_ptr<int64_t>(),
          separate_coo_eids.data_ptr<int64_t>(), num_input_dim,
          num_output_per_head_dim, num_heads,
          thrust::raw_pointer_cast(
              dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
          num_relations);
    } else {
      // NB: KWU: use by default the new reg tiled version here
      HET_RGNNFeatPerEdgeFWProp<THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y,
                                WORK_BLOCK_SIZE_X, WORK_BLOCK_SIZE_Y,
                                WORK_BLOCK_SIZE_K, int64_t, int64_t *,
                                InputNumHeadOneFlag>
          <<<nblks, nthrs, 0, stream>>>(
              node_feat.data_ptr<float>(), weights.data_ptr<float>(),
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

template <bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
          int WORK_BLOCK_SIZE>
void _RelationalMatmulNoScatterGatherList(at::Tensor &ntype_offset_ptrs,
                                          at::Tensor &weights,
                                          at::Tensor &inputs, at::Tensor &ret) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  assert(weights.size(1) == 1 && "assertion n_head == 1 failed");
  const int64_t num_input_dim = weights.size(2);
  const int64_t num_output_dim = weights.size(3);  // weight shape (num_ntypes,
                                                   // in_feat, out_feat)
  int64_t num_ntypes = ntype_offset_ptrs.numel() - 1;
  int64_t num_nodes = inputs.size(0);

  constexpr int THREADING_BLOCK_SIZE_X =
      COARSEN_FACTOR_2_FLAG_X ? WORK_BLOCK_SIZE / 2 : WORK_BLOCK_SIZE;
  constexpr int THREADING_BLOCK_SIZE_Y =
      COARSEN_FACTOR_2_FLAG_Y ? WORK_BLOCK_SIZE / 2 : WORK_BLOCK_SIZE;

  int grid_dim_y = std::min(
      ceil_div<>(num_nodes, (int64_t)WORK_BLOCK_SIZE),
      (int64_t)32768);  // using 32768 instead of 65535 to leave some space in
                        // case the total number of blocks is slightly larger
                        // due to relationship with very few workloads
  std::vector<int> num_blocks_assignment_for_same_ntype_vect,
      num_blocks_assignment_for_all_prev_ntype_vect;

  at::Tensor ntype_offset_ptrs_cpu_contiguous =
      ntype_offset_ptrs.cpu().contiguous();
  std::tie(num_blocks_assignment_for_same_ntype_vect,
           num_blocks_assignment_for_all_prev_ntype_vect) =
      get_schedule_by_relation_kernel_launch_metadata<false, false, int64_t *>(
          grid_dim_y, num_ntypes, WORK_BLOCK_SIZE,
          ntype_offset_ptrs_cpu_contiguous.data_ptr<int64_t>(),
          ntype_offset_ptrs_cpu_contiguous.data_ptr<int64_t>() + num_ntypes +
              1);

  grid_dim_y = num_blocks_assignment_for_all_prev_ntype_vect.back();

  thrust::device_vector<int> dev_num_blocks_assignment_for_same_ntype_vect(
      num_blocks_assignment_for_same_ntype_vect.begin(),
      num_blocks_assignment_for_same_ntype_vect.end());
  thrust::device_vector<int> dev_num_blocks_assignment_for_all_prev_ntype_vect(
      num_blocks_assignment_for_all_prev_ntype_vect.begin(),
      num_blocks_assignment_for_all_prev_ntype_vect.end());
  // NB: my shmem sgemm matmul scheme
  // in NoScatterScatter scenario, there is no such thing as multi-headed
  const dim3 nblks(ceil_div<>(num_output_dim, (long)WORK_BLOCK_SIZE),
                   grid_dim_y, 1);
  const dim3 nthrs(THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y);
  HET_RGNNMatmulNoScatterGatherListFwOrBwProp<
      COARSEN_FACTOR_2_FLAG_X, COARSEN_FACTOR_2_FLAG_Y, WORK_BLOCK_SIZE,
      int64_t, int64_t *><<<nblks, nthrs, 0, stream>>>(
      inputs.data_ptr<float>(), weights.data_ptr<float>(),
      ret.data_ptr<float>(), ntype_offset_ptrs.data_ptr<int64_t>(),

      thrust::raw_pointer_cast(
          dev_num_blocks_assignment_for_all_prev_ntype_vect.data()),
      num_ntypes, num_input_dim, num_output_dim);
}

void RelationalMatmulNoScatterGatherList(at::Tensor &ntype_offset_ptrs,
                                         at::Tensor &weights,
                                         at::Tensor &inputs, at::Tensor &ret) {
  _RelationalMatmulNoScatterGatherList<true, true, 32>(ntype_offset_ptrs,
                                                       weights, inputs, ret);
}

void RelationalMatMul_separatecoo(at::Tensor &separate_coo_relptrs,
                                  at::Tensor &separate_coo_node_indices,
                                  at::Tensor &separate_coo_eids,
                                  at::Tensor &weights, at::Tensor &node_feat,
                                  at::Tensor &ret, bool InputNumHeadOneFlag) {
  at::Tensor dummy_tensor;
  // TODO: KWU: simplify the PyThon API so that we don't need to have two APIs
  // for num_heads==1 vs. num_heads>1 if separate_coo_node_indices and
  // separate_coo_eids are the same tenssor
  if (separate_coo_node_indices.data_ptr() == separate_coo_eids.data_ptr()) {
    if (InputNumHeadOneFlag) {
      _RelationalMatMul_separatecoo<false, true, true>(
          separate_coo_relptrs, dummy_tensor, separate_coo_eids, dummy_tensor,
          dummy_tensor, weights, node_feat, ret);
    } else {
      _RelationalMatMul_separatecoo<false, true, false>(
          separate_coo_relptrs, dummy_tensor, separate_coo_eids, dummy_tensor,
          dummy_tensor, weights, node_feat, ret);
    }
  } else {
    if (InputNumHeadOneFlag) {
      _RelationalMatMul_separatecoo<false, false, true>(
          separate_coo_relptrs, separate_coo_node_indices, separate_coo_eids,
          dummy_tensor, dummy_tensor, weights, node_feat, ret);
    } else {
      _RelationalMatMul_separatecoo<false, false, false>(
          separate_coo_relptrs, separate_coo_node_indices, separate_coo_eids,
          dummy_tensor, dummy_tensor, weights, node_feat, ret);
    }
  }
}

// void RelationalMatMul_ACGatherScatterListIdentical_separatecoo(
//     at::Tensor &separate_coo_relptrs, at::Tensor &separate_coo_eids,
//     at::Tensor &weights, at::Tensor &node_feat, at::Tensor &ret,
//     bool InputNumHeadOneFlag) {
//   at::Tensor dummy_tensor;
//   assert(0 && "deprecated");
//   if (InputNumHeadOneFlag) {
//     _RelationalMatMul_separatecoo<false, true, true>(
//         separate_coo_relptrs, dummy_tensor, separate_coo_eids, dummy_tensor,
//         dummy_tensor, weights, node_feat, ret);
//   } else {
//     _RelationalMatMul_separatecoo<false, true, false>(
//         separate_coo_relptrs, dummy_tensor, separate_coo_eids, dummy_tensor,
//         dummy_tensor, weights, node_feat, ret);
//   }
// }

void RelationalMatMulCompactAsOfNode_unique_rel_node_indices(
    at::Tensor &unique_srcs_and_dests_rel_ptr,
    at::Tensor &unique_srcs_and_dests_node_indices, at::Tensor &weight,
    at::Tensor &node_feat, at::Tensor &ret, bool InputNumHeadOneFlag) {
  at::Tensor dummy_tensor;
  if (InputNumHeadOneFlag) {
    _RelationalMatMul_separatecoo<true, false, true>(
        dummy_tensor, dummy_tensor, dummy_tensor, unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_indices, weight, node_feat, ret);
  } else {
    _RelationalMatMul_separatecoo<true, false, false>(
        dummy_tensor, dummy_tensor, dummy_tensor, unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_indices, weight, node_feat, ret);
  }
}

// void RelationalMatMulCompactAsOfNodeSingleEnded_unique_rel_node_indices(
//     at::Tensor& unique_srcs_and_dests_rel_ptr,
//     at::Tensor& unique_srcs_and_dests_node_indices,
//     at::Tensor& separate_coo_rel_ptr, at::Tensor& separate_coo_node_indices,
//     at::Tensor& separate_coo_eids, at::Tensor& weight, at::Tensor& node_feat,
//     at::Tensor& ret, bool InputNumHeadOneFlag) {
//   at::Tensor dummy_tensor;
//   if (InputNumHeadOneFlag) {
//     _RelationalMatMul_separatecoo<true, true, 32, true, true, false, true>(
//         separate_coo_rel_ptr, separate_coo_node_indices, separate_coo_eids,
//         unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
//         weight, node_feat, ret);
//   } else {
//     _RelationalMatMul_separatecoo<true, true, 32, true, true, false, false>(
//         separate_coo_rel_ptr, separate_coo_node_indices, separate_coo_eids,
//         unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
//         weight, node_feat, ret);
//   }
// }

// NB: We may refer to (edge parallel version)
// HET_HGTExpermentalEdgeAttentionConcatenatedSecondStageSrcInnerProductDestIntemediateCOOKernel
// in hetero_edgesoftmax/include/EdgeSoftmax_4/EdgeSoftmaxMultiCOOs_4.h and
// (node parallel version) _gatExpLeakyReluSumKernel in
// [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGAT.cu.h]]
// adapted from _RelationalFusedGATKernel in
// hetero_edgesoftmax/include/DGLHackKernel/OpExport/RGATOps.inc.h
template </*int XPU, */ typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool IntegratedFormatRatherThanSeparateFlag,
          bool CSRRatherThanCOOFlag>
void inner_product_various_left_and_node_right(
    at::Tensor &separate_coo_eids, at::Tensor &separate_coo_rel_ptrs,
    at::Tensor &separate_coo_row_indices, at::Tensor &separate_coo_col_indices,
    at::Tensor &incsr_row_ptr, at::Tensor &incsr_col_idx,
    at::Tensor &incsr_eids, at::Tensor &incsr_reltypes,
    at::Tensor &unique_srcs_and_dests_rel_ptr,
    at::Tensor &unique_srcs_and_dests_node_indices, at::Tensor &feat_src,
    at::Tensor &feat_dst, at::Tensor &edge_inner_product) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  // NB: in this case gdata.n, calculation is removed since el is now per edge
  // rather than per node
  InnerProductData<Idx, DType> gdata{
      .feat_src_xlen = SeastarComputeXLength<>(feat_src),
      .num_heads = SeastarComputeXLength<>(edge_inner_product),
      .eids = nullptr,  // assign later in if branches
      .feat_src = feat_src.data_ptr<DType>(),
      .feat_dst = feat_dst.data_ptr<DType>(),
      .edge_inner_product = edge_inner_product.data_ptr<DType>()};

  if constexpr (IntegratedFormatRatherThanSeparateFlag &&
                CSRRatherThanCOOFlag) {
    // Integrated CSR
    gdata.eids = incsr_eids.data_ptr<Idx>();
    // Configure kernel launch parameters.

    // NB: updated to Type 2 Schedule:
    // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
    // head -> threadIdx.y
    // node -> blockIdx.y
    // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x

    int64_t incsr_num_rows = incsr_row_ptr.numel() - 1;
    auto [nblks2, nthrs2] = get_type2_schedule(
        gdata.num_heads, gdata.feat_src_xlen, incsr_num_rows);

    Idx *incsr_row_ptr_data_ptr =
        incsr_row_ptr.numel() > 0 ? incsr_row_ptr.data_ptr<Idx>() : nullptr;
    Idx *incsr_col_idx_data_ptr =
        incsr_col_idx.numel() > 0 ? incsr_col_idx.data_ptr<Idx>() : nullptr;
    Idx *incsr_reltypes_data_ptr =
        incsr_reltypes.numel() > 0 ? incsr_reltypes.data_ptr<Idx>() : nullptr;
    Idx *unique_srcs_and_dests_rel_ptr_data_ptr =
        unique_srcs_and_dests_rel_ptr.numel() > 0
            ? unique_srcs_and_dests_rel_ptr.data_ptr<Idx>()
            : nullptr;
    Idx *unique_srcs_and_dests_node_indices_data_ptr =
        unique_srcs_and_dests_node_indices.numel() > 0
            ? unique_srcs_and_dests_node_indices.data_ptr<Idx>()
            : nullptr;

    HET_inner_product_fw_kernel<Idx, DType, CompactAsOfNodeFlag, true, false,
                                false><<<nblks2, nthrs2, 0, stream>>>(
        gdata, incsr_row_ptr_data_ptr, incsr_col_idx_data_ptr,
        incsr_reltypes_data_ptr, incsr_num_rows,
        unique_srcs_and_dests_rel_ptr_data_ptr,
        unique_srcs_and_dests_node_indices_data_ptr);
  } else if constexpr (!IntegratedFormatRatherThanSeparateFlag &&
                       !CSRRatherThanCOOFlag) {
    // separate coo
    gdata.eids = separate_coo_eids.data_ptr<Idx>();
    int64_t num_edges = separate_coo_row_indices.numel();
    int64_t num_relations = separate_coo_rel_ptrs.numel() - 1;

    // NB: Type 2 Schedule:
    // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
    // head -> threadIdx.y
    // edge -> blockIdx.y
    // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x
    // threadIdx.x and threadIdx.y and only this pair is exchanged compared with
    // original seastar schedule to allow reduction within the warp, i.e., along
    // x-axis
    auto [nblks_inner_product, nthrs_inner_product] =
        get_type2_schedule(gdata.num_heads, gdata.feat_src_xlen, num_edges);
    Idx *separate_coo_row_indices_data_ptr =
        separate_coo_row_indices.numel() > 0
            ? separate_coo_row_indices.data_ptr<Idx>()
            : nullptr;
    Idx *separate_coo_col_indices_data_ptr =
        separate_coo_col_indices.numel() > 0
            ? separate_coo_col_indices.data_ptr<Idx>()
            : nullptr;
    Idx *separate_coo_rel_ptrs_data_ptr =
        separate_coo_rel_ptrs.numel() > 0
            ? separate_coo_rel_ptrs.data_ptr<Idx>()
            : nullptr;
    Idx *unique_srcs_and_dests_rel_ptr_data_ptr =
        unique_srcs_and_dests_rel_ptr.numel() > 0
            ? unique_srcs_and_dests_rel_ptr.data_ptr<Idx>()
            : nullptr;
    Idx *unique_srcs_and_dests_node_indices_data_ptr =
        unique_srcs_and_dests_node_indices.numel() > 0
            ? unique_srcs_and_dests_node_indices.data_ptr<Idx>()
            : nullptr;
    if (gdata.feat_src_xlen / gdata.num_heads >= 32) {
      HET_inner_product_fw_kernel_edge_parallel<Idx, DType, CompactAsOfNodeFlag,
                                                true, true, false, true>
          <<<nblks_inner_product, nthrs_inner_product, 0, stream>>>(
              gdata, separate_coo_row_indices_data_ptr,
              separate_coo_col_indices_data_ptr, separate_coo_rel_ptrs_data_ptr,
              num_edges, unique_srcs_and_dests_rel_ptr_data_ptr,
              unique_srcs_and_dests_node_indices_data_ptr, num_relations);
    } else {
      assert(0 && "Not implemented");
      // HET_inner_product_fw_kernel_edge_parallel<Idx, DType,
      // CompactAsOfNodeFlag,
      //                                       true, true, false, false>
      // <<<nblks_inner_product, nthrs_inner_product, 0, stream>>>(
      //     gdata, separate_coo_row_indices_data_ptr,
      //     separate_coo_col_indices_data_ptr, separate_coo_rel_ptrs_data_ptr,
      //     num_edges, unique_srcs_and_dests_rel_ptr_data_ptr,
      //     unique_srcs_and_dests_node_indices_data_ptr, num_relations);
    }
  } else {
    assert(0 && "Not implemented");
  }
}

void inner_product_node_compact_and_node_separatecoo(
    at::Tensor &unique_srcs_and_dests_rel_ptr,
    at::Tensor &unique_srcs_and_dests_node_idx,
    at::Tensor &separate_coo_rel_ptr, at::Tensor &separate_coo_eids,
    at::Tensor &separate_coo_row_indices, at::Tensor &separate_coo_col_indices,
    at::Tensor &left_node_compact_data, at::Tensor &right_node_vectors,
    at::Tensor &edge_inner_product) {
  cudaMemsetAsync(edge_inner_product.data_ptr<float>(), 0,
                  edge_inner_product.numel() * sizeof(float),
                  c10::cuda::getCurrentCUDAStream());
  at::Tensor dummy_tensor;
  inner_product_various_left_and_node_right<int64_t, float, true, false, false>(
      separate_coo_eids, separate_coo_rel_ptr, separate_coo_row_indices,
      separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
      dummy_tensor, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_idx, left_node_compact_data,
      right_node_vectors, edge_inner_product);
}

void inner_product_edge_and_node_separatecoo(
    at::Tensor &separate_coo_eids, at::Tensor &separate_coo_row_indices,
    at::Tensor &separate_coo_col_indices, at::Tensor &left_edge_data,
    at::Tensor &right_node_vectors, at::Tensor &edge_inner_product) {
  cudaMemsetAsync(edge_inner_product.data_ptr<float>(), 0,
                  edge_inner_product.numel() * sizeof(float),
                  c10::cuda::getCurrentCUDAStream());
  at::Tensor dummy_tensor;
  inner_product_various_left_and_node_right<int64_t, float, false, false,
                                            false>(
      separate_coo_eids, dummy_tensor, separate_coo_row_indices,
      separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
      dummy_tensor, dummy_tensor, dummy_tensor, left_edge_data,
      right_node_vectors, edge_inner_product);
}

}  // namespace FwProp

namespace BckProp {
template <bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
          int WORK_BLOCK_SIZE, bool CompactAsOfNodeFlag,
          bool ACGatherScatterListIdenticalFlag, bool InputNumHeadOneFlag>
void _BackwardRelationalMatMul_separatecoo(
    at::Tensor &separate_coo_relptrs, at::Tensor &separate_coo_node_indices,
    at::Tensor &separate_coo_eids, at::Tensor &unique_srcs_and_dests_rel_ptr,
    at::Tensor &unique_srcs_and_dests_node_indices,
    at::Tensor &weights_transposed, at::Tensor &node_feat, at::Tensor &gradout,
    at::Tensor &grad_node_feat, at::Tensor &grad_weights) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int64_t num_relations =
      separate_coo_relptrs.numel() == 0
          ? (unique_srcs_and_dests_rel_ptr.numel() - 1)
          : (separate_coo_relptrs.numel() - 1);
  const int64_t num_heads = weights_transposed.size(1);
  const int64_t num_input_dim = weights_transposed.size(3);
  const int64_t num_output_per_head_dim =
      weights_transposed.size(2);  // weight shape (num_relations, n_heads,
                                   // in_feat, out_feat // n_heads)

  constexpr int THREADING_BLOCK_SIZE_X =
      COARSEN_FACTOR_2_FLAG_X ? WORK_BLOCK_SIZE / 2 : WORK_BLOCK_SIZE;
  constexpr int THREADING_BLOCK_SIZE_Y =
      COARSEN_FACTOR_2_FLAG_Y ? WORK_BLOCK_SIZE / 2 : WORK_BLOCK_SIZE;

  int64_t num_edges;
  assert(weights_transposed.is_contiguous());
  assert(node_feat.is_contiguous());
  assert(gradout.is_contiguous());
  assert(grad_node_feat.is_contiguous());
  assert(grad_weights.is_contiguous());

  if constexpr (CompactAsOfNodeFlag) {
    num_edges = unique_srcs_and_dests_node_indices.numel();
  } else {
    num_edges = separate_coo_eids.numel();
  }
  int grid_dim_y = std::min(
      ceil_div<>(num_edges, (int64_t)WORK_BLOCK_SIZE),
      (int64_t)4096);  // using 32768 instead of 65535 to leave some space in
                       // case the total number of blocks is slightly larger
                       // due to relationship with very few workloads
  std::vector<int> num_blocks_assignment_for_same_relation_vect,
      num_blocks_assignment_for_all_prev_relation_vect;
  if constexpr (CompactAsOfNodeFlag) {
    at::Tensor unique_srcs_and_dests_rel_ptr_cpu_contiguous =
        unique_srcs_and_dests_rel_ptr.cpu().contiguous();
    std::tie(num_blocks_assignment_for_same_relation_vect,
             num_blocks_assignment_for_all_prev_relation_vect) =
        get_schedule_by_relation_kernel_launch_metadata<false, false,
                                                        int64_t *>(
            grid_dim_y, num_relations, WORK_BLOCK_SIZE,
            unique_srcs_and_dests_rel_ptr_cpu_contiguous.data_ptr<int64_t>(),
            unique_srcs_and_dests_rel_ptr_cpu_contiguous.data_ptr<int64_t>() +
                num_relations + 1);
  } else {
    at::Tensor separate_coo_relptrs_cpu_contiguous =
        separate_coo_relptrs.cpu().contiguous();
    std::tie(num_blocks_assignment_for_same_relation_vect,
             num_blocks_assignment_for_all_prev_relation_vect) =
        get_schedule_by_relation_kernel_launch_metadata<false, false,
                                                        int64_t *>(
            grid_dim_y, num_relations, WORK_BLOCK_SIZE,
            separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>(),
            separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>() +
                num_relations + 1);
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
    // NB: my shmem sgemm matmul scheme
    const dim3 nblks(ceil_div<>(num_input_dim, (long)WORK_BLOCK_SIZE),
                     grid_dim_y, num_heads);
    const dim3 nblks_outer_product(
        ceil_div<>(num_output_per_head_dim, (long)WORK_BLOCK_SIZE),
        ceil_div<>(num_input_dim, (long)WORK_BLOCK_SIZE),
        num_heads * grid_dim_y);
    assert(num_heads * grid_dim_y < 65535 && "num_head*grid_dim_y>=65535");

    const dim3 nthrs(THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y);
    // NB: #head of node_feat is 1 when InputNumHeadOneFlag is true
    // cuda_err_chk(cudaGetLastError());
    std::cout << gradout.numel() << std::endl;
    HET_RGNNDeltaNodeFeatInputCompactBWProp<
        COARSEN_FACTOR_2_FLAG_X, COARSEN_FACTOR_2_FLAG_Y, WORK_BLOCK_SIZE,
        int64_t, int64_t *, InputNumHeadOneFlag><<<nblks, nthrs, 0, stream>>>(
        gradout.data_ptr<float>(), weights_transposed.data_ptr<float>(),
        grad_node_feat.data_ptr<float>(),
        unique_srcs_and_dests_rel_ptr.data_ptr<int64_t>(),
        unique_srcs_and_dests_node_indices.data_ptr<int64_t>(), num_edges,
        num_output_per_head_dim, num_input_dim, num_heads,
        thrust::raw_pointer_cast(
            dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
        num_relations);
    // cuda_err_chk(cudaGetLastError());
    HET_RGNNDeltaWeightCompactBWProp<COARSEN_FACTOR_2_FLAG_X,
                                     COARSEN_FACTOR_2_FLAG_Y, WORK_BLOCK_SIZE,
                                     int64_t, int64_t *, InputNumHeadOneFlag>
        <<<nblks_outer_product, nthrs, 0, stream>>>(
            gradout.data_ptr<float>(), node_feat.data_ptr<float>(),
            grad_weights.data_ptr<float>(),
            unique_srcs_and_dests_rel_ptr.data_ptr<int64_t>(),
            unique_srcs_and_dests_node_indices.data_ptr<int64_t>(), num_edges,
            num_input_dim, num_output_per_head_dim, num_heads,
            thrust::raw_pointer_cast(
                dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
            num_relations);
    // cuda_err_chk(cudaGetLastError());
  } else {
    // NB: my shmem sgemm matmul scheme
    const dim3 nblks(ceil_div<>(num_input_dim, (long)WORK_BLOCK_SIZE),
                     grid_dim_y, num_heads);
    const dim3 nblks_outer_product(
        ceil_div<>(num_output_per_head_dim, (long)WORK_BLOCK_SIZE),
        ceil_div<>(num_input_dim, (long)WORK_BLOCK_SIZE),
        num_heads * grid_dim_y);
    assert(num_heads * grid_dim_y < 65535 && "num_head*grid_dim_y>=65535");
    const dim3 nthrs(THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y);
    // NB: #head of node_feat is 1 when InputNumHeadOneFlag is true
    if constexpr (ACGatherScatterListIdenticalFlag) {
      HET_RGNNDeltaNodeFeatInputBWPropACGatherScatterListIdentical<
          COARSEN_FACTOR_2_FLAG_X, COARSEN_FACTOR_2_FLAG_Y, WORK_BLOCK_SIZE,
          int64_t, int64_t *, InputNumHeadOneFlag><<<nblks, nthrs, 0, stream>>>(
          gradout.data_ptr<float>(), weights_transposed.data_ptr<float>(),
          grad_node_feat.data_ptr<float>(),
          separate_coo_eids.data_ptr<int64_t>(),
          separate_coo_relptrs.data_ptr<int64_t>(), num_output_per_head_dim,
          num_input_dim, num_heads,
          thrust::raw_pointer_cast(
              dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
          num_relations);
      // NB: NumHeadOneFlag addresses the case where num_heads == 1. in
      // deltaweight case, InputNumHeadOneFlag is true for RGAT and false for
      // HGT, and the delta weight is calculated accordingly. The original grid
      // configuration scheme dependent on weight dimensions should  still work
      HET_RGNNDeltaWeightBWPropACGatherScatterListIdentical<
          COARSEN_FACTOR_2_FLAG_X, COARSEN_FACTOR_2_FLAG_Y, WORK_BLOCK_SIZE,
          int64_t, int64_t *, InputNumHeadOneFlag>
          <<<nblks_outer_product, nthrs, 0, stream>>>(
              node_feat.data_ptr<float>(), gradout.data_ptr<float>(),
              grad_weights.data_ptr<float>(),
              separate_coo_relptrs.data_ptr<int64_t>(),
              separate_coo_eids.data_ptr<int64_t>(), num_input_dim,
              num_output_per_head_dim, num_heads,
              thrust::raw_pointer_cast(
                  dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
              num_relations);
    } else {
      HET_RGNNDeltaNodeFeatInputBWProp<COARSEN_FACTOR_2_FLAG_X,
                                       COARSEN_FACTOR_2_FLAG_Y, WORK_BLOCK_SIZE,
                                       int64_t, int64_t *, InputNumHeadOneFlag>
          <<<nblks, nthrs, 0, stream>>>(
              gradout.data_ptr<float>(), weights_transposed.data_ptr<float>(),
              grad_node_feat.data_ptr<float>(),
              separate_coo_eids.data_ptr<int64_t>(),
              separate_coo_relptrs.data_ptr<int64_t>(),
              separate_coo_node_indices.data_ptr<int64_t>(),
              num_output_per_head_dim, num_input_dim, num_heads,
              thrust::raw_pointer_cast(
                  dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
              num_relations);
      HET_RGNNDeltaWeightBWProp<COARSEN_FACTOR_2_FLAG_X,
                                COARSEN_FACTOR_2_FLAG_Y, WORK_BLOCK_SIZE,
                                int64_t, int64_t *, InputNumHeadOneFlag>
          <<<nblks_outer_product, nthrs, 0, stream>>>(
              node_feat.data_ptr<float>(), gradout.data_ptr<float>(),
              grad_weights.data_ptr<float>(),
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

void RelationalMatMulCompactAsOfNode_unique_rel_node_indices(
    at::Tensor &unique_srcs_and_dests_rel_ptr,
    at::Tensor &unique_srcs_and_dests_node_indices,
    at::Tensor &weights_transposed, at::Tensor &node_feat, at::Tensor &gradout,
    at::Tensor &grad_node_feat, at::Tensor &grad_weights,
    bool InputNumHeadOneFlag) {
  at::Tensor dummy_tensor;
  if (InputNumHeadOneFlag) {
    _BackwardRelationalMatMul_separatecoo<true, true, 32, true, false, true>(
        dummy_tensor, dummy_tensor, dummy_tensor, unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_indices, weights_transposed, node_feat,
        gradout, grad_node_feat, grad_weights);
  } else {
    _BackwardRelationalMatMul_separatecoo<true, true, 32, true, false, false>(
        dummy_tensor, dummy_tensor, dummy_tensor, unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_indices, weights_transposed, node_feat,
        gradout, grad_node_feat, grad_weights);
  }
}

void RelationalMatMul_separatecoo(
    at::Tensor &separate_coo_relptrs, at::Tensor &separate_coo_node_indices,
    at::Tensor &separate_coo_eids, at::Tensor &weights_transposed,
    at::Tensor &node_feat, at::Tensor &gradout, at::Tensor &grad_node_feat,
    at::Tensor &grad_weights, bool InputNumHeadOneFlag) {
  at::Tensor dummy_tensor;
  if (separate_coo_eids.data_ptr() == separate_coo_node_indices.data_ptr()) {
    if (InputNumHeadOneFlag) {
      _BackwardRelationalMatMul_separatecoo<true, true, 32, false, true, true>(
          separate_coo_relptrs, dummy_tensor, separate_coo_eids, dummy_tensor,
          dummy_tensor, weights_transposed, node_feat, gradout, grad_node_feat,
          grad_weights);
    } else {
      _BackwardRelationalMatMul_separatecoo<true, true, 32, false, true, false>(
          separate_coo_relptrs, dummy_tensor, separate_coo_eids, dummy_tensor,
          dummy_tensor, weights_transposed, node_feat, gradout, grad_node_feat,
          grad_weights);
    }
  } else {
    if (InputNumHeadOneFlag) {
      _BackwardRelationalMatMul_separatecoo<true, true, 32, false, false, true>(
          separate_coo_relptrs, separate_coo_node_indices, separate_coo_eids,
          dummy_tensor, dummy_tensor, weights_transposed, node_feat, gradout,
          grad_node_feat, grad_weights);
    } else {
      _BackwardRelationalMatMul_separatecoo<true, true, 32, false, false,
                                            false>(
          separate_coo_relptrs, separate_coo_node_indices, separate_coo_eids,
          dummy_tensor, dummy_tensor, weights_transposed, node_feat, gradout,
          grad_node_feat, grad_weights);
    }
  }
}

// void RelationalMatMul_ACGatherScatterListIdentical_separatecoo(
//     at::Tensor &separate_coo_relptrs, at::Tensor &separate_coo_eids,
//     at::Tensor &weights_transposed, at::Tensor &node_feat, at::Tensor
//     &gradout, at::Tensor &grad_node_feat, at::Tensor &grad_weights, bool
//     InputNumHeadOneFlag) {
//   assert(0 && "deprecated");
//   at::Tensor dummy_tensor;
//   if (InputNumHeadOneFlag) {
//     _BackwardRelationalMatMul_separatecoo<true, true, 32, false, true, true>(
//         separate_coo_relptrs, dummy_tensor, separate_coo_eids, dummy_tensor,
//         dummy_tensor, weights_transposed, node_feat, gradout, grad_node_feat,
//         grad_weights);
//   } else {
//     _BackwardRelationalMatMul_separatecoo<true, true, 32, false, true,
//     false>(
//         separate_coo_relptrs, dummy_tensor, separate_coo_eids, dummy_tensor,
//         dummy_tensor, weights_transposed, node_feat, gradout, grad_node_feat,
//         grad_weights);
//   }
// }

// void RelationalMatMulCompactAsOfNodeSingleEnded_unique_rel_node_indices(
//     at::Tensor& unique_srcs_and_dests_rel_ptr,
//     at::Tensor& unique_srcs_and_dests_node_indices,
//     at::Tensor& separate_coo_rel_ptr, at::Tensor& separate_coo_node_indices,
//     at::Tensor& separate_coo_eids, at::Tensor& weight_transposed,
//     at::Tensor& node_feat, at::Tensor& ret, at::Tensor& gradout,
//     at::Tensor& grad_weights, at::Tensor& grad_node_feat,
//     bool InputNumHeadOneFlag) {
//   at::Tensor dummy_tensor;
//   if (InputNumHeadOneFlag) {
//     _BackwardRelationalMatMul_separatecoo<true, true, 32, true, true, false,
//                                           true>(
//         separate_coo_rel_ptr, separate_coo_node_indices, separate_coo_eids,
//         unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
//         weight_transposed, node_feat, gradout, grad_node_feat, grad_weights);
//   } else {
//     _BackwardRelationalMatMul_separatecoo<true, true, 32, true, true, false,
//                                           false>(
//         separate_coo_rel_ptr, separate_coo_node_indices, separate_coo_eids,
//         unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
//         weight_transposed, node_feat, gradout, grad_node_feat, grad_weights);
//   }
// }

// NB: We may rely on HGTCompactAsOfNodesEdgeAttentionSecondStage in
// [[hetero_edgesoftmax/include/DGLHackKernel/HGT/HGTForwardKernels.cu.h]]
// adapted from _RelationalFusedGATKernel in
// hetero_edgesoftmax/include/DGLHackKernel/OpExport/RGATOps.inc.h
template </*int XPU, */ typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool IntegratedFormatRatherThanSeparateFlag,
          bool CSRRatherThanCOOFlag>
void inner_product_various_left_and_node_right(
    at::Tensor &separate_coo_eids, at::Tensor &separate_coo_rel_ptrs,
    at::Tensor &separate_coo_row_indices, at::Tensor &separate_coo_col_indices,
    at::Tensor &outcsr_row_ptr, at::Tensor &outcsr_col_idx,
    at::Tensor &outcsr_eids, at::Tensor &outcsr_reltypes,
    at::Tensor &unique_srcs_and_dests_rel_ptr,
    at::Tensor &unique_srcs_and_dests_node_indices, at::Tensor &feat_src,
    at::Tensor &feat_dst, at::Tensor &grad_inner_product,
    at::Tensor &grad_feat_src, at::Tensor &grad_feat_dst) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  BackwardInnerProductData<Idx, DType> gdata{
      .feat_src_xlen = SeastarComputeXLength<>(feat_src),
      .num_heads = SeastarComputeXLength<>(grad_inner_product),
      .eids = nullptr,  // assigned later in if branches
      .feat_src = feat_src.data_ptr<DType>(),
      .feat_dst = feat_dst.data_ptr<DType>(),
      .grad_inner_product = grad_inner_product.data_ptr<DType>(),
      .grad_feat_dst = grad_feat_dst.data_ptr<DType>(),
      .grad_feat_src = grad_feat_src.data_ptr<DType>()};

  if constexpr (IntegratedFormatRatherThanSeparateFlag &&
                CSRRatherThanCOOFlag) {
    // Integrated CSR
    gdata.eids = outcsr_eids.data_ptr<Idx>();
    // NB: updated to Type 2 Schedule:
    // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
    // head -> threadIdx.y
    // node -> blockIdx.y
    // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x
    int64_t outcsr_num_rows = outcsr_row_ptr.numel() - 1;
    auto [nblks, nthrs] = get_type2_schedule(
        gdata.num_heads, gdata.feat_src_xlen, outcsr_num_rows);

    HET_inner_product_bck_kernel<Idx, DType, CompactAsOfNodeFlag, true, false>
        <<<nblks, nthrs, 0, stream>>>(
            gdata, outcsr_row_ptr.data_ptr<Idx>(),
            outcsr_col_idx.data_ptr<Idx>(), outcsr_reltypes.data_ptr<Idx>(),
            outcsr_num_rows, unique_srcs_and_dests_rel_ptr.data_ptr<Idx>(),
            unique_srcs_and_dests_node_indices.data_ptr<Idx>(),
            unique_srcs_and_dests_rel_ptr.numel() - 1);
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
    Idx *separate_coo_row_indices_data_ptr =
        separate_coo_row_indices.numel() > 0
            ? separate_coo_row_indices.data_ptr<Idx>()
            : nullptr;
    Idx *separate_coo_col_indices_data_ptr =
        separate_coo_col_indices.numel() > 0
            ? separate_coo_col_indices.data_ptr<Idx>()
            : nullptr;
    Idx *separate_coo_rel_ptrs_data_ptr =
        separate_coo_rel_ptrs.numel() > 0
            ? separate_coo_rel_ptrs.data_ptr<Idx>()
            : nullptr;
    Idx *unique_srcs_and_dests_rel_ptr_data_ptr =
        unique_srcs_and_dests_rel_ptr.numel() > 0
            ? unique_srcs_and_dests_rel_ptr.data_ptr<Idx>()
            : nullptr;
    Idx *unique_srcs_and_dests_node_indices_data_ptr =
        unique_srcs_and_dests_node_indices.numel() > 0
            ? unique_srcs_and_dests_node_indices.data_ptr<Idx>()
            : nullptr;

    HET_inner_product_bck_kernel_edge_parallel<Idx, DType, CompactAsOfNodeFlag,
                                               true, true>
        <<<nblks, nthrs, 0, stream>>>(
            gdata, separate_coo_row_indices_data_ptr,
            separate_coo_col_indices_data_ptr, separate_coo_rel_ptrs_data_ptr,
            num_edges, unique_srcs_and_dests_rel_ptr_data_ptr,
            unique_srcs_and_dests_node_indices_data_ptr, num_relations);
  } else {
    assert(0 && "Not implemented");
  }
}

void inner_product_node_compact_and_node_separatecoo(
    at::Tensor &unique_srcs_and_dests_rel_ptr,
    at::Tensor &unique_srcs_and_dests_node_idx,
    at::Tensor &separate_coo_rel_ptr, at::Tensor &separate_coo_eids,
    at::Tensor &separate_coo_row_indices, at::Tensor &separate_coo_col_indices,
    at::Tensor &left_node_compact_data,
    at::Tensor &right_node_vectors,  // at::Tensor& ret,
    at::Tensor &grad_inner_product, at::Tensor &grad_left_node_compact_data,
    at::Tensor &grad_right_node_vectors) {
  at::Tensor dummy_tensor;
  inner_product_various_left_and_node_right<int64_t, float, true, false, false>(
      separate_coo_eids, separate_coo_rel_ptr, separate_coo_row_indices,
      separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
      dummy_tensor, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_idx, left_node_compact_data,
      right_node_vectors, grad_inner_product, grad_left_node_compact_data,
      grad_right_node_vectors);
}

void inner_product_edge_and_node_separatecoo(
    at::Tensor &separate_coo_eids, at::Tensor &separate_coo_row_indices,
    at::Tensor &separate_coo_col_indices, at::Tensor &left_edge_data,
    at::Tensor &right_node_vectors,
    at::Tensor &grad_inner_product,  // at::Tensor& gradout,
    at::Tensor &grad_left_edge_data, at::Tensor &grad_right_node_vectors) {
  at::Tensor dummy_tensor;
  inner_product_various_left_and_node_right<int64_t, float, false, false,
                                            false>(
      separate_coo_eids, dummy_tensor, separate_coo_row_indices,
      separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
      dummy_tensor, dummy_tensor, dummy_tensor, left_edge_data,
      right_node_vectors, grad_inner_product, grad_left_edge_data,
      grad_right_node_vectors);
}

template <bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
          int WORK_BLOCK_SIZE>
void _RelationalMatmulNoScatterGatherList(at::Tensor &ntype_offset_ptrs,
                                          at::Tensor &weights_transposed,
                                          at::Tensor &node_feat_input,
                                          at::Tensor &grad_node_feat_output,
                                          at::Tensor &grad_weights,
                                          at::Tensor &grad_node_feat_input) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  assert(weights_transposed.size(1) == 1 && "assertion n_head == 1 failed");
  const int64_t num_input_dim = weights_transposed.size(3);
  const int64_t num_output_dim =
      weights_transposed.size(2);  // weight shape (num_ntypes,
                                   // in_feat, out_feat)
  int64_t num_ntypes = ntype_offset_ptrs.numel() - 1;
  int64_t num_nodes = node_feat_input.size(0);

  constexpr int THREADING_BLOCK_SIZE_X =
      COARSEN_FACTOR_2_FLAG_X ? WORK_BLOCK_SIZE / 2 : WORK_BLOCK_SIZE;
  constexpr int THREADING_BLOCK_SIZE_Y =
      COARSEN_FACTOR_2_FLAG_Y ? WORK_BLOCK_SIZE / 2 : WORK_BLOCK_SIZE;

  int grid_dim_y = std::min(
      ceil_div<>(num_nodes, (int64_t)WORK_BLOCK_SIZE),
      (int64_t)32768);  // using 32768 instead of 65535 to leave some space in
                        // case the total number of blocks is slightly larger
                        // due to relationship with very few workloads
  std::vector<int> num_blocks_assignment_for_same_ntype_vect,
      num_blocks_assignment_for_all_prev_ntype_vect;

  at::Tensor ntype_offset_ptrs_cpu_contiguous =
      ntype_offset_ptrs.cpu().contiguous();
  std::tie(num_blocks_assignment_for_same_ntype_vect,
           num_blocks_assignment_for_all_prev_ntype_vect) =
      get_schedule_by_relation_kernel_launch_metadata<false, false, int64_t *>(
          grid_dim_y, num_ntypes, WORK_BLOCK_SIZE,
          ntype_offset_ptrs_cpu_contiguous.data_ptr<int64_t>(),
          ntype_offset_ptrs_cpu_contiguous.data_ptr<int64_t>() + num_ntypes +
              1);

  grid_dim_y = num_blocks_assignment_for_all_prev_ntype_vect.back();

  thrust::device_vector<int> dev_num_blocks_assignment_for_same_ntype_vect(
      num_blocks_assignment_for_same_ntype_vect.begin(),
      num_blocks_assignment_for_same_ntype_vect.end());
  thrust::device_vector<int> dev_num_blocks_assignment_for_all_prev_ntype_vect(
      num_blocks_assignment_for_all_prev_ntype_vect.begin(),
      num_blocks_assignment_for_all_prev_ntype_vect.end());
  // NB: my shmem sgemm matmul scheme
  // in NoScatterGather scenario, there is no such thing as multi-headed
  const dim3 nblks(ceil_div<>(num_input_dim, (long)WORK_BLOCK_SIZE), grid_dim_y,
                   1);
  const dim3 nthrs(THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y);
  HET_RGNNMatmulNoScatterGatherListFwOrBwProp<
      COARSEN_FACTOR_2_FLAG_X, COARSEN_FACTOR_2_FLAG_Y, WORK_BLOCK_SIZE,
      int64_t, int64_t *><<<nblks, nthrs, 0, stream>>>(
      grad_node_feat_output.data_ptr<float>(),
      weights_transposed.data_ptr<float>(),
      grad_node_feat_input.data_ptr<float>(),
      ntype_offset_ptrs.data_ptr<int64_t>(),
      thrust::raw_pointer_cast(
          dev_num_blocks_assignment_for_all_prev_ntype_vect.data()),
      num_ntypes, num_output_dim, num_input_dim);
  // NB: my shmem sgemm matmul scheme
  const dim3 nblks_outer_product(
      ceil_div<>(num_output_dim, (long)WORK_BLOCK_SIZE),
      ceil_div<>(num_input_dim, (long)WORK_BLOCK_SIZE), grid_dim_y);
  HET_RGNNDeltaWeightNoScatterGatherListBWProp<
      COARSEN_FACTOR_2_FLAG_X, COARSEN_FACTOR_2_FLAG_Y, WORK_BLOCK_SIZE,
      int64_t, int64_t *><<<nblks_outer_product, nthrs, 0, stream>>>(
      node_feat_input.data_ptr<float>(),
      grad_node_feat_output.data_ptr<float>(), grad_weights.data_ptr<float>(),
      ntype_offset_ptrs.data_ptr<int64_t>(), num_input_dim, num_output_dim,
      thrust::raw_pointer_cast(
          dev_num_blocks_assignment_for_all_prev_ntype_vect.data()),
      num_ntypes);
}

void RelationalMatmulNoScatterGatherList(at::Tensor &ntype_offset_ptrs,
                                         at::Tensor &weights_transposed,
                                         at::Tensor &node_feat_input,
                                         at::Tensor &grad_node_feat_output,
                                         at::Tensor &grad_weights,
                                         at::Tensor &grad_node_feat_input) {
  _RelationalMatmulNoScatterGatherList<true, true, 32>(
      ntype_offset_ptrs, weights_transposed, node_feat_input,
      grad_node_feat_output, grad_weights, grad_node_feat_input);
}

}  // namespace BckProp
}  // namespace RGNN
}  // namespace TorchExport
}  // namespace HET

using namespace HET::TorchExport;
TORCH_LIBRARY_FRAGMENT(torch_hetero_edgesoftmax, m) {
  // kernels for generic hetero-gnn use declaration
  // RGNN Relational GEMM
  m.def("rgnn_relational_matmul", RGNN::FwProp::RelationalMatMul_separatecoo);
  m.def("backward_rgnn_relational_matmul",
        RGNN::BckProp::RelationalMatMul_separatecoo);
  //   m.def(
  //       "rgnn_relational_matmul_ac_gather_scatter_list_identical",
  //       RGNN::FwProp::RelationalMatMul_ACGatherScatterListIdentical_separatecoo);
  //   m.def(
  //       "backward_rgnn_relational_matmul_ac_gather_scatter_list_identical",
  //       RGNN::BckProp::RelationalMatMul_ACGatherScatterListIdentical_separatecoo);
  m.def("backward_rgnn_relational_matmul_compact_as_of_node",
        RGNN::BckProp::RelationalMatMulCompactAsOfNode_unique_rel_node_indices);
  m.def("rgnn_relational_matmul_compact_as_of_node",
        RGNN::FwProp::RelationalMatMulCompactAsOfNode_unique_rel_node_indices);
  //   m.def(
  //       "rgnn_relational_matmul_compact_as_of_node_single_ended",
  //       RGNN::FwProp::
  //           RelationalMatMulCompactAsOfNodeSingleEnded_unique_rel_node_indices);
  //   m.def(
  //       "backward_rgnn_relational_matmul_compact_as_of_node_single_ended",
  //       RGNN::BckProp::
  //           RelationalMatMulCompactAsOfNodeSingleEnded_unique_rel_node_indices);
  m.def("rgnn_relational_matmul_no_scatter_gather_list",
        RGNN::FwProp::RelationalMatmulNoScatterGatherList);
  m.def("backward_rgnn_relational_matmul_no_scatter_gather_list",
        RGNN::BckProp::RelationalMatmulNoScatterGatherList);
  // RGNN innerproduct
  m.def("rgnn_inner_product_node_compact_and_node",
        RGNN::FwProp::inner_product_node_compact_and_node_separatecoo);
  m.def("backward_rgnn_inner_product_node_compact_and_node",
        RGNN::BckProp::inner_product_node_compact_and_node_separatecoo);
  m.def("rgnn_inner_product_edge_and_node",
        RGNN::FwProp::inner_product_edge_and_node_separatecoo);
  m.def("backward_rgnn_inner_product_edge_and_node",
        RGNN::BckProp::inner_product_edge_and_node_separatecoo);
}
