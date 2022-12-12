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

namespace HET {
namespace TorchExport {
namespace RGNN {
namespace FwProp {
template <bool COARSEN_FACTOR_2_FLAG, int THREADING_BLOCK_SIZE,
          bool CompactAsOfNodeFlag, bool SingleSidedCompactAsOfNodeFlag,
          bool ACGatherScatterListIdenticalFlag, bool InputNumHeadOneFlag>
void _RelationalMatMul_separatecoo(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_node_indices,
    at::Tensor& separate_coo_eids, at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& weights,
    at::Tensor& node_feat, at::Tensor& ret) {
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
  constexpr int WORK_BLOCK_SIZE =
      COARSEN_FACTOR_2_FLAG ? (THREADING_BLOCK_SIZE * 2) : THREADING_BLOCK_SIZE;
  if constexpr (SingleSidedCompactAsOfNodeFlag) {
    CONSTEXPR_TRUE_CLAUSE_STATIC_ASSERT(
        SingleSidedCompactAsOfNodeFlag, CompactAsOfNodeFlag,
        "SingleSidedCompactAsOfNodeFlag requires CompactAsOfNodeFlag");
  }

  if constexpr (CompactAsOfNodeFlag && !SingleSidedCompactAsOfNodeFlag) {
    num_edges = unique_srcs_and_dests_node_indices.numel();
  } else {
    num_edges = separate_coo_eids.numel();
  }
  int grid_dim_y = std::min(
      ceil_div<>(num_edges, (int64_t)WORK_BLOCK_SIZE),
      (int64_t)32768);  // using 32768 instead of 65535 to leave some space in
                        // case the total number of blocks is slightly larger
                        // due to relationship with very few workloads
  std::vector<int> num_blocks_assignment_for_same_relation_vect,
      num_blocks_assignment_for_all_prev_relation_vect;
  if constexpr (CompactAsOfNodeFlag && !SingleSidedCompactAsOfNodeFlag) {
    at::Tensor unique_srcs_and_dests_rel_ptr_cpu_contiguous =
        unique_srcs_and_dests_rel_ptr.cpu().contiguous();
    std::tie(num_blocks_assignment_for_same_relation_vect,
             num_blocks_assignment_for_all_prev_relation_vect) =
        get_schedule_by_relation_kernel_launch_metadata<false, false, int64_t*>(
            grid_dim_y, num_relations, WORK_BLOCK_SIZE,
            unique_srcs_and_dests_rel_ptr_cpu_contiguous.data_ptr<int64_t>(),
            unique_srcs_and_dests_rel_ptr_cpu_contiguous.data_ptr<int64_t>() +
                num_relations + 1);
  } else {
    at::Tensor separate_coo_relptrs_cpu_contiguous =
        separate_coo_relptrs.cpu().contiguous();
    std::tie(num_blocks_assignment_for_same_relation_vect,
             num_blocks_assignment_for_all_prev_relation_vect) =
        get_schedule_by_relation_kernel_launch_metadata<false, false, int64_t*>(
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
    const dim3 nblks(ceil_div<>(num_output_per_head_dim, (long)WORK_BLOCK_SIZE),
                     grid_dim_y, num_heads);
    const dim3 nthrs(THREADING_BLOCK_SIZE, THREADING_BLOCK_SIZE);
    // std::cout << "nblks.x: " << nblks.x << " nblks.y: " << nblks.y
    //           << " nblks.z: " << nblks.z << std::endl;
    if constexpr (!SingleSidedCompactAsOfNodeFlag) {
      HET_RGNNFeatCompactFWProp<COARSEN_FACTOR_2_FLAG, THREADING_BLOCK_SIZE,
                                int64_t, int64_t*, InputNumHeadOneFlag>
          <<<nblks, nthrs, 0, stream>>>(
              node_feat.data_ptr<float>(), weights.data_ptr<float>(),
              ret.data_ptr<float>(),
              unique_srcs_and_dests_rel_ptr.data_ptr<int64_t>(),
              unique_srcs_and_dests_node_indices.data_ptr<int64_t>(),
              num_input_dim, num_output_per_head_dim, num_heads,
              thrust::raw_pointer_cast(
                  dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
              num_relations);

    } else {
      HET_RGNNFeatCompactFWPropSingleSided<COARSEN_FACTOR_2_FLAG,
                                           THREADING_BLOCK_SIZE, int64_t,
                                           int64_t*, InputNumHeadOneFlag>
          <<<nblks, nthrs, 0, stream>>>(
              node_feat.data_ptr<float>(), weights.data_ptr<float>(),
              ret.data_ptr<float>(),
              unique_srcs_and_dests_rel_ptr.data_ptr<int64_t>(),
              unique_srcs_and_dests_node_indices.data_ptr<int64_t>(),
              separate_coo_relptrs.data_ptr<int64_t>(),
              separate_coo_node_indices.data_ptr<int64_t>(),
              separate_coo_eids.data_ptr<int64_t>(), num_input_dim,
              num_output_per_head_dim, num_heads,
              thrust::raw_pointer_cast(
                  dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
              num_relations);
    }

  } else {
    // NB: my shmem sgemm matmul scheme
    const dim3 nblks(ceil_div<>(num_output_per_head_dim, (long)WORK_BLOCK_SIZE),
                     grid_dim_y, num_heads);
    const dim3 nthrs(THREADING_BLOCK_SIZE, THREADING_BLOCK_SIZE);
    // std::cout << "nblks.x: " << nblks.x << " nblks.y: " << nblks.y
    //           << " nblks.z: " << nblks.z << std::endl;
    if constexpr (ACGatherScatterListIdenticalFlag) {
      HET_RGNNFeatPerEdgeFWPropACGatherScatterListIdentical<
          COARSEN_FACTOR_2_FLAG, THREADING_BLOCK_SIZE, int64_t, int64_t*,
          InputNumHeadOneFlag><<<nblks, nthrs, 0, stream>>>(
          node_feat.data_ptr<float>(), weights.data_ptr<float>(),
          ret.data_ptr<float>(), separate_coo_relptrs.data_ptr<int64_t>(),
          separate_coo_eids.data_ptr<int64_t>(), num_input_dim,
          num_output_per_head_dim, num_heads,
          thrust::raw_pointer_cast(
              dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
          num_relations);
    } else {
      HET_RGNNFeatPerEdgeFWProp<COARSEN_FACTOR_2_FLAG, THREADING_BLOCK_SIZE,
                                int64_t, int64_t*, InputNumHeadOneFlag>
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

template <bool COARSEN_FACTOR_2_FLAG, int THREADING_BLOCK_SIZE>
void _RelationalMatmulNoScatterGatherList(at::Tensor& ntype_offset_ptrs,
                                          at::Tensor& weights,
                                          at::Tensor& inputs, at::Tensor& ret) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int64_t num_heads = 1;

  assert(weights.size(1) == 1 && "assertion n_head == 1 failed");
  const int64_t num_input_dim = weights.size(2);
  const int64_t num_output_dim = weights.size(3);  // weight shape (num_ntypes,
                                                   // in_feat, out_feat)
  int64_t num_ntypes = ntype_offset_ptrs.numel() - 1;
  int64_t num_nodes = inputs.size(0);

  constexpr int WORK_BLOCK_SIZE =
      (COARSEN_FACTOR_2_FLAG ? (THREADING_BLOCK_SIZE * 2)
                             : THREADING_BLOCK_SIZE);

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
      get_schedule_by_relation_kernel_launch_metadata<false, false, int64_t*>(
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
  const dim3 nblks(ceil_div<>(num_output_dim, (long)WORK_BLOCK_SIZE),
                   grid_dim_y, num_heads);
  const dim3 nthrs(THREADING_BLOCK_SIZE, THREADING_BLOCK_SIZE);
  // std::cout << "nblks.x: " << nblks.x << " nblks.y: " << nblks.y
  //           << " nblks.z: " << nblks.z << std::endl;
  HET_RGNNMatmulNoScatterGatherListFwOrBwProp<
      COARSEN_FACTOR_2_FLAG, THREADING_BLOCK_SIZE, int64_t, int64_t*>
      <<<nblks, nthrs, 0, stream>>>(
          inputs.data_ptr<float>(), weights.data_ptr<float>(),
          ret.data_ptr<float>(), ntype_offset_ptrs.data_ptr<int64_t>(),

          thrust::raw_pointer_cast(
              dev_num_blocks_assignment_for_all_prev_ntype_vect.data()),
          num_ntypes, num_input_dim, num_output_dim);
}

void RelationalMatmulNoScatterGatherList(at::Tensor& ntype_offset_ptrs,
                                         at::Tensor& weights,
                                         at::Tensor& inputs, at::Tensor& ret) {
  _RelationalMatmulNoScatterGatherList<true, 16>(ntype_offset_ptrs, weights,
                                                 inputs, ret);
}

void RelationalMatMul_separatecoo(at::Tensor& separate_coo_relptrs,
                                  at::Tensor& separate_coo_node_indices,
                                  at::Tensor& separate_coo_eids,
                                  at::Tensor& weights, at::Tensor& node_feat,
                                  at::Tensor& ret, bool InputNumHeadOneFlag) {
  at::Tensor dummy_tensor;
  if (InputNumHeadOneFlag) {
    _RelationalMatMul_separatecoo<true, 16, false, false, false, true>(
        separate_coo_relptrs, separate_coo_node_indices, separate_coo_eids,
        dummy_tensor, dummy_tensor, weights, node_feat, ret);
  } else {
    _RelationalMatMul_separatecoo<true, 16, false, false, false, false>(
        separate_coo_relptrs, separate_coo_node_indices, separate_coo_eids,
        dummy_tensor, dummy_tensor, weights, node_feat, ret);
  }
}

void RelationalMatMul_ACGatherScatterListIdentical_separatecoo(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_eids,
    at::Tensor& weights, at::Tensor& node_feat, at::Tensor& ret,
    bool InputNumHeadOneFlag) {
  at::Tensor dummy_tensor;
  if (InputNumHeadOneFlag) {
    _RelationalMatMul_separatecoo<true, 16, false, false, true, true>(
        separate_coo_relptrs, dummy_tensor, separate_coo_eids, dummy_tensor,
        dummy_tensor, weights, node_feat, ret);
  } else {
    _RelationalMatMul_separatecoo<true, 16, false, false, true, false>(
        separate_coo_relptrs, dummy_tensor, separate_coo_eids, dummy_tensor,
        dummy_tensor, weights, node_feat, ret);
  }
}

void RelationalMatMulCompactAsOfNode_unique_rel_node_indices(
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& weight,
    at::Tensor& node_feat, at::Tensor& ret, bool InputNumHeadOneFlag) {
  at::Tensor dummy_tensor;
  if (InputNumHeadOneFlag) {
    _RelationalMatMul_separatecoo<true, 16, true, false, false, true>(
        dummy_tensor, dummy_tensor, dummy_tensor, unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_indices, weight, node_feat, ret);
  } else {
    _RelationalMatMul_separatecoo<true, 16, true, false, false, false>(
        dummy_tensor, dummy_tensor, dummy_tensor, unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_indices, weight, node_feat, ret);
  }
}

void RelationalMatMulCompactAsOfNodeSingleEnded_unique_rel_node_indices(
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices,
    at::Tensor& separate_coo_rel_ptr, at::Tensor& separate_coo_node_indices,
    at::Tensor& separate_coo_eids, at::Tensor& weight, at::Tensor& node_feat,
    at::Tensor& ret, bool InputNumHeadOneFlag) {
  at::Tensor dummy_tensor;
  if (InputNumHeadOneFlag) {
    _RelationalMatMul_separatecoo<true, 16, true, true, false, true>(
        separate_coo_rel_ptr, separate_coo_node_indices, separate_coo_eids,
        unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
        weight, node_feat, ret);
  } else {
    _RelationalMatMul_separatecoo<true, 16, true, true, false, false>(
        separate_coo_rel_ptr, separate_coo_node_indices, separate_coo_eids,
        unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
        weight, node_feat, ret);
  }
}

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
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_rel_ptrs,
    at::Tensor& separate_coo_row_indices, at::Tensor& separate_coo_col_indices,
    at::Tensor& incsr_row_ptr, at::Tensor& incsr_col_idx,
    at::Tensor& incsr_eids, at::Tensor& incsr_reltypes,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& feat_dst, at::Tensor& edge_inner_product) {
  const Idx MAX_NBLKS = 65535;
  const Idx MAX_NTHRS = 1024;
  InnerProductData<Idx, DType> gdata;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  gdata.num_heads = SeastarComputeXLength<>(edge_inner_product);
  int64_t feat_src_xlen = SeastarComputeXLength<>(feat_src);
  // NB: in this case gdata.n, calculation is removed since el is now per edge
  // rather than per node
  gdata.feat_src = feat_src.data_ptr<DType>();
  gdata.feat_dst = feat_dst.data_ptr<DType>();
  gdata.edge_inner_product = edge_inner_product.data_ptr<DType>();
  // gdata.n = el.numel() / el_xlen;
  gdata.feat_src_xlen = feat_src_xlen;

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

    int64_t nthrs_y = SeastarFindNumThreads(gdata.num_heads, 64);
    int64_t nthrs_x = SeastarFindNumThreads(
        gdata.feat_src_xlen / gdata.num_heads, MAX_NTHRS / nthrs_y);
    int64_t nblks_x = 1;
    int64_t incsr_num_rows = incsr_row_ptr.numel() - 1;
    int64_t nblks_y = std::min(incsr_num_rows, MAX_NBLKS);
    const dim3 nthrs2(nthrs_x, nthrs_y);
    const dim3 nblks2(nblks_x, nblks_y);

    Idx* incsr_row_ptr_data_ptr =
        incsr_row_ptr.numel() > 0 ? incsr_row_ptr.data_ptr<Idx>() : nullptr;
    Idx* incsr_col_idx_data_ptr =
        incsr_col_idx.numel() > 0 ? incsr_col_idx.data_ptr<Idx>() : nullptr;
    Idx* incsr_reltypes_data_ptr =
        incsr_reltypes.numel() > 0 ? incsr_reltypes.data_ptr<Idx>() : nullptr;
    Idx* unique_srcs_and_dests_rel_ptr_data_ptr =
        unique_srcs_and_dests_rel_ptr.numel() > 0
            ? unique_srcs_and_dests_rel_ptr.data_ptr<Idx>()
            : nullptr;
    Idx* unique_srcs_and_dests_node_indices_data_ptr =
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
    int nthrs_y_inner_product = SeastarFindNumThreads(gdata.num_heads, 64);
    int nthrs_x_inner_product =
        SeastarFindNumThreads(gdata.feat_src_xlen / gdata.num_heads,
                              MAX_NTHRS / nthrs_y_inner_product);
    int nblks_inner_product_x = 1;
    int nblks_inner_product_y = std::min(num_edges, MAX_NBLKS);
    const dim3 nthrs_inner_product(nthrs_x_inner_product,
                                   nthrs_y_inner_product);
    const dim3 nblks_inner_product(nblks_inner_product_x,
                                   nblks_inner_product_y);
    Idx* separate_coo_row_indices_data_ptr =
        separate_coo_row_indices.numel() > 0
            ? separate_coo_row_indices.data_ptr<Idx>()
            : nullptr;
    Idx* separate_coo_col_indices_data_ptr =
        separate_coo_col_indices.numel() > 0
            ? separate_coo_col_indices.data_ptr<Idx>()
            : nullptr;
    Idx* separate_coo_rel_ptrs_data_ptr =
        separate_coo_rel_ptrs.numel() > 0
            ? separate_coo_rel_ptrs.data_ptr<Idx>()
            : nullptr;
    Idx* unique_srcs_and_dests_rel_ptr_data_ptr =
        unique_srcs_and_dests_rel_ptr.numel() > 0
            ? unique_srcs_and_dests_rel_ptr.data_ptr<Idx>()
            : nullptr;
    Idx* unique_srcs_and_dests_node_indices_data_ptr =
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
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_idx,
    at::Tensor& separate_coo_rel_ptr, at::Tensor& separate_coo_eids,
    at::Tensor& separate_coo_row_indices, at::Tensor& separate_coo_col_indices,
    at::Tensor& left_node_compact_data, at::Tensor& right_node_vectors,
    at::Tensor& edge_inner_product) {
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
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_row_indices,
    at::Tensor& separate_coo_col_indices, at::Tensor& left_edge_data,
    at::Tensor& right_node_vectors, at::Tensor& edge_inner_product) {
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
template <bool COARSEN_2_FACTOR_FLAG, int THREADING_BLOCK_SIZE,
          bool CompactAsOfNodeFlag, bool SingleSidedCompactAsOfNodeFlag,
          bool ACGatherScatterListIdenticalFlag, bool InputNumHeadOneFlag>
void _BackwardRelationalMatMul_separatecoo(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_node_indices,
    at::Tensor& separate_coo_eids, at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices,
    at::Tensor& weights_transposed, at::Tensor& node_feat, at::Tensor& gradout,
    at::Tensor& grad_node_feat, at::Tensor& grad_weights) {
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

  constexpr int64_t WORK_BLOCK_SIZE =
      (COARSEN_2_FACTOR_FLAG ? (THREADING_BLOCK_SIZE * 2)
                             : THREADING_BLOCK_SIZE);

  int64_t num_edges;
  assert(weights_transposed.is_contiguous());
  assert(node_feat.is_contiguous());
  assert(gradout.is_contiguous());
  assert(grad_node_feat.is_contiguous());
  assert(grad_weights.is_contiguous());
  if constexpr (SingleSidedCompactAsOfNodeFlag) {
    CONSTEXPR_TRUE_CLAUSE_STATIC_ASSERT(
        SingleSidedCompactAsOfNodeFlag, CompactAsOfNodeFlag,
        "SingleSidedCompactAsOfNodeFlag requires CompactAsOfNodeFlag");
  }

  if constexpr (CompactAsOfNodeFlag && !SingleSidedCompactAsOfNodeFlag) {
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
  if constexpr (CompactAsOfNodeFlag && !SingleSidedCompactAsOfNodeFlag) {
    at::Tensor unique_srcs_and_dests_rel_ptr_cpu_contiguous =
        unique_srcs_and_dests_rel_ptr.cpu().contiguous();
    std::tie(num_blocks_assignment_for_same_relation_vect,
             num_blocks_assignment_for_all_prev_relation_vect) =
        get_schedule_by_relation_kernel_launch_metadata<false, false, int64_t*>(
            grid_dim_y, num_relations, WORK_BLOCK_SIZE,
            unique_srcs_and_dests_rel_ptr_cpu_contiguous.data_ptr<int64_t>(),
            unique_srcs_and_dests_rel_ptr_cpu_contiguous.data_ptr<int64_t>() +
                num_relations + 1);

  } else {
    at::Tensor separate_coo_relptrs_cpu_contiguous =
        separate_coo_relptrs.cpu().contiguous();
    std::tie(num_blocks_assignment_for_same_relation_vect,
             num_blocks_assignment_for_all_prev_relation_vect) =
        get_schedule_by_relation_kernel_launch_metadata<false, false, int64_t*>(
            grid_dim_y, num_relations, WORK_BLOCK_SIZE,
            separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>(),
            separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>() +
                num_relations + 1);
  }
  // print all elements of num_blocks_assignment_for_all_prev_relation_vect
  // for (int i = 0; i < num_relations; i++) {
  //   std::cout << num_blocks_assignment_for_all_prev_relation_vect[i] << ' ';
  // }
  // std::cout << std::endl;
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
    // NB: my shmem sgemm matmul scheme
    const dim3 nblks(ceil_div<>(num_input_dim, (long)WORK_BLOCK_SIZE),
                     grid_dim_y, num_heads);
    const dim3 nblks_outer_product(
        ceil_div<>(num_output_per_head_dim, (long)WORK_BLOCK_SIZE),
        ceil_div<>(num_input_dim, (long)WORK_BLOCK_SIZE),
        num_heads * grid_dim_y);
    assert(num_heads * grid_dim_y < 65535 && "num_head*grid_dim_y>=65535");

    const dim3 nthrs(THREADING_BLOCK_SIZE, THREADING_BLOCK_SIZE);
    // NB: #head of node_feat is 1 when InputNumHeadOneFlag is true
    if constexpr (!SingleSidedCompactAsOfNodeFlag) {
      // cuda_err_chk(cudaGetLastError());
      std::cout << gradout.numel() << std::endl;
      HET_RGNNDeltaNodeFeatInputCompactBWProp<COARSEN_2_FACTOR_FLAG,
                                              THREADING_BLOCK_SIZE, int64_t,
                                              int64_t*, InputNumHeadOneFlag>
          <<<nblks, nthrs, 0, stream>>>(
              gradout.data_ptr<float>(), weights_transposed.data_ptr<float>(),
              grad_node_feat.data_ptr<float>(),
              unique_srcs_and_dests_rel_ptr.data_ptr<int64_t>(),
              unique_srcs_and_dests_node_indices.data_ptr<int64_t>(), num_edges,
              num_output_per_head_dim, num_input_dim, num_heads,
              thrust::raw_pointer_cast(
                  dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
              num_relations);
      // cuda_err_chk(cudaGetLastError());
      HET_RGNNDeltaWeightCompactBWProp<COARSEN_2_FACTOR_FLAG,
                                       THREADING_BLOCK_SIZE, int64_t, int64_t*,
                                       InputNumHeadOneFlag>
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
      HET_RGNNDeltaNodeFeatInputCompactBWPropSingleSided<
          COARSEN_2_FACTOR_FLAG, THREADING_BLOCK_SIZE, int64_t, int64_t*,
          InputNumHeadOneFlag><<<nblks, nthrs, 0, stream>>>(
          gradout.data_ptr<float>(), weights_transposed.data_ptr<float>(),
          grad_node_feat.data_ptr<float>(),
          unique_srcs_and_dests_rel_ptr.data_ptr<int64_t>(),
          unique_srcs_and_dests_node_indices.data_ptr<int64_t>(),
          separate_coo_relptrs.data_ptr<int64_t>(),
          separate_coo_eids.data_ptr<int64_t>(), num_edges,
          num_output_per_head_dim, num_input_dim, num_heads,
          thrust::raw_pointer_cast(
              dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
          num_relations);
      HET_RGNNDeltaWeightCompactBWPropSingleSided<COARSEN_2_FACTOR_FLAG,
                                                  THREADING_BLOCK_SIZE, int64_t,
                                                  int64_t*, InputNumHeadOneFlag>
          <<<nblks_outer_product, nthrs, 0, stream>>>(
              gradout.data_ptr<float>(), node_feat.data_ptr<float>(),
              grad_weights.data_ptr<float>(),
              unique_srcs_and_dests_rel_ptr.data_ptr<int64_t>(),
              unique_srcs_and_dests_node_indices.data_ptr<int64_t>(),
              separate_coo_relptrs.data_ptr<int64_t>(),
              separate_coo_eids.data_ptr<int64_t>(), num_edges, num_input_dim,
              num_output_per_head_dim, num_heads,
              thrust::raw_pointer_cast(
                  dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
              num_relations);
    }
  } else {
    // NB: my shmem sgemm matmul scheme
    const dim3 nblks(ceil_div<>(num_input_dim, (long)WORK_BLOCK_SIZE),
                     grid_dim_y, num_heads);
    const dim3 nblks_outer_product(
        ceil_div<>(num_output_per_head_dim, (long)WORK_BLOCK_SIZE),
        ceil_div<>(num_input_dim, (long)WORK_BLOCK_SIZE),
        num_heads * grid_dim_y);
    assert(num_heads * grid_dim_y < 65535 && "num_head*grid_dim_y>=65535");
    const dim3 nthrs(THREADING_BLOCK_SIZE, THREADING_BLOCK_SIZE);
    // NB: #head of node_feat is 1 when InputNumHeadOneFlag is true
    if constexpr (ACGatherScatterListIdenticalFlag) {
      HET_RGNNDeltaNodeFeatInputBWPropACGatherScatterListIdentical<
          COARSEN_2_FACTOR_FLAG, THREADING_BLOCK_SIZE, int64_t, int64_t*,
          InputNumHeadOneFlag><<<nblks, nthrs, 0, stream>>>(
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
          COARSEN_2_FACTOR_FLAG, THREADING_BLOCK_SIZE, int64_t, int64_t*,
          InputNumHeadOneFlag><<<nblks_outer_product, nthrs, 0, stream>>>(
          node_feat.data_ptr<float>(), gradout.data_ptr<float>(),
          grad_weights.data_ptr<float>(), separate_coo_eids.data_ptr<int64_t>(),
          separate_coo_relptrs.data_ptr<int64_t>(), num_input_dim,
          num_output_per_head_dim, num_heads,
          thrust::raw_pointer_cast(
              dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
          num_relations);
    } else {
      HET_RGNNDeltaNodeFeatInputBWProp<COARSEN_2_FACTOR_FLAG,
                                       THREADING_BLOCK_SIZE, int64_t, int64_t*,
                                       InputNumHeadOneFlag>
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
      HET_RGNNDeltaWeightBWProp<COARSEN_2_FACTOR_FLAG, THREADING_BLOCK_SIZE,
                                int64_t, int64_t*, InputNumHeadOneFlag>
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
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices,
    at::Tensor& weights_transposed, at::Tensor& node_feat, at::Tensor& gradout,
    at::Tensor& grad_node_feat, at::Tensor& grad_weights,
    bool InputNumHeadOneFlag) {
  at::Tensor dummy_tensor;
  if (InputNumHeadOneFlag) {
    _BackwardRelationalMatMul_separatecoo<true, 16, true, false, false, true>(
        dummy_tensor, dummy_tensor, dummy_tensor, unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_indices, weights_transposed, node_feat,
        gradout, grad_node_feat, grad_weights);
  } else {
    _BackwardRelationalMatMul_separatecoo<true, 16, true, false, false, false>(
        dummy_tensor, dummy_tensor, dummy_tensor, unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_indices, weights_transposed, node_feat,
        gradout, grad_node_feat, grad_weights);
  }
}

void RelationalMatMul_separatecoo(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_node_indices,
    at::Tensor& separate_coo_eids, at::Tensor& weights_transposed,
    at::Tensor& node_feat, at::Tensor& gradout, at::Tensor& grad_node_feat,
    at::Tensor& grad_weights, bool InputNumHeadOneFlag) {
  at::Tensor dummy_tensor;
  if (InputNumHeadOneFlag) {
    _BackwardRelationalMatMul_separatecoo<true, 16, false, false, false, true>(
        separate_coo_relptrs, separate_coo_node_indices, separate_coo_eids,
        dummy_tensor, dummy_tensor, weights_transposed, node_feat, gradout,
        grad_node_feat, grad_weights);
  } else {
    _BackwardRelationalMatMul_separatecoo<true, 16, false, false, false, false>(
        separate_coo_relptrs, separate_coo_node_indices, separate_coo_eids,
        dummy_tensor, dummy_tensor, weights_transposed, node_feat, gradout,
        grad_node_feat, grad_weights);
  }
}

void RelationalMatMul_ACGatherScatterListIdentical_separatecoo(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_eids,
    at::Tensor& weights_transposed, at::Tensor& node_feat, at::Tensor& gradout,
    at::Tensor& grad_node_feat, at::Tensor& grad_weights,
    bool InputNumHeadOneFlag) {
  at::Tensor dummy_tensor;
  if (InputNumHeadOneFlag) {
    _BackwardRelationalMatMul_separatecoo<true, 16, false, false, true, true>(
        separate_coo_relptrs, dummy_tensor, separate_coo_eids, dummy_tensor,
        dummy_tensor, weights_transposed, node_feat, gradout, grad_node_feat,
        grad_weights);
  } else {
    _BackwardRelationalMatMul_separatecoo<true, 16, false, false, true, false>(
        separate_coo_relptrs, dummy_tensor, separate_coo_eids, dummy_tensor,
        dummy_tensor, weights_transposed, node_feat, gradout, grad_node_feat,
        grad_weights);
  }
}

void RelationalMatMulCompactAsOfNodeSingleEnded_unique_rel_node_indices(
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices,
    at::Tensor& separate_coo_rel_ptr, at::Tensor& separate_coo_node_indices,
    at::Tensor& separate_coo_eids, at::Tensor& weight_transposed,
    at::Tensor& node_feat, at::Tensor& ret, at::Tensor& gradout,
    at::Tensor& grad_weights, at::Tensor& grad_node_feat,
    bool InputNumHeadOneFlag) {
  at::Tensor dummy_tensor;
  if (InputNumHeadOneFlag) {
    _BackwardRelationalMatMul_separatecoo<true, 16, true, true, false, true>(
        separate_coo_rel_ptr, separate_coo_node_indices, separate_coo_eids,
        unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
        weight_transposed, node_feat, gradout, grad_node_feat, grad_weights);
  } else {
    _BackwardRelationalMatMul_separatecoo<true, 16, true, true, false, false>(
        separate_coo_rel_ptr, separate_coo_node_indices, separate_coo_eids,
        unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
        weight_transposed, node_feat, gradout, grad_node_feat, grad_weights);
  }
}

// NB: We may rely on HGTCompactAsOfNodesEdgeAttentionSecondStage in
// [[hetero_edgesoftmax/include/DGLHackKernel/HGT/HGTForwardKernels.cu.h]]
// adapted from _RelationalFusedGATKernel in
// hetero_edgesoftmax/include/DGLHackKernel/OpExport/RGATOps.inc.h
template </*int XPU, */ typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool IntegratedFormatRatherThanSeparateFlag,
          bool CSRRatherThanCOOFlag>
void inner_product_various_left_and_node_right(
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_rel_ptrs,
    at::Tensor& separate_coo_row_indices, at::Tensor& separate_coo_col_indices,
    at::Tensor& outcsr_row_ptr, at::Tensor& outcsr_col_idx,
    at::Tensor& outcsr_eids, at::Tensor& outcsr_reltypes,
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& feat_src,
    at::Tensor& feat_dst, at::Tensor& grad_inner_product,
    at::Tensor& grad_feat_src, at::Tensor& grad_feat_dst) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const Idx MAX_NBLKS = 65535;
  const Idx MAX_NTHRS = 1024;
  BackwardInnerProductData<Idx, DType> gdata;
  int64_t feat_src_xlen = SeastarComputeXLength<>(feat_src);
  gdata.feat_src = feat_src.data_ptr<DType>();
  gdata.feat_dst = feat_dst.data_ptr<DType>();

  gdata.grad_inner_product = grad_inner_product.data_ptr<DType>();
  gdata.grad_feat_src = grad_feat_src.data_ptr<DType>();
  gdata.grad_feat_dst = grad_feat_dst.data_ptr<DType>();

  gdata.num_heads = SeastarComputeXLength<>(grad_inner_product);  // num_heads
  gdata.feat_src_xlen = feat_src_xlen;

  if constexpr (IntegratedFormatRatherThanSeparateFlag &&
                CSRRatherThanCOOFlag) {
    // Integrated CSR
    gdata.eids = outcsr_eids.data_ptr<Idx>();
    // NB: updated to Type 2 Schedule:
    // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
    // head -> threadIdx.y
    // node -> blockIdx.y
    // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x
    int nthrs_y = SeastarFindNumThreads(gdata.num_heads, 64);
    int nthrs_x = SeastarFindNumThreads(gdata.feat_src_xlen / gdata.num_heads,
                                        MAX_NTHRS / nthrs_y);
    int nblks_x = 1;
    int64_t outcsr_num_rows = outcsr_row_ptr.numel() - 1;
    int nblks_y = std::min(outcsr_num_rows, MAX_NBLKS);
    const dim3 nthrs(nthrs_x, nthrs_y);
    const dim3 nblks(nblks_x, nblks_y);

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
    int nthrs_y = SeastarFindNumThreads(gdata.num_heads, 64);
    int nthrs_x = SeastarFindNumThreads(gdata.feat_src_xlen / gdata.num_heads,
                                        MAX_NTHRS / nthrs_y);
    int nblks_x = 1;
    int nblks_y = std::min(num_edges, MAX_NBLKS);
    int64_t outcsr_num_rows = outcsr_row_ptr.numel() - 1;
    const dim3 nthrs(nthrs_x, nthrs_y);
    const dim3 nblks(nblks_x, nblks_y);
    Idx* separate_coo_row_indices_data_ptr =
        separate_coo_row_indices.numel() > 0
            ? separate_coo_row_indices.data_ptr<Idx>()
            : nullptr;
    Idx* separate_coo_col_indices_data_ptr =
        separate_coo_col_indices.numel() > 0
            ? separate_coo_col_indices.data_ptr<Idx>()
            : nullptr;
    Idx* separate_coo_rel_ptrs_data_ptr =
        separate_coo_rel_ptrs.numel() > 0
            ? separate_coo_rel_ptrs.data_ptr<Idx>()
            : nullptr;
    Idx* unique_srcs_and_dests_rel_ptr_data_ptr =
        unique_srcs_and_dests_rel_ptr.numel() > 0
            ? unique_srcs_and_dests_rel_ptr.data_ptr<Idx>()
            : nullptr;
    Idx* unique_srcs_and_dests_node_indices_data_ptr =
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
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_idx,
    at::Tensor& separate_coo_rel_ptr, at::Tensor& separate_coo_eids,
    at::Tensor& separate_coo_row_indices, at::Tensor& separate_coo_col_indices,
    at::Tensor& left_node_compact_data,
    at::Tensor& right_node_vectors,  // at::Tensor& ret,
    at::Tensor& grad_inner_product, at::Tensor& grad_left_node_compact_data,
    at::Tensor& grad_right_node_vectors) {
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
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_row_indices,
    at::Tensor& separate_coo_col_indices, at::Tensor& left_edge_data,
    at::Tensor& right_node_vectors,
    at::Tensor& grad_inner_product,  // at::Tensor& gradout,
    at::Tensor& grad_left_edge_data, at::Tensor& grad_right_node_vectors) {
  at::Tensor dummy_tensor;
  inner_product_various_left_and_node_right<int64_t, float, false, false,
                                            false>(
      separate_coo_eids, dummy_tensor, separate_coo_row_indices,
      separate_coo_col_indices, dummy_tensor, dummy_tensor, dummy_tensor,
      dummy_tensor, dummy_tensor, dummy_tensor, left_edge_data,
      right_node_vectors, grad_inner_product, grad_left_edge_data,
      grad_right_node_vectors);
}

template <bool COARSEN_FACTOR_2_FLAG, int THREADING_BLOCK_SIZE>
void _RelationalMatmulNoScatterGatherList(at::Tensor& ntype_offset_ptrs,
                                          at::Tensor& weights_transposed,
                                          at::Tensor& node_feat_input,
                                          at::Tensor& grad_node_feat_output,
                                          at::Tensor& grad_weights,
                                          at::Tensor& grad_node_feat_input) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int64_t num_heads = 1;
  assert(weights_transposed.size(1) == 1 && "assertion n_head == 1 failed");
  const int64_t num_input_dim = weights_transposed.size(3);
  const int64_t num_output_dim =
      weights_transposed.size(2);  // weight shape (num_ntypes,
                                   // in_feat, out_feat)
  int64_t num_ntypes = ntype_offset_ptrs.numel() - 1;
  int64_t num_nodes = node_feat_input.size(0);

  constexpr int64_t WORK_BLOCK_SIZE =
      (COARSEN_FACTOR_2_FLAG ? (THREADING_BLOCK_SIZE * 2)
                             : THREADING_BLOCK_SIZE);

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
      get_schedule_by_relation_kernel_launch_metadata<false, false, int64_t*>(
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
  const dim3 nblks(ceil_div<>(num_input_dim, (long)WORK_BLOCK_SIZE), grid_dim_y,
                   num_heads);
  const dim3 nthrs(THREADING_BLOCK_SIZE, THREADING_BLOCK_SIZE);
  // std::cout << "nblks.x: " << nblks.x << " nblks.y: " << nblks.y
  //           << " nblks.z: " << nblks.z << std::endl;
  HET_RGNNMatmulNoScatterGatherListFwOrBwProp<
      COARSEN_FACTOR_2_FLAG, THREADING_BLOCK_SIZE, int64_t, int64_t*>
      <<<nblks, nthrs, 0, stream>>>(
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
      COARSEN_FACTOR_2_FLAG, THREADING_BLOCK_SIZE, int64_t, int64_t*>
      <<<nblks_outer_product, nthrs, 0, stream>>>(
          node_feat_input.data_ptr<float>(),
          grad_node_feat_output.data_ptr<float>(),
          grad_weights.data_ptr<float>(), ntype_offset_ptrs.data_ptr<int64_t>(),
          num_input_dim, num_output_dim,
          thrust::raw_pointer_cast(
              dev_num_blocks_assignment_for_all_prev_ntype_vect.data()),
          num_ntypes);
}

void RelationalMatmulNoScatterGatherList(at::Tensor& ntype_offset_ptrs,
                                         at::Tensor& weights_transposed,
                                         at::Tensor& node_feat_input,
                                         at::Tensor& grad_node_feat_output,
                                         at::Tensor& grad_weights,
                                         at::Tensor& grad_node_feat_input) {
  _RelationalMatmulNoScatterGatherList<true, 16>(
      ntype_offset_ptrs, weights_transposed, node_feat_input,
      grad_node_feat_output, grad_weights, grad_node_feat_input);
}

}  // namespace BckProp
}  // namespace RGNN
}  // namespace TorchExport
}  // namespace HET
