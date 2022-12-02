#pragma once
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "DGLHackKernel/DGLHackUtils.h"
#include "DGLHackKernel/mysgemm/my_shmem_sgemm_func.cu.h"
#include "DGLHackKernel/mysgemm/mysgemm_KernelsBlockConfigurations.h"

namespace HET {
namespace TorchExport {
namespace RGNN {
namespace FwProp {
template <int BLOCK_SIZE, bool CompactAsOfNodeFlag,
          bool SingleSidedCompactAsOfNodeFlag,
          bool ACGatherScatterListIdenticalFlag, bool InputNumHeadOneFlag>
void _RelationalMatMul_separatecoo(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_node_indices,
    at::Tensor& separate_coo_eids, at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& weights,
    at::Tensor& input, at::Tensor& ret) {
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
      ceil_div<>(num_edges, (int64_t)BLOCK_SIZE),
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
    // std::cout << "nblks.x: " << nblks.x << " nblks.y: " << nblks.y
    //           << " nblks.z: " << nblks.z << std::endl;
    if constexpr (!SingleSidedCompactAsOfNodeFlag) {
      RGNNFeatCompactFWProp<BLOCK_SIZE, int64_t, int64_t*, InputNumHeadOneFlag>
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
      RGNNFeatCompactFWPropSingleSided<BLOCK_SIZE, int64_t, int64_t*,
                                       InputNumHeadOneFlag>
          <<<nblks, nthrs, 0, stream>>>(
              input.data_ptr<float>(), weights.data_ptr<float>(),
              ret.data_ptr<float>(),
              unique_srcs_and_dests_rel_ptr.data_ptr<int64_t>(),
              unique_srcs_and_dests_node_indices.data_ptr<int64_t>(),
              separate_coo_relptrs.data_ptr<int64_t>(),
              separate_coo_eids.data_ptr<int64_t>(), num_input_dim,
              num_output_per_head_dim, num_heads,
              thrust::raw_pointer_cast(
                  dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
              num_relations);
    }

  } else {
    const dim3 nblks(ceil_div<>(num_output_per_head_dim, (long)BLOCK_SIZE),
                     grid_dim_y, num_heads);
    const dim3 nthrs(BLOCK_SIZE, BLOCK_SIZE);
    // std::cout << "nblks.x: " << nblks.x << " nblks.y: " << nblks.y
    //           << " nblks.z: " << nblks.z << std::endl;
    if constexpr (ACGatherScatterListIdenticalFlag) {
      RGNNFeatPerEdgeFWPropACGatherScatterListIdentical<
          BLOCK_SIZE, int64_t, int64_t*, InputNumHeadOneFlag>
          <<<nblks, nthrs, 0, stream>>>(
              input.data_ptr<float>(), weights.data_ptr<float>(),
              ret.data_ptr<float>(), separate_coo_relptrs.data_ptr<int64_t>(),
              separate_coo_eids.data_ptr<int64_t>(), num_input_dim,
              num_output_per_head_dim, num_heads,
              thrust::raw_pointer_cast(
                  dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
              num_relations);
    } else {
      RGNNFeatPerEdgeFWProp<BLOCK_SIZE, int64_t, int64_t*, InputNumHeadOneFlag>
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

void RelationalMatMul_separatecoo(at::Tensor& separate_coo_relptrs,
                                  at::Tensor& separate_coo_node_indices,
                                  at::Tensor& separate_coo_eids,
                                  at::Tensor& weights, at::Tensor& input,
                                  at::Tensor& ret, bool InputNumHeadOneFlag) {
  at::Tensor dummy_tensor;
  if (InputNumHeadOneFlag) {
    _RelationalMatMul_separatecoo<16, false, false, false, true>(
        separate_coo_relptrs, separate_coo_node_indices, separate_coo_eids,
        dummy_tensor, dummy_tensor, weights, input, ret);
  } else {
    _RelationalMatMul_separatecoo<16, false, false, false, false>(
        separate_coo_relptrs, separate_coo_node_indices, separate_coo_eids,
        dummy_tensor, dummy_tensor, weights, input, ret);
  }
}

void RelationalMatMul_ACGatherScatterListIdentical_separatecoo(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_eids,
    at::Tensor& weights, at::Tensor& input, at::Tensor& ret,
    bool InputNumHeadOneFlag) {
  at::Tensor dummy_tensor;
  if (InputNumHeadOneFlag) {
    _RelationalMatMul_separatecoo<16, false, false, true, true>(
        separate_coo_relptrs, dummy_tensor, separate_coo_eids, dummy_tensor,
        dummy_tensor, weights, input, ret);
  } else {
    _RelationalMatMul_separatecoo<16, false, false, true, false>(
        separate_coo_relptrs, dummy_tensor, separate_coo_eids, dummy_tensor,
        dummy_tensor, weights, input, ret);
  }
}

void RelationalMatMulCompactAsOfNode_unique_rel_node_indices(
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices, at::Tensor& weight,
    at::Tensor& node_feat, at::Tensor& ret, bool InputNumHeadOneFlag) {
  at::Tensor dummy_tensor;
  if (InputNumHeadOneFlag) {
    _RelationalMatMul_separatecoo<16, true, false, false, true>(
        dummy_tensor, dummy_tensor, dummy_tensor, unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_indices, weight, node_feat, ret);
  } else {
    _RelationalMatMul_separatecoo<16, true, false, false, true>(
        dummy_tensor, dummy_tensor, dummy_tensor, unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_indices, weight, node_feat, ret);
  }
}

void RelationalMatMulCompactAsOfNodeSingleEnded_unique_rel_node_indices(
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_idx,
    at::Tensor& separate_coo_rel_ptr, at::Tensor& separate_coo_node_indices,
    at::Tensor& weight, at::Tensor& node_feat, at::Tensor& ret) {
  assert(0 && "Not implemented yet");
}

void inner_product_node_compact_and_node_separatecoo(
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_idx,
    at::Tensor& separate_coo_rel_ptr, at::Tensor& separate_coo_eids,
    at::Tensor& separate_coo_node_indices, at::Tensor& left_node_compact_data,
    at::Tensor& right_node_vectors, at::Tensor& ret) {
  // NB: We may refer to (edge parallel version)
  // HGTExpermentalEdgeAttentionConcatenatedSecondStageSrcInnerProductDestIntemediateCOOKernel
  // in hetero_edgesoftmax/include/EdgeSoftmax_4/EdgeSoftmaxMultiCOOs_4.h and
  // (node parallel version) _gatExpLeakyReluSumKernel in
  // hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGAT.cu.h
  assert(0 && "Not implemented yet");
}

void inner_product_edge_and_node_separatecoo(
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_node_indices,
    at::Tensor& left_edge_data, at::Tensor& right_node_vectors,
    at::Tensor& ret) {
  // NB: We may refer to (edge parallel version)
  // HGTExpermentalEdgeAttentionConcatenatedSecondStageSrcInnerProductDestIntemediateCOOKernel
  // in hetero_edgesoftmax/include/EdgeSoftmax_4/EdgeSoftmaxMultiCOOs_4.h and
  // (node parallel version) _gatExpLeakyReluSumKernel in
  // hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGAT.cu.h
  assert(0 && "Not implemented yet");
}

}  // namespace FwProp

namespace BckProp {
template <int BLOCK_SIZE, bool CompactAsOfNodeFlag,
          bool SingleSidedCompactAsOfNodeFlag,
          bool ACGatherScatterListIdenticalFlag, bool InputNumHeadOneFlag>
void _BackwardRelationalMatMul_separatecoo(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_node_indices,
    at::Tensor& separate_coo_eids, at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_indices,
    at::Tensor& weights_transposed, at::Tensor& input, at::Tensor& gradout,
    at::Tensor& grad_input, at::Tensor& grad_weights) {
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

  int64_t num_edges;

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
      ceil_div<>(num_edges, (int64_t)BLOCK_SIZE),
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
        ceil_div<>(num_output_per_head_dim, (long)BLOCK_SIZE),
        ceil_div<>(num_input_dim, (long)BLOCK_SIZE), num_heads * grid_dim_y);
    assert(num_head * grid_dim_y < 65535 && "num_head*grid_dim_y>=65535");

    const dim3 nthrs(BLOCK_SIZE, BLOCK_SIZE);
    // NB: #head of input is 1 when InputNumHeadOneFlag is true
    if constexpr (!SingleSidedCompactAsOfNodeFlag) {
      // cuda_err_chk(cudaGetLastError());
      RGNNDeltaNodeFeatInputCompactBWProp<BLOCK_SIZE, int64_t, int64_t*,
                                          InputNumHeadOneFlag>
          <<<nblks, nthrs, 0, stream>>>(
              gradout.data_ptr<float>(), weights_transposed.data_ptr<float>(),
              grad_input.data_ptr<float>(),
              unique_srcs_and_dests_rel_ptr.data_ptr<int64_t>(),
              unique_srcs_and_dests_node_indices.data_ptr<int64_t>(), num_edges,
              num_output_per_head_dim, num_input_dim, num_heads,
              thrust::raw_pointer_cast(
                  dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
              num_relations);
      // cuda_err_chk(cudaGetLastError());
      RGNNDeltaWeightCompactBWProp<BLOCK_SIZE, int64_t, int64_t*,
                                   InputNumHeadOneFlag>
          <<<nblks_outer_product, nthrs, 0, stream>>>(
              gradout.data_ptr<float>(), input.data_ptr<float>(),
              grad_weights.data_ptr<float>(),
              unique_srcs_and_dests_rel_ptr.data_ptr<int64_t>(),
              unique_srcs_and_dests_node_indices.data_ptr<int64_t>(), num_edges,
              num_output_per_head_dim, num_input_dim, num_heads,
              thrust::raw_pointer_cast(
                  dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
              num_relations);
      // cuda_err_chk(cudaGetLastError());
    } else {
      RGNNDeltaNodeFeatInputCompactBWPropSingleSided<
          BLOCK_SIZE, int64_t, int64_t*, InputNumHeadOneFlag>
          <<<nblks, nthrs, 0, stream>>>(
              gradout.data_ptr<float>(), weights_transposed.data_ptr<float>(),
              grad_input.data_ptr<float>(),
              unique_srcs_and_dests_rel_ptr.data_ptr<int64_t>(),
              unique_srcs_and_dests_node_indices.data_ptr<int64_t>(),
              separate_coo_relptrs.data_ptr<int64_t>(),
              separate_coo_eids.data_ptr<int64_t>(), num_edges,
              num_output_per_head_dim, num_input_dim, num_heads,
              thrust::raw_pointer_cast(
                  dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
              num_relations);
      RGNNDeltaWeightCompactBWPropSingleSided<BLOCK_SIZE, int64_t, int64_t*,
                                              InputNumHeadOneFlag>
          <<<nblks_outer_product, nthrs, 0, stream>>>(
              gradout.data_ptr<float>(), input.data_ptr<float>(),
              grad_weights.data_ptr<float>(),
              unique_srcs_and_dests_rel_ptr.data_ptr<int64_t>(),
              unique_srcs_and_dests_node_indices.data_ptr<int64_t>(),
              separate_coo_relptrs.data_ptr<int64_t>(),
              separate_coo_eids.data_ptr<int64_t>(), num_edges,
              num_output_per_head_dim, num_input_dim, num_heads,
              thrust::raw_pointer_cast(
                  dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
              num_relations);
    }
  } else {
    const dim3 nblks(ceil_div<>(num_output_per_head_dim, (long)BLOCK_SIZE),
                     grid_dim_y, num_heads);
    const dim3 nblks_outer_product(
        ceil_div<>(num_output_per_head_dim, (long)BLOCK_SIZE),
        ceil_div<>(num_input_dim, (long)BLOCK_SIZE), num_heads * grid_dim_y);
    assert(num_head * grid_dim_y < 65535 && "num_head*grid_dim_y>=65535");
    const dim3 nthrs(BLOCK_SIZE, BLOCK_SIZE);
    // NB: #head of input is 1 when InputNumHeadOneFlag is true
    if constexpr (ACGatherScatterListIdenticalFlag) {
      RGNNDeltaNodeFeatInputBWPropACGatherScatterListIdentical<
          BLOCK_SIZE, int64_t, int64_t*, InputNumHeadOneFlag>
          <<<nblks, nthrs, 0, stream>>>(
              gradout.data_ptr<float>(), weights_transposed.data_ptr<float>(),
              grad_input.data_ptr<float>(),
              separate_coo_eids.data_ptr<int64_t>(),
              separate_coo_relptrs.data_ptr<int64_t>(), num_output_per_head_dim,
              num_input_dim, num_heads,
              thrust::raw_pointer_cast(
                  dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
              num_relations);
      // FIXME: outer_product should assume num_heads == 1 and change blockDim.x
      // and/or blockDim.y accordingly
      // FIXME: outer_product should have num_heads == 1 and num_heads for input
      // and output respectively, and therefore distinction should be made when
      // get row major from A and B
      RGNNDeltaWeightBWPropACGatherScatterListIdentical<
          BLOCK_SIZE, int64_t, int64_t*, InputNumHeadOneFlag>
          <<<nblks_outer_product, nthrs, 0, stream>>>(
              input.data_ptr<float>(), gradout.data_ptr<float>(),
              grad_weights.data_ptr<float>(),
              separate_coo_eids.data_ptr<int64_t>(),
              separate_coo_relptrs.data_ptr<int64_t>(), num_input_dim,
              num_output_per_head_dim, num_heads,
              thrust::raw_pointer_cast(
                  dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
              num_relations);
    } else {
      RGNNDeltaNodeFeatInputBWProp<BLOCK_SIZE, int64_t, int64_t*,
                                   InputNumHeadOneFlag>
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
      RGNNDeltaWeightBWProp<BLOCK_SIZE, int64_t, int64_t*, InputNumHeadOneFlag>
          <<<nblks_outer_product, nthrs, 0, stream>>>(
              input.data_ptr<float>(), gradout.data_ptr<float>(),
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
    at::Tensor& weights_transposed, at::Tensor& input, at::Tensor& gradout,
    at::Tensor& grad_input, at::Tensor& grad_weights,
    bool InputNumHeadOneFlag) {
  at::Tensor dummy_tensor;
  if (InputNumHeadOneFlag) {
    _BackwardRelationalMatMul_separatecoo<16, true, false, false, true>(
        dummy_tensor, dummy_tensor, dummy_tensor, unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_indices, weights_transposed, input, gradout,
        grad_input, grad_weights);
  } else {
    _BackwardRelationalMatMul_separatecoo<16, true, false, false, false>(
        dummy_tensor, dummy_tensor, dummy_tensor, unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_indices, weights_transposed, input, gradout,
        grad_input, grad_weights);
  }
}

void RelationalMatMul_separatecoo(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_node_indices,
    at::Tensor& separate_coo_eids, at::Tensor& weights_transposed,
    at::Tensor& input, at::Tensor& gradout, at::Tensor& grad_input,
    at::Tensor& grad_weights, bool InputNumHeadOneFlag) {
  at::Tensor dummy_tensor;
  if (InputNumHeadOneFlag) {
    _BackwardRelationalMatMul_separatecoo<16, false, false, false, true>(
        separate_coo_relptrs, separate_coo_node_indices, separate_coo_eids,
        dummy_tensor, dummy_tensor, weights_transposed, input, gradout,
        grad_input, grad_weights);
  } else {
    _BackwardRelationalMatMul_separatecoo<16, false, false, false, false>(
        separate_coo_relptrs, separate_coo_node_indices, separate_coo_eids,
        dummy_tensor, dummy_tensor, weights_transposed, input, gradout,
        grad_input, grad_weights);
  }
}

void RelationalMatMul_ACGatherScatterListIdentical_separatecoo(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_eids,
    at::Tensor& weights_transposed, at::Tensor& input, at::Tensor& gradout,
    at::Tensor& grad_input, at::Tensor& grad_weights,
    bool InputNumHeadOneFlag) {
  at::Tensor dummy_tensor;
  if (InputNumHeadOneFlag) {
    _BackwardRelationalMatMul_separatecoo<16, false, false, true, true>(
        separate_coo_relptrs, dummy_tensor, separate_coo_eids, dummy_tensor,
        dummy_tensor, weights_transposed, input, gradout, grad_input,
        grad_weights);
  } else {
    _BackwardRelationalMatMul_separatecoo<16, false, false, true, false>(
        separate_coo_relptrs, dummy_tensor, separate_coo_eids, dummy_tensor,
        dummy_tensor, weights_transposed, input, gradout, grad_input,
        grad_weights);
  }
}

void RelationalMatMulCompactAsOfNodeSingleEnded_unique_rel_node_indices(
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_idx,
    at::Tensor& separate_coo_rel_ptr, at::Tensor& separate_coo_node_indices,
    at::Tensor& weight_transposed, at::Tensor& node_feat, at::Tensor& ret,
    at::Tensor& gradout, at::Tensor& grad_weight, at::Tensor& grad_node_feat) {
  assert(0 && "Not implemented yet");
}

void inner_product_node_compact_and_node_separatecoo(
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_idx,
    at::Tensor& separate_coo_rel_ptr, at::Tensor& separate_coo_eids,
    at::Tensor& separate_coo_node_indices, at::Tensor& left_node_compact_data,
    at::Tensor& right_node_vectors, at::Tensor& ret, at::Tensor& gradout,
    at::Tensor& grad_left_node_compact_data,
    at::Tensor& grad_right_node_vectors) {
  // We may rely on HGTCompactAsOfNodesEdgeAttentionSecondStage in
  // [[hetero_edgesoftmax/include/DGLHackKernel/HGT/HGTForwardKernels.cu.h]]
  assert(0 && "Not implemented yet");
}

void inner_product_edge_and_node_separatecoo(
    at::Tensor& separate_coo_eids, at::Tensor& separate_coo_node_indices,
    at::Tensor& left_edge_data, at::Tensor& right_node_vectors, at::Tensor& ret,
    at::Tensor& gradout, at::Tensor& grad_left_edge_data,
    at::Tensor& grad_right_node_vectors) {
  assert(0 && "Not implemented yet");
}

}  // namespace BckProp
}  // namespace RGNN
}  // namespace TorchExport
}  // namespace HET
