#pragma once
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "DGLHackKernel/DGLHackUtils.h"
#include "DGLHackKernel/PyctorPlayground/enumerate.cu.h"
#include "DGLHackKernel/PyctorPlayground/gemm.cu.h"
#include "DGLHackKernel/RGNN/InnerProductEdgeParallel.cu.h"
#include "DGLHackKernel/RGNN/my_shmem_sgemm_func.cu.h"
#include "DGLHackKernel/RGNN/my_shmem_sgemm_func_rgcn_hgt.cu.h"
#include "DGLHackKernel/RGNN/mysgemm_KernelsBlockConfigurations.h"
#include "ThreadingGridsBlocksSchedules.h"
#include "macros.h"

namespace HET {
namespace TorchExport {
namespace PYCTORPLAYGROUND {
// matmul_rgcn_hgt launcher example adapted from
// HGT::FwProp::SeparateCOO::EdgeParallel::FullGraphFusedMessageCalcAndMeanAggregation
void full_graph_hetero_attention_ops(
    torch::Dict<std::string, at::Tensor> graph_tensors_dict,
    at::Tensor &applied_klinear_node_features,
    at::Tensor &applied_qlinear_node_features, at::Tensor &attn_score_weight,
    at::Tensor &attn_score_inner_product, at::Tensor &unnormalized_attn_score) {
  // we need to implement a fused kernel based on W*t via RGNN relational_matmul
  // and RGNN inner_product
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  // TODO: KWU: implement the switch to disable reg-tiling
  constexpr bool REG_TILING_FLAG = true;

  const int64_t num_relations =
      (graph_tensors_dict.at("separate_coo_relptrs").numel() - 1);

  /// PYCTOR: Dimension calculation
  // PYCTOR: in some cases num_heads is fixed to 1
  const int64_t num_heads = attn_score_weight.size(1);
  const int64_t num_input_dim = attn_score_weight.size(2);
  const int64_t num_output_dim = attn_score_weight.size(3);
  int64_t num_edges = graph_tensors_dict.at("separate_coo_eids").numel();

  /// PYCTOR: Kernel configuration
  MY_SGEMM_GRID_CONFIG()

  int grid_dim_y = std::min(ceil_div<>(num_edges, (int64_t)WORK_BLOCK_SIZE_Y),
                            (int64_t)4096);
  at::Tensor separate_coo_relptrs_cpu_contiguous =
      graph_tensors_dict.at("separate_coo_relptrs").cpu().contiguous();
  std::vector<int> num_blocks_assignment_for_same_relation_vect,
      num_blocks_assignment_for_all_prev_relation_vect;
  std::tie(num_blocks_assignment_for_same_relation_vect,
           num_blocks_assignment_for_all_prev_relation_vect) =
      get_schedule_by_relation_kernel_launch_metadata<false, false, int64_t *>(
          grid_dim_y, num_relations, WORK_BLOCK_SIZE_Y,
          separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>(),
          separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>() +
              num_relations + 1);
  grid_dim_y = num_blocks_assignment_for_all_prev_relation_vect.back();

  thrust::device_vector<int>
      dev_num_blocks_assignment_for_all_prev_relation_vect(
          num_blocks_assignment_for_all_prev_relation_vect.begin(),
          num_blocks_assignment_for_all_prev_relation_vect.end());

  /// PYCTOR: Grid configuration and Kernel launch
  // NB: my shmem sgemm matmul scheme
  const dim3 nblks(ceil_div<>(num_output_dim, (long)WORK_BLOCK_SIZE_X),
                   grid_dim_y, num_heads);
  const dim3 nthrs(THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y);

  // NB: KWU: using reg tiled version by default
  HET_PYCTOR_HGTFusedAttnScoreFwProp<REG_TILING_FLAG, THREADING_BLOCK_SIZE_X,
                                     THREADING_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_X,
                                     WORK_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_K,
                                     int64_t, int64_t *>
      <<<nblks, nthrs, 0, stream>>>(
          applied_klinear_node_features.data_ptr<float>(),
          attn_score_weight.data_ptr<float>(),
          attn_score_inner_product.data_ptr<float>(),
          unnormalized_attn_score.data_ptr<float>(),
          applied_qlinear_node_features.data_ptr<float>(),
          graph_tensors_dict.at("separate_coo_row_indices").data_ptr<int64_t>(),
          graph_tensors_dict.at("separate_coo_col_indices").data_ptr<int64_t>(),
          graph_tensors_dict.at("separate_coo_eids").data_ptr<int64_t>(),
          graph_tensors_dict.at("separate_coo_relptrs").data_ptr<int64_t>(),
          thrust::raw_pointer_cast(
              dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
          num_relations, num_input_dim, num_output_dim, num_heads);
}

// matmul launcher example adapted from
// RGNN::FwProp::SeparateCOO::_RelationalMatMul in
// hrt/include/DGLHackKernel/OpExport/RGNNOps.inc.h
template <CompactAsOfNodeKind kind, bool ACGatherScatterListIdenticalFlag,
          bool InputNumHeadOneFlag>
void _RelationalMatMul(torch::Dict<std::string, at::Tensor> graph_tensors_dict,
                       at::Tensor &Amat, at::Tensor &Bmat, at::Tensor &ret) {
  /// constants to be emitted by pyctor

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  /// PYCTOR: Dimension calculation
  const int64_t num_relations =
      graph_tensors_dict.at("separate_coo_relptrs").numel() == 0
          ? (graph_tensors_dict.at("unique_srcs_and_dests_rel_ptrs").numel() -
             1)
          : (graph_tensors_dict.at("separate_coo_relptrs").numel() - 1);
  const int64_t num_heads = Bmat.size(1);
  const int64_t num_input_dim = Bmat.size(2);
  const int64_t num_output_per_head_dim =
      Bmat.size(3);  // weight shape (num_relations, n_heads,
                     // in_feat, out_feat // n_heads)
  int64_t num_edges;

  /// PYCTOR: Kernel configuration
  // TODO: KWU: add reg-tiled specific configurations by introducing tenary
  // operators

  // NB: configuration specific to shmem-tiled sgemm

  // assuming coarsening in both x and y direction if shmem is used instead of
  // reg tiling
  // TODO: KWU: enable reg tiling for compact as of node
  constexpr bool REG_TILING_FLAG = true;

  MY_SGEMM_GRID_CONFIG()

  if constexpr (IsCompact(kind)) {
    num_edges =
        graph_tensors_dict.at("unique_srcs_and_dests_node_indices").numel();
  } else {
    num_edges = graph_tensors_dict.at("separate_coo_eids").numel();
  }
  int grid_dim_y = std::min(
      ceil_div<>(num_edges, (int64_t)WORK_BLOCK_SIZE_Y),
      (int64_t)32768);  // using 32768 instead of 65535 to leave some space in
                        // case the total number of blocks is slightly larger
                        // due to relationship with very few workloads
  std::vector<int> num_blocks_assignment_for_same_relation_vect,
      num_blocks_assignment_for_all_prev_relation_vect;
  if constexpr (IsCompact(kind)) {
    at::Tensor unique_srcs_and_dests_rel_ptrs_cpu_contiguous =
        graph_tensors_dict.at("unique_srcs_and_dests_rel_ptrs")
            .cpu()
            .contiguous();
    std::tie(num_blocks_assignment_for_same_relation_vect,
             num_blocks_assignment_for_all_prev_relation_vect) =
        get_schedule_by_relation_kernel_launch_metadata<false, false,
                                                        int64_t *>(
            grid_dim_y, num_relations, WORK_BLOCK_SIZE_Y,
            unique_srcs_and_dests_rel_ptrs_cpu_contiguous.data_ptr<int64_t>(),
            unique_srcs_and_dests_rel_ptrs_cpu_contiguous.data_ptr<int64_t>() +
                num_relations + 1);
  } else {
    at::Tensor separate_coo_relptrs_cpu_contiguous =
        graph_tensors_dict.at("separate_coo_relptrs").cpu().contiguous();
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

  /// PYCTOR: Grid configuration and Kernel launch
  if constexpr (IsCompact(kind)) {
    if constexpr (ACGatherScatterListIdenticalFlag) {
      CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
          IsCompact(kind) && ACGatherScatterListIdenticalFlag,
          "CompactAsOfNodeFlag && ACGatherScatterListIdenticalFlag");
    }

    // NB: my shmem sgemm matmul scheme
    const dim3 nblks(
        ceil_div<>(num_output_per_head_dim, (long)WORK_BLOCK_SIZE_X),
        grid_dim_y, num_heads);
    const dim3 nthrs(THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y);
    // TODO: KWU: allow more dtype options in this file
    ETypeMapperData<int64_t, kind> etype_mapper_data{
        .unique_srcs_and_dests_rel_ptrs =
            graph_tensors_dict.at("unique_srcs_and_dests_rel_ptrs")
                .data_ptr<int64_t>(),
        .unique_srcs_and_dests_node_indices =
            graph_tensors_dict.at("unique_srcs_and_dests_node_indices")
                .data_ptr<int64_t>()};
    HET_RGNNFeatCompactFwProp<REG_TILING_FLAG, THREADING_BLOCK_SIZE_X,
                              THREADING_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_X,
                              WORK_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_K, int64_t,
                              int64_t *, InputNumHeadOneFlag>
        <<<nblks, nthrs, 0, stream>>>(
            Amat.data_ptr<float>(), Bmat.data_ptr<float>(),
            ret.data_ptr<float>(), etype_mapper_data, num_input_dim,
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
      HET_RGNNFeatPerEdgeFwPropACGatherScatterListIdentical<
          REG_TILING_FLAG, THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y,
          WORK_BLOCK_SIZE_X, WORK_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_K, int64_t,
          int64_t *, InputNumHeadOneFlag><<<nblks, nthrs, 0, stream>>>(
          Amat.data_ptr<float>(), Bmat.data_ptr<float>(), ret.data_ptr<float>(),
          graph_tensors_dict.at("separate_coo_relptrs").data_ptr<int64_t>(),
          graph_tensors_dict.at("separate_coo_eids").data_ptr<int64_t>(),
          num_input_dim, num_output_per_head_dim, num_heads,
          thrust::raw_pointer_cast(
              dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
          num_relations);
    } else {
      // NB: KWU: use by default the new reg tiled version here
      HET_PYCTOR_RGNNFeatPerEdgeFwProp<
          REG_TILING_FLAG, THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y,
          WORK_BLOCK_SIZE_X, WORK_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_K, int64_t,
          int64_t *, InputNumHeadOneFlag><<<nblks, nthrs, 0, stream>>>(
          Amat.data_ptr<float>(), Bmat.data_ptr<float>(), ret.data_ptr<float>(),
          graph_tensors_dict.at("separate_coo_node_indices")
              .data_ptr<int64_t>(),
          graph_tensors_dict.at("separate_coo_relptrs").data_ptr<int64_t>(),
          graph_tensors_dict.at("separate_coo_eids").data_ptr<int64_t>(),
          num_input_dim, num_output_per_head_dim, num_heads,
          thrust::raw_pointer_cast(
              dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
          num_relations);
    }
  }
}

// We must input the constexpr variables through template arguments in order to
// conditionally compile the if constexpr branches. Reference:
// https://stackoverflow.com/a/63469419/5555077
void RelationalMatMul(torch::Dict<std::string, at::Tensor> graph_tensors_dict,
                      at::Tensor &Bmat, at::Tensor &Amat, at::Tensor &ret) {
  constexpr CompactAsOfNodeKind kind = CompactAsOfNodeKind::Disabled;
  constexpr bool ACGatherScatterListIdenticalFlag = false;
  constexpr bool InputNumHeadOneFlag = false;
  _RelationalMatMul<kind, ACGatherScatterListIdenticalFlag,
                    InputNumHeadOneFlag>(graph_tensors_dict, Amat, Bmat, ret);
}

// Adapted from
// HET::TorchExport::RGNN::FwProp::_InnerProductVariousLeftAndNodeRight in
// hrt/include/DGLHackKernel/OpExport/RGNNOps.inc.h
template </*int XPU, */ typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool IntegratedFormatRatherThanSeparateFlag,
          bool CSRRatherThanCOOFlag>
void _InnerProductVariousLeftAndNodeRight(
    torch::Dict<std::string, at::Tensor> graph_tensors_dict,
    at::Tensor &feat_src, at::Tensor &feat_dst,
    at::Tensor &edge_inner_product) {
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

  PYCTORInnerProductData<Idx, DType> gdata_pyctor{
      .feat_src_xlen = SeastarComputeXLength<>(feat_src),
      .num_heads = SeastarComputeXLength<>(edge_inner_product),
      .eids = nullptr,  // assign later in if branches
      .feat_src = feat_src.data_ptr<DType>(),
      .feat_dst = feat_dst.data_ptr<DType>(),
      .edge_inner_product = edge_inner_product.data_ptr<DType>()};

  if constexpr (IntegratedFormatRatherThanSeparateFlag &&
                CSRRatherThanCOOFlag) {
    // Integrated CSR
    gdata.eids = graph_tensors_dict.at("incsr_eids").data_ptr<Idx>();
    // Configure kernel launch parameters.

    // NB: updated to Type 2 Schedule:
    // https://github.com/K-Wu/HET/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
    // head -> threadIdx.y
    // node -> blockIdx.y
    // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x

    int64_t incsr_num_rows = graph_tensors_dict.at("incsr_row_ptr").numel() - 1;
    auto [nblks2, nthrs2] = get_type2_schedule(
        gdata.num_heads, gdata.feat_src_xlen, incsr_num_rows);

    Idx *incsr_col_indices_data_ptr =
        graph_tensors_dict.at("incsr_col_indices").numel() > 0
            ? graph_tensors_dict.at("incsr_col_indices").data_ptr<Idx>()
            : nullptr;
    Idx *incsr_reltypes_data_ptr =
        graph_tensors_dict.at("incsr_reltypes").numel() > 0
            ? graph_tensors_dict.at("incsr_reltypes").data_ptr<Idx>()
            : nullptr;

    ETypeMapperData<Idx, kind> etype_mapper_data;

    if constexpr (kind == CompactAsOfNodeKind::EnabledWithDirectIndexing) {
      assert(graph_tensors_dict.at("edata_idx_to_inverse_idx").numel() > 0);
      etype_mapper_data.edata_idx_to_inverse_idx =
          graph_tensors_dict.at("edata_idx_to_inverse_idx").data_ptr<Idx>();
    } else if constexpr (kind == CompactAsOfNodeKind::Enabled) {
      assert(graph_tensors_dict.at("unique_srcs_and_dests_rel_ptrs").numel() >
             0);
      assert(
          graph_tensors_dict.at("unique_srcs_and_dests_node_indices").numel() >
          0);
      etype_mapper_data.unique_srcs_and_dests_rel_ptrs =
          graph_tensors_dict.at("unique_srcs_and_dests_rel_ptrs")
              .data_ptr<Idx>();
      etype_mapper_data.unique_srcs_and_dests_node_indices =
          graph_tensors_dict.at("unique_srcs_and_dests_node_indices")
              .data_ptr<Idx>();
    } else {
      assert(kind == CompactAsOfNodeKind::Disabled);
    }

    ETypeData<Idx, false> etype_data{
        .etypes = graph_tensors_dict.at("incsr_reltypes").numel() > 0
                      ? graph_tensors_dict.at("incsr_reltypes").data_ptr<Idx>()
                      : nullptr};

    HET_inner_product_fw_kernel<Idx, DType, kind, true, false>
        <<<nblks2, nthrs2, 0, stream>>>(
            gdata, etype_data, incsr_col_indices_data_ptr,
            incsr_reltypes_data_ptr, incsr_num_rows, etype_mapper_data);
  } else if constexpr (!IntegratedFormatRatherThanSeparateFlag &&
                       !CSRRatherThanCOOFlag) {
    // separate coo
    gdata.eids = graph_tensors_dict.at("separate_coo_eids").data_ptr<Idx>();
    int64_t num_edges =
        graph_tensors_dict.at("separate_coo_row_indices").numel();
    int64_t num_relations =
        graph_tensors_dict.at("separate_coo_rel_ptrs").numel() - 1;

    // NB: Type 2 Schedule:
    // https://github.com/K-Wu/HET/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
    // head -> threadIdx.y
    // edge -> blockIdx.y
    // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x
    // threadIdx.x and threadIdx.y and only this pair is exchanged compared with
    // original seastar schedule to allow reduction within the warp, i.e., along
    // x-axis
    auto [nblks_inner_product, nthrs_inner_product] =
        get_type2_schedule(gdata.num_heads, gdata.feat_src_xlen, num_edges);
    Idx *separate_coo_row_indices_data_ptr =
        graph_tensors_dict.at("separate_coo_row_indices").numel() > 0
            ? graph_tensors_dict.at("separate_coo_row_indices").data_ptr<Idx>()
            : nullptr;
    Idx *separate_coo_col_indices_data_ptr =
        graph_tensors_dict.at("separate_coo_col_indices").numel() > 0
            ? graph_tensors_dict.at("separate_coo_col_indices").data_ptr<Idx>()
            : nullptr;
    ETypeMapperData<Idx, kind> etype_mapper_data;
    ETypeData<Idx, true> etype_data{
        .etypes =
            graph_tensors_dict.at("separate_coo_rel_ptrs").numel() > 0
                ? graph_tensors_dict.at("separate_coo_rel_ptrs").data_ptr<Idx>()
                : nullptr,
        .num_relations = num_relations,
    };

    if constexpr (IsCompact(kind)) {
      if constexpr (IsBinarySearch(kind)) {
        assert(graph_tensors_dict.at("unique_srcs_and_dests_rel_ptrs").numel() >
               0);
        assert(graph_tensors_dict.at("unique_srcs_and_dests_node_indices")
                   .numel() > 0);
        etype_mapper_data.unique_srcs_and_dests_rel_ptrs =
            graph_tensors_dict.at("unique_srcs_and_dests_rel_ptrs")
                .data_ptr<Idx>();
        etype_mapper_data.unique_srcs_and_dests_node_indices =
            graph_tensors_dict.at("unique_srcs_and_dests_node_indices")
                .data_ptr<Idx>();
      } else {
        assert(graph_tensors_dict.at("edata_idx_to_inverse_idx").numel() > 0);
        etype_mapper_data.edata_idx_to_inverse_idx =
            graph_tensors_dict.at("edata_idx_to_inverse_idx").data_ptr<Idx>();
      }
    } else {
      assert(kind == CompactAsOfNodeKind::Disabled);
    }
    HET_PYCTOR_inner_product_fw_kernel_edge_parallel<Idx, DType, kind, true>
        <<<nblks_inner_product, nthrs_inner_product, 0, stream>>>(
            gdata_pyctor, separate_coo_row_indices_data_ptr,
            separate_coo_col_indices_data_ptr, etype_data, num_edges,
            etype_mapper_data);

  } else {
    assert(0 && "Not implemented");
  }
}

void InnerProductRightNode(
    torch::Dict<std::string, at::Tensor> graph_tensors_dict, int64_t IntKind,
    at::Tensor &left_side_data, at::Tensor &right_node_vectors,
    at::Tensor &edge_inner_product) {
  cudaMemsetAsync(edge_inner_product.data_ptr<float>(), 0,
                  edge_inner_product.numel() * sizeof(float),
                  c10::cuda::getCurrentCUDAStream());
  at::Tensor dummy_tensor;
  auto Kind = static_cast<CompactAsOfNodeKind>(IntKind);
  if (Kind == CompactAsOfNodeKind::EnabledWithDirectIndexing) {
    _InnerProductVariousLeftAndNodeRight<
        int64_t, float, CompactAsOfNodeKind::EnabledWithDirectIndexing, false,
        false>(graph_tensors_dict, left_side_data, right_node_vectors,
               edge_inner_product);
  } else if (Kind == CompactAsOfNodeKind::Enabled) {
    // originally inner_product_node_compact_and_node_separatecoo
    // left_side_data is left_node_compact_data
    _InnerProductVariousLeftAndNodeRight<
        int64_t, float, CompactAsOfNodeKind::Enabled, false, false>(
        graph_tensors_dict, left_side_data, right_node_vectors,
        edge_inner_product);
  } else if (Kind == CompactAsOfNodeKind::Disabled) {
    // originally inner_product_edge_and_node_separatecoo
    // left_side_data is left_edge_data
    _InnerProductVariousLeftAndNodeRight<
        int64_t, float, CompactAsOfNodeKind::Disabled, false, false>(
        graph_tensors_dict, left_side_data, right_node_vectors,
        edge_inner_product);
  } else {
    assert(0 && "Not implemented");
  }
}

}  // namespace PYCTORPLAYGROUND
}  // namespace TorchExport
}  // namespace HET

using namespace HET::TorchExport;
TORCH_LIBRARY_FRAGMENT(torch_hrt, m) {
  m.def("pyctor_playground_full_graph_hetero_attention_ops",
        PYCTORPLAYGROUND::full_graph_hetero_attention_ops);
  m.def("pyctor_playground_relational_matmul",
        PYCTORPLAYGROUND::RelationalMatMul);
  m.def("pyctor_playground_inner_product_right_node",
        PYCTORPLAYGROUND::InnerProductRightNode);
}
