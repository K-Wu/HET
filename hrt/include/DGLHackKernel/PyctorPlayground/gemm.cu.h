#pragma once
#include <cuda_runtime.h>

#include "DGLHackKernel/RGNN/my_shmem_sgemm_func_functor.cu.h"
#include "DGLHackKernel/RGNN/my_shmem_sgemm_func_rgcn_hgt_functor.cu.h"
#include "kernel_enums.h"
#include "macros.h"
#include "utils.cu.h"

// Adapted from HET_RGNNFeatPerEdgeFwProp in
// hrt/include/DGLHackKernel/RGNN/my_shmem_sgemm_func.cu.h
template <
    bool RIGHT_REG_TILED_FLAG, int THREADING_BLOCK_SIZE_X,
    int THREADING_BLOCK_SIZE_Y, int WORK_BLOCK_SIZE_X, int WORK_BLOCK_SIZE_Y,
    int WORK_BLOCK_SIZE_K, typename Idx, typename IdxPtr,
    // TODO: check if InputNumHeadOneFlag could be removed
    bool InputNumHeadOneFlag /*whether (delta_)input_feat is single-headed*/>
__global__ void MY_SGEMM_LAUNCH_BOUNDS HET_PYCTOR_RGNNFeatPerEdgeFwProp(
    float *A, float *B, float *C, IdxPtr A_col_row_idx_gather_list,
    IdxPtr A_rel_ptr, IdxPtr C_eid_scatter_list, Idx num_A_cols, Idx num_B_cols,
    int num_heads, int *accum_num_blocks_per_relation, Idx num_relations) {
  // (input, weight, output) are 1, NH, NH or NH, NH, NH depending on whether
  // A_num_head_one_flag is true. NH is num_heads
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int *>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<RIGHT_REG_TILED_FLAG, /*double buffer flag*/ false,
                      THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y,
                      WORK_BLOCK_SIZE_X, WORK_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_K,
                      /*OuterProductFlag*/ false,
                      /*AGatherKind*/ MySGEMMGatherKind::Basic,
                      /*BGatherKind*/ MySGEMMGatherKind::Disabled,
                      /*CScatterKind*/ MySGEMMGatherKind::Basic,
                      /*AtomicUpdateFlag*/ false, Idx, IdxPtr,
                      /*numHeadKind*/ MySGEMMNumHeadKind::AssertANumIsOne,
                      /*compactKind*/ CompactAsOfNodeKind::Disabled>::
      execute_function(
          /*A*/ A,
          /*B*/
          &B[idx_relation * (InputNumHeadOneFlag ? num_heads : 1) * num_A_cols *
             num_B_cols],
          /*C*/ C, /*A_gather_list*/ A_col_row_idx_gather_list,
          /*B_gather_list*/ nullptr, /*C_scatter_list*/ C_eid_scatter_list,
          // TODO: remove etype_mapper_data as the two_order acccess
          // scheme is never used.
          /*etype_mapper_data*/ {}, idx_relation,
          /*numARows*/ A_rel_ptr[idx_relation + 1] - A_rel_ptr[idx_relation],
          /*blockIdxAlongRowBeg*/ accum_num_blocks_per_relation[idx_relation],
          /*strideNumBlocksAlongRow*/
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          /*blockRowJobEntryBeg*/ A_rel_ptr[idx_relation],
          /*num_A_cols*/ num_A_cols, /*num_B_cols*/ num_B_cols,
          /*num_heads*/ num_heads);
}

// Adapted from HET_HGTFusedAttnScoreFwProp in
// hrt/include/DGLHackKernel/RGNN/my_shmem_sgemm_func_rgcn_hgt.cu.h
template <bool RIGHT_REG_TILED_FLAG, int THREADING_BLOCK_SIZE_X,
          int THREADING_BLOCK_SIZE_Y, int SHMEM_BLOCK_SIZE_X,
          int SHMEM_BLOCK_SIZE_Y, int SHMEM_BLOCK_SIZE_K, typename Idx,
          typename IdxPtr>
__global__ void MY_SGEMM_LAUNCH_BOUNDS HET_PYCTOR_HGTFusedAttnScoreFwProp(
    float *A, float *B, float *C, float *inner_product,
    float *input_node_feat_for_inner_product, IdxPtr separate_coo_row_idx,
    IdxPtr separate_coo_col_idx, IdxPtr separate_coo_eids,
    IdxPtr separate_coo_rel_ptrs, int *accum_num_blocks_per_relation,
    Idx num_relations, Idx num_A_cols, Idx num_B_cols, int num_heads) {
  // TODO: KWU: supercede blockIdx/threadIdx with pretended blockIdx and
  // threadIdx if in a mega-kernel
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int *>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  // NB: should be mode 1 since we need to output inner product for bck prop use
  _simplified_basic_MatMulKernel<
      RIGHT_REG_TILED_FLAG, /*double buffer flag*/ false,
      THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y, SHMEM_BLOCK_SIZE_X,
      SHMEM_BLOCK_SIZE_Y, SHMEM_BLOCK_SIZE_K, Idx, IdxPtr,
      /*HGT_INSTEAD_OF_RGCN_FLAG_FOR_NUM_HEAD_ASSERTION*/ true,
      /*OuterProductFlag*/ false,
      /*DoInnerProductSwitch*/ MySGEMMInnerProductKind::Enabled,
      /*InnerProductGatherListNodeInsteadOfEdge*/ false,
      /*NoEdgeNormFlag*/ true,
      /*AtomicUpdateFlag*/ false>::
      execute_function(
          /*A*/ A,
          /*B*/
          &B[idx_relation * num_heads * num_B_cols * num_A_cols],
          /*C*/ C, /*edge_norm*/ nullptr,
          /*inner_product*/ inner_product,
          /*input_node_feat_for_inner_product*/
          input_node_feat_for_inner_product, separate_coo_row_idx,
          separate_coo_col_idx, separate_coo_eids, idx_relation,
          /*numARows*/ separate_coo_rel_ptrs[idx_relation + 1] -
              separate_coo_rel_ptrs[idx_relation],
          /*blockIdxAlongRowBeg*/ accum_num_blocks_per_relation[idx_relation],
          /*strideNumBlocksAlongRow*/
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          /*blockRowJobEntryBeg*/ separate_coo_rel_ptrs[idx_relation],
          /*num_A_cols*/ num_A_cols,
          /*num_B_cols*/ num_B_cols, num_heads);
}