#pragma once

#include <cuda_runtime.h>

#include "DGLHackKernel/GAT/FusedGAT.cu.h"
#include "kernel_enums.h"

// edge-centric schedule cf. HET_gatSumProdZipDivKernel in
// [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGAT.cu.h]]
template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag, bool FullCartesianFlag>
__device__ __forceinline__ void _gatSumProdZipDivKernel_edge_parallel(
    GatFusedData<Idx, DType> gdata, const Idx *etypes, const Idx *row_indices,
    const Idx *col_indices, int64_t num_edges,
    const ETypeMapperData<Idx, kind> etype_mapper_data,
    const ETypeMapperData<Idx, kind> etype_mapper_data_col,
    int64_t num_relations) {
  constexpr bool EtypeRelPtrIndexSearch = true;
  Idx resume_from = 0;

  constexpr bool ETypeRelPtrFlag = true;
  constexpr bool CompactAsOfNodeFlag = IsCompact(kind);
  constexpr bool DualUniqueNodeList = IsCompactWithDualList(kind);
  Idx num_heads = gdata.num_heads;
  Idx hidden_xlen = gdata.feat_src_xlen / num_heads;
  for (Idx eidx = blockIdx.y; eidx < num_edges; eidx += gridDim.y) {
    Idx dst_vid = col_indices[eidx];

    Idx src_vid = row_indices[eidx];
    for (Idx head_idx = threadIdx.y; head_idx < num_heads;
         head_idx += blockDim.y) {
      for (Idx feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
           feat_idx < hidden_xlen; feat_idx += blockDim.x * gridDim.x) {
        DType s = 0.;
        Idx feat_src_entry_id = -1;
        Idx edata_idx = gdata.eids[eidx];
        if constexpr (RelationalFlag) {
          Idx etype = -1;
          if constexpr (ETypeRelPtrFlag) {
            if constexpr (EtypeRelPtrIndexSearch) {
              etype = linear_search(num_relations, etypes, eidx, resume_from);
              resume_from = etype;
            } else {
              etype = binary_search(num_relations, etypes, eidx);
            }

          } else {
            etype = etypes[eidx];
          }
          if constexpr (CompactAsOfNodeFlag) {
            feat_src_entry_id = find_relational_compact_as_of_node_index(
                etype, src_vid, edata_idx, etype_mapper_data);
            if constexpr (FullCartesianFlag) {
              // NB: This is the case where we have the data stored in
              // (relation, node) but do not compress the (relation, node)
              // matrix. It could be a case in subgraph where compressing along
              // the node dimension may not be worth it.
              CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
                  (FullCartesianFlag) && CompactAsOfNodeFlag,
                  "should be non-reachable not implemented");
            }
          } else {
            feat_src_entry_id = edata_idx;
          }
          s += (gdata.exp[edata_idx * num_heads + head_idx] /
                gdata.sum[dst_vid * num_heads + head_idx] *
                gdata.feat_src[feat_src_entry_id * gdata.feat_src_xlen +
                               head_idx * hidden_xlen + feat_idx]);
          atomicAdd(&gdata.ret[dst_vid * gdata.feat_src_xlen +
                               head_idx * hidden_xlen + feat_idx],
                    gdata.exp[edata_idx * num_heads + head_idx] /
                        gdata.sum[dst_vid * num_heads + head_idx] *
                        gdata.feat_src[feat_src_entry_id * gdata.feat_src_xlen +
                                       head_idx * hidden_xlen + feat_idx]);
        } else {  // !RelationalFlag
          feat_src_entry_id = edata_idx;
          s += gdata.exp[edata_idx * num_heads + head_idx] /
               gdata.sum[dst_vid * num_heads + head_idx] *
               gdata.feat_src[feat_src_entry_id * gdata.feat_src_xlen +
                              head_idx * hidden_xlen + feat_idx];

          atomicAdd(&gdata.ret[dst_vid * gdata.feat_src_xlen +
                               head_idx * hidden_xlen + feat_idx],
                    gdata.exp[edata_idx * num_heads + head_idx] /
                        gdata.sum[dst_vid * num_heads + head_idx] *
                        gdata.feat_src[feat_src_entry_id * gdata.feat_src_xlen +
                                       head_idx * hidden_xlen + feat_idx]);
        }
      }
    }
  }
}

template <typename Idx, typename DType, CompactAsOfNodeKind kind>
__global__ void HET_gatSumProdZipDivKernel_relational_separate_coo(
    GatFusedData<Idx, DType> gdata, const Idx *rel_ptrs, const Idx *row_indices,
    const Idx *col_indices, int64_t num_edges,
    const ETypeMapperData<Idx, kind> etype_mapper_data,
    const ETypeMapperData<Idx, kind> etype_mapper_data_col,
    int64_t num_relations) {
  _gatSumProdZipDivKernel_edge_parallel<Idx, DType, kind, true, false>(
      gdata, rel_ptrs, row_indices, col_indices, num_edges, etype_mapper_data,
      etype_mapper_data_col, num_relations);
}

// edge-centric schedule cf. HET_gatExpLeakyReluSumKernel in
// [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGAT.cu.h]]
template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag, bool FullCartesianFlag>
__device__ __forceinline__ void _gatExpLeakyReluSumKernel_edge_parallel(
    GatFusedData<Idx, DType> gdata, const Idx *etypes, const Idx *row_indices,
    const Idx *col_indices, int64_t num_edges,
    const ETypeMapperData<Idx, kind> etype_mapper_data,
    const ETypeMapperData<Idx, kind> etype_mapper_data_col,
    int64_t num_relations) {
  constexpr bool EtypeRelPtrIndexSearch = true;
  Idx resume_from = 0;

  constexpr bool ETypeRelPtrFlag = true;
  constexpr bool CompactAsOfNodeFlag = IsCompact(kind);
  constexpr bool DualUniqueNodeList = IsCompactWithDualList(kind);
  Idx tx = blockIdx.x * blockDim.x + threadIdx.x;
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;

  Idx num_heads = gdata.num_heads;
  for (Idx eidx = ty; eidx < num_edges; eidx += blockDim.y * gridDim.y) {
    Idx dst_vid = col_indices[eidx];

    for (Idx feat_idx = tx; feat_idx < num_heads;
         feat_idx += blockDim.x * gridDim.x) {
      // 1. Load dstnation vertex into shared memory
      Idx feat_off_dst = -1;
      if constexpr (CompactAsOfNodeFlag) {
        feat_off_dst = dst_vid * num_heads + feat_idx;
      }
      // 2. Do the computation
      DType sum = 0.;
      Idx src_id = *(row_indices + eidx);
      Idx feat_off_src = -1;
      Idx edata_idx = gdata.eids[eidx];
      Idx dst_vid_relational = -1;
      if constexpr (CompactAsOfNodeFlag) {
        if constexpr (RelationalFlag) {
          Idx etype = -1;
          if constexpr (ETypeRelPtrFlag) {
            if constexpr (EtypeRelPtrIndexSearch) {
              etype = linear_search(num_relations, etypes, eidx, resume_from);
              resume_from = etype;
            } else {
              etype = binary_search(num_relations, etypes, eidx);
            }
          } else {
            etype = etypes[eidx];
          }
          Idx src_vid_relational = find_relational_compact_as_of_node_index(
              etype, src_id, edata_idx, etype_mapper_data);
          if constexpr (DualUniqueNodeList) {
            dst_vid_relational = find_relational_compact_as_of_node_index(
                etype, dst_vid, edata_idx, etype_mapper_data_col);
          } else {
            dst_vid_relational = find_relational_compact_as_of_node_index(
                etype, dst_vid, edata_idx, etype_mapper_data);
          }
          feat_off_src = src_vid_relational * num_heads + feat_idx;
          feat_off_dst = dst_vid_relational * num_heads + feat_idx;
          if constexpr (FullCartesianFlag) {
            // NB: This is the case where we have the data stored in
            // (relation, node) but do not compress the (relation, node)
            // matrix. It could be a case in subgraph where compressing along
            // the node dimension may not be worth it.
            CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
                CompactAsOfNodeFlag && RelationalFlag && FullCartesianFlag,
                "should be non-reachable not implemented");
          }
        } else {
          feat_off_src = src_id * num_heads + feat_idx;
        }
      } else {
        feat_off_src = edata_idx * num_heads + feat_idx;
        feat_off_dst = edata_idx * num_heads + feat_idx;
      }
      DType tmp =
          gatLeakyReluExp(gdata.el[feat_off_src] + gdata.er[feat_off_dst],
                          gdata.leaky_relu_slope);
      gdata.exp[Idx(edata_idx * num_heads) + feat_idx] = tmp;
      if constexpr (RelationalFlag) {
        atomicAdd(&gdata.sum[Idx(dst_vid * num_heads) + feat_idx], tmp);
      }
      sum += tmp;
      //}
      if constexpr (!RelationalFlag) {
        atomicAdd(&gdata.sum[Idx(dst_vid * num_heads) + feat_idx], sum);
      }
    }
  }
}

template <typename Idx, typename DType, CompactAsOfNodeKind kind>
__global__ void HET_gatExpLeakyReluSumKernel_relational_separate_coo(
    GatFusedData<Idx, DType> gdata, const Idx *rel_ptrs, const Idx *row_indices,
    const Idx *col_indices, int64_t num_edges,
    const ETypeMapperData<Idx, kind> etype_mapper_data,
    const ETypeMapperData<Idx, kind> etype_mapper_data_col,
    int64_t num_relations) {
  _gatExpLeakyReluSumKernel_edge_parallel<Idx, DType, kind, true, false>(
      gdata, rel_ptrs, row_indices, col_indices, num_edges, etype_mapper_data,
      etype_mapper_data_col, num_relations);
}
