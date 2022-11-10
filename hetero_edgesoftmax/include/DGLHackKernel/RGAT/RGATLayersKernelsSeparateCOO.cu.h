#pragma once
#include "DGLHackKernel/DGLHackKernel.h"
#include "DGLHackKernel/GAT/FusedGAT.cu.h"

// edge-centric schedule cf. gatSumProdZipDivKernel in
// [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGAT.cu.h]]
template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool RelationalFlag>
__device__ __forceinline__ void _gatSumProdZipDivKernel_edge_parallel(
    GatFusedData<Idx, DType> gdata, const Idx* etypes, const Idx* row_indices,
    const Idx* col_indices, int64_t num_edges,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  constexpr bool ETypeRelPtrFlag = true;
  Idx e_xlen = gdata.e_xlen;
  Idx hidden_xlen = gdata.feat_src_xlen / e_xlen;
  for (Idx eidx = blockIdx.y; eidx < num_edges; eidx += gridDim.y) {
    Idx dst_vid = col_indices[e];
    // for (Idx dst_vid = blockIdx.y; dst_vid < num_rows; dst_vid += gridDim.y)
    // { Idx start_off = *(row_offsets + dst_vid); Idx end_off = *(row_offsets +
    // dst_vid + 1);
    for (Idx head_idx = blockIdx.x * blockDim.x + threadIdx.x;
         head_idx < e_xlen; head_idx += blockDim.x * gridDim.x) {
      for (Idx feat_idx = threadIdx.y; feat_idx < hidden_xlen;
           feat_idx += blockDim.y) {
        DType s = 0.;
        // for (Idx eidx = start_off; eidx < end_off; eidx++) {
        Idx src_vid = column_indices[eidx];
        Idx feat_src_entry_id;
        Idx edge_id = gdata.eids[eidx];
        if constexpr (RelationalFlag) {
          Idx sum_idx;
          Idx etype;
          if constexpr (ETypeRelPtrFlag) {
            etype = binary_search(num_relations, etypes, eidx);
          } else {
            etype = etypes[eidx];
          }
          if constexpr (CompactAsOfNodeFlag) {
            feat_src_entry_id = src_vid;
            sum_idx = find_relational_compact_as_of_node_index(
                etype, dst_vid, unique_srcs_and_dests_node_indices,
                unique_srcs_and_dests_rel_ptr);
          } else {
            // NB: This is the case where we have the data stored in
            // (relation, node) but do not compress the (relation, node)
            // matrix. It could be a case in subgraph where compressing along
            // the node dimension may not be worth it.
            assert(0 && "should be non-reachable not implemented");
          }
          s += (gdata.exp[edge_id * e_xlen + head_idx] /
                gdata.sum[sum_idx * e_xlen + head_idx] *
                gdata.feat_src[feat_src_entry_id * gdata.feat_src_xlen +
                               head_idx * hidden_xlen + feat_idx]);
          atomicAdd(gdata.feat_dst + dst_vid * gdata.feat_dst_xlen +
                        head_idx * hidden_xlen + feat_idx,
                    s);
        } else {  // !RelationalFlag
          feat_src_entry_id = edge_id;
          s += gdata.exp[edge_id * e_xlen + head_idx] /
               gdata.sum[dst_vid * e_xlen + head_idx] *
               gdata.feat_src[feat_src_entry_id * gdata.feat_src_xlen +
                              head_idx * hidden_xlen + feat_idx];

          atomicAdd(gdata.feat_dst + dst_vid * gdata.feat_dst_xlen +
                        head_idx * hidden_xlen + feat_idx,
                    s);
        }
        //}
        // gdata.ret[dst_vid * gdata.feat_src_xlen + head_idx * hidden_xlen +
        //           feat_idx] = s;
      }
    }
  }
}

template <typename Idx, typename DType>
__global__ void
gatSumProdZipDivKernel_relational_separate_coo_compact_as_of_node(
    GatFusedData<Idx, DType> gdata, const Idx* rel_ptrs, const Idx* row_indices,
    const Idx* col_indices, int64_t num_edges,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  _gatSumProdZipDivKernel_edge_parallel<Idx, DType, true, true>(
      gdata, rel_ptrs, row_indices, col_indices, num_edges,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
      num_relations);
}

// edge-centric schedule cf. gatExpLeakyReluSumKernel in
// [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGAT.cu.h]]
template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool RelationalFlag>
__device__ __forceinline__ void _gatExpLeakyReluSumKernel_edge_parallel(
    GatFusedData<Idx, DType> gdata, const Idx* etypes, const Idx* row_indices,
    const Idx* col_indices, int64_t num_edges,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  constexpr bool ETypeRelPtrFlag = true;
  Idx tx = blockIdx.x * blockDim.x + threadIdx.x;
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;

  Idx e_xlen = gdata.e_xlen;
  for (Idx eidx = ty; eidx < num_edges; eidx += blockDim.y * gridDim.y) {
    Idx dst_vid = col_indices[eidx];
    // for (Idx dst_vid = ty; dst_vid < num_rows;
    //     dst_vid += blockDim.y * gridDim.y;) {
    // Idx start_off = *(row_offsets + dst_vid);
    // Idx end_off = *(row_offsets + dst_vid + 1);

    for (Idx feat_idx = tx; feat_idx < e_xlen;
         feat_idx += blockDim.x * gridDim.x;) {
      // 1. Load dstnation vertex into shared memory
      Idx feat_off_dst;
      if constexpr (CompactAsOfNodeFlag) {
        feat_off_dst = dst_vid * e_xlen + feat_idx;
      }
      // er[threadIdx.x] = gdata.er[feat_off_dst];
      //__syncthreads();
      // 2. Do the computation
      DType sum = 0.;
      // for (Idx eidx = start_off; eidx < end_off; ++eidx) {
      Idx src_id = *(column_indices + eidx);
      Idx feat_off_src;
      Idx edge_id = gdata.eids[eidx];
      Idx dst_vid_relational;
      if constexpr (CompactAsOfNodeFlag) {
        if constexpr (RelationalFlag) {
          // Idx etype = etypes[eidx];
          Idx etype;
          if constexpr (ETypeRelPtrFlag) {
            etype = binary_search(num_relations, etypes, eidx);
          } else {
            etype = etypes[eidx];
          }
          Idx src_vid_temp = find_relational_compact_as_of_node_index(
              etype, src_id, unique_srcs_and_dests_node_indices,
              unique_srcs_and_dests_rel_ptr);
          dst_vid_relational = find_relational_compact_as_of_node_index(
              etype, dst_vid, unique_srcs_and_dests_node_indices,
              unique_srcs_and_dests_rel_ptr);
          feat_off_src = src_vid_temp * e_xlen + feat_idx;
          feat_off_dst = dst_vid_relational * e_xlen + feat_idx;
        } else {
          feat_off_src = src_id * e_xlen + feat_idx;
        }
      } else {
        if constexpr (RelationalFlag) {
          // NB: This is the case where we have the data stored in
          // (relation, node) but do not compress the (relation, node)
          // matrix. It could be a case in subgraph where compressing along
          // the node dimension may not be worth it.
          assert(0 && "should be non-reachable not implemented");
        } else {
          feat_off_src = edge_id * e_xlen + feat_idx;
          feat_off_dst = edge_id * e_xlen + feat_idx;
        }
      }
      // DType tmp = gatLeakyReluExp(gdata.el[feat_off_src] + er[threadIdx.x],
      // gdata.leaky_relu_slope);
      DType tmp =
          gatLeakyReluExp(gdata.el[feat_off_src] + gdata.er[feat_off_dst],
                          gdata.leaky_relu_slope);
      gdata.exp[Idx(edge_id * e_xlen) + feat_idx] = tmp;
      if constexpr (RelationalFlag) {
        atomicAdd(&gdata.sum[Idx(dst_vid_relational * e_xlen) + feat_idx], tmp);
      }
      sum += tmp;
      //}
      if constexpr (!RelationalFlag) {
        // gdata.sum[Idx(dst_vid * e_xlen) + feat_idx] = sum;
        atomicAdd(&gdata.sum[Idx(dst_vid * e_xlen) + feat_idx], sum);
      }
    }
  }
}

template <typename Idx, typename DType>
__global__ void
gatExpLeakyReluSumKernel_relational_separate_coo_compact_as_of_node(
    GatFusedData<Idx, DType> gdata, const Idx* rel_ptrs, const Idx* row_indices,
    const Idx* col_indices, int64_t num_edges,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  _gatExpLeakyReluSumKernel_edge_parallel<Idx, DType, true, true>(
      gdata, rel_ptrs, row_indices, col_indices, num_edges,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
      num_relations);
}
