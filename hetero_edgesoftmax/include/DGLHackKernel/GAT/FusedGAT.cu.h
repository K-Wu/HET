#pragma once
#include "DGLHackKernel/DGLHackKernel.h"

template <typename Idx, typename DType>
struct GatFusedData {
  // feat_size size along feature dimension
  Idx feat_src_xlen{0};
  Idx feat_src_hidden{0};
  Idx e_xlen{0};
  Idx ret_xlen{0};
  // num nodes
  // Idx n{0};
  Idx* eids;
  DType leaky_relu_slope;
  // Inputs
  DType *feat_src{nullptr}, *el{nullptr}, *er{nullptr};
  // Intermediates
  DType *sum{nullptr}, *exp{nullptr};
  // Output
  DType* ret{nullptr};
};

template <typename DType>
__device__ __forceinline__ DType gatLeakyReluExp(DType val, DType slope) {
  return val > 0 ? exp(val) : exp(slope * val);
}

// from seastar dgl-hack src/kernel/cuda/binary_reduce_impl.cu
// NB: when CompactAsOfNodeFlag is false, gdata.el, gdata.er, gdata.feat_src are
// edge-wise data instead of node-wise.
template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool RelationalFlag, bool ETypeRelPtrFlag, bool FullCartesianFlag>
__device__ __forceinline__ void _gatSumProdZipDivKernel(
    GatFusedData<Idx, DType> gdata, const Idx* row_offsets,
    const Idx* column_indices, const Idx* etypes, int64_t num_rows,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  Idx e_xlen = gdata.e_xlen;
  Idx hidden_xlen = gdata.feat_src_xlen / e_xlen;
  for (Idx dst_vid = blockIdx.y; dst_vid < num_rows; dst_vid += gridDim.y) {
    Idx start_off = *(row_offsets + dst_vid);
    Idx end_off = *(row_offsets + dst_vid + 1);
    for (Idx head_idx = blockIdx.x * blockDim.x + threadIdx.x;
         head_idx < e_xlen; head_idx += blockDim.x * gridDim.x) {
      for (Idx feat_idx = threadIdx.y; feat_idx < hidden_xlen;
           feat_idx += blockDim.y) {
        DType s = 0.;
        for (Idx eidx = start_off; eidx < end_off; eidx++) {
          Idx src_vid = column_indices[eidx];
          Idx feat_src_entry_id = -1;
          Idx edge_id = gdata.eids[eidx];
          if constexpr (RelationalFlag) {
            Idx sum_idx = -1;
            Idx etype = -1;
            if constexpr (ETypeRelPtrFlag) {
              etype = binary_search(num_relations, etypes, eidx);
            } else {
              etype = etypes[eidx];
            }
            if constexpr (CompactAsOfNodeFlag) {
              feat_src_entry_id = find_relational_compact_as_of_node_index(
                  etype, src_vid, unique_srcs_and_dests_node_indices,
                  unique_srcs_and_dests_rel_ptr);

            } else {
              // NB: we need to use edge_id instead of eidx here
              feat_src_entry_id = edge_id;
            }

            if constexpr (FullCartesianFlag) {
              // NB: This is the case where we have the data stored in
              // (relation, node) but do not compress the (relation, node)
              // matrix. It could be a case in subgraph where compressing along
              // the node dimension may not be worth it.
              CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
                  FullCartesianFlag, "should be non-reachable not implemented");
            } else {
              sum_idx = find_relational_compact_as_of_node_index(
                  etype, dst_vid, unique_srcs_and_dests_node_indices,
                  unique_srcs_and_dests_rel_ptr);
            }

            s += (gdata.exp[edge_id * e_xlen + head_idx] /
                  gdata.sum[sum_idx * e_xlen + head_idx] *
                  gdata.feat_src[feat_src_entry_id * gdata.feat_src_xlen +
                                 head_idx * hidden_xlen + feat_idx]);
          } else {  // !RelationalFlag
            // NB: feat_src_entry_id varies between edge_id and src_vid
            // depending on compactasofnodeflag
            if constexpr (CompactAsOfNodeFlag) {
              feat_src_entry_id = src_vid;
            } else {
              feat_src_entry_id = edge_id;
            }
            s += gdata.exp[edge_id * e_xlen + head_idx] /
                 gdata.sum[dst_vid * e_xlen + head_idx] *
                 gdata.feat_src[feat_src_entry_id * gdata.feat_src_xlen +
                                head_idx * hidden_xlen + feat_idx];
          }
        }

        gdata.ret[dst_vid * gdata.feat_src_xlen + head_idx * hidden_xlen +
                  feat_idx] = s;
      }
    }
  }
}

template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool RelationalFlag>
__global__ void gatSumProdZipDivKernel(
    GatFusedData<Idx, DType> gdata, const Idx* row_offsets,
    const Idx* column_indices, const Idx* etypes, int64_t num_rows,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices) {
  _gatSumProdZipDivKernel<Idx, DType, CompactAsOfNodeFlag, RelationalFlag,
                          false, false>(
      gdata, row_offsets, column_indices, etypes, num_rows,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices, -1);
}

// from seastar dgl-hack src/kernel/cuda/binary_reduce_impl.cu
// NB: when CompactAsOfNodeFlag is false, gdata.el, gdata.er, gdata.feat_src are
// edge-wise data instead of node-wise.
template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool RelationalFlag, bool ETypeRelPtrFlag, bool FullCartesianFlag>
__device__ __forceinline__ void _gatExpLeakyReluSumKernel(
    GatFusedData<Idx, DType> gdata, const Idx* row_offsets,
    const Idx* column_indices, const Idx* etypes, int64_t num_rows,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  // extern __shared__ DType er[];
  Idx tx = blockIdx.x * blockDim.x + threadIdx.x;
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;

  Idx e_xlen = gdata.e_xlen;
  for (Idx dst_vid = ty; dst_vid < num_rows;
       dst_vid += blockDim.y * gridDim.y) {
    Idx start_off = *(row_offsets + dst_vid);
    Idx end_off = *(row_offsets + dst_vid + 1);

    for (Idx feat_idx = tx; feat_idx < e_xlen;
         feat_idx += blockDim.x * gridDim.x) {
      // 1. Load dstnation vertex into shared memory
      Idx feat_off_dst = -1;
      if constexpr (CompactAsOfNodeFlag) {
        feat_off_dst = dst_vid * e_xlen + feat_idx;
      }
      // er[threadIdx.x] = gdata.er[feat_off_dst];
      //__syncthreads();
      // 2. Do the computation
      DType sum = 0.;
      for (Idx eidx = start_off; eidx < end_off; ++eidx) {
        Idx src_id = *(column_indices + eidx);
        Idx feat_off_src = -1;
        Idx edge_id = gdata.eids[eidx];
        Idx dst_vid_relational = -1;
        Idx etype = -1;
        if constexpr (RelationalFlag) {
          if constexpr (ETypeRelPtrFlag) {
            etype = binary_search(num_relations, etypes, eidx);
          } else {
            etype = etypes[eidx];
          }
          dst_vid_relational = find_relational_compact_as_of_node_index(
              etype, dst_vid, unique_srcs_and_dests_node_indices,
              unique_srcs_and_dests_rel_ptr);
        }
        if constexpr (CompactAsOfNodeFlag) {
          if constexpr (RelationalFlag) {
            // Idx etype = etypes[eidx];
            if constexpr (FullCartesianFlag) {
              // NB: This is the case where we have the data stored in
              // (relation, node) but do not compress the (relation, node)
              // matrix. It could be a case in subgraph where compressing along
              // the node dimension may not be worth it.
              assert(0 && "should be non-reachable not implemented");
            }
            Idx src_vid_temp = find_relational_compact_as_of_node_index(
                etype, src_id, unique_srcs_and_dests_node_indices,
                unique_srcs_and_dests_rel_ptr);
            feat_off_src = src_vid_temp * e_xlen + feat_idx;
            feat_off_dst = dst_vid_relational * e_xlen + feat_idx;
          } else {
            feat_off_src = src_id * e_xlen + feat_idx;
          }
        } else {
          // per edge
          feat_off_src = edge_id * e_xlen + feat_idx;
          feat_off_dst = edge_id * e_xlen + feat_idx;
        }
        // DType tmp = gatLeakyReluExp(gdata.el[feat_off_src] + er[threadIdx.x],
        // gdata.leaky_relu_slope);
        DType tmp =
            gatLeakyReluExp(gdata.el[feat_off_src] + gdata.er[feat_off_dst],
                            gdata.leaky_relu_slope);
        gdata.exp[Idx(edge_id * e_xlen) + feat_idx] = tmp;
        if constexpr (RelationalFlag) {
          // NB: double check dst_vid_relational is defined when
          // !CompactAsOfNodeFlag && RelationalFlag
          // TODO: fix this and align dst_vid_relational definition with
          // _fusedGatBackwardGradElErFeatSrcFused
          atomicAdd(&gdata.sum[Idx(dst_vid_relational * e_xlen) + feat_idx],
                    tmp);
        }
        sum += tmp;
      }
      if constexpr (!RelationalFlag) {
        gdata.sum[Idx(dst_vid * e_xlen) + feat_idx] = sum;
      }
    }
  }
}

template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool RelationalFlag>
__global__ void gatExpLeakyReluSumKernel(
    GatFusedData<Idx, DType> gdata, const Idx* row_offsets,
    const Idx* column_indices, const Idx* etypes, int64_t num_rows,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices) {
  _gatExpLeakyReluSumKernel<Idx, DType, CompactAsOfNodeFlag, RelationalFlag,
                            false, false>(
      gdata, row_offsets, column_indices, etypes, num_rows,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices, -1);
}

template <typename Idx, typename DType>
constexpr auto relational_gatExpLeakyReluSumKernel_per_edge =
    gatExpLeakyReluSumKernel<Idx, DType, false, false, false>;
template <typename Idx, typename DType>
constexpr auto relational_gatSumProdZipDivKernel_per_edge =
    gatSumProdZipDivKernel<Idx, DType, false, false>;