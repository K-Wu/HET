#pragma once
#include <cuda_runtime.h>
#include "kernel_enums.h"

// FIXME: check if RGAT needs different a vector for different etypes
template <typename Idx, typename DType>
struct GatFusedData {
  // feat_size size along feature dimension
  Idx feat_src_xlen{0};
  // Idx feat_src_hidden{0};
  Idx num_heads{0};
  // Idx ret_xlen{0};
  // num nodes
  // Idx n{0};
  Idx *__restrict__ eids{nullptr};
  DType leaky_relu_slope;
  // Inputs
  DType *__restrict__ feat_src{nullptr}, *__restrict__ el{nullptr},
      *__restrict__ er{nullptr};
  // Intermediates
  DType *__restrict__ sum{nullptr}, *__restrict__ exp{nullptr};
  // Output
  DType *__restrict__ ret{nullptr};
};

template <typename DType>
__device__ __forceinline__ DType gatLeakyReluExp(DType val, DType slope) {
  return val > 0 ? exp(val) : exp(slope * val);
}

// from seastar dgl-hack src/kernel/cuda/binary_reduce_impl.cu
// NB: when CompactAsOfNodeFlag is false, gdata.el, gdata.er, gdata.feat_src are
// edge-wise data instead of node-wise.
template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag, bool ETypeRelPtrFlag, bool FullCartesianFlag>
__device__ __forceinline__ void _gatSumProdZipDivKernel(
    GatFusedData<Idx, DType> gdata, const Idx *row_offsets,
    const Idx *column_indices, const Idx *etypes, int64_t num_rows,
    const Idx *unique_srcs_and_dests_rel_ptr,
    const Idx *unique_srcs_and_dests_node_indices, int64_t num_relations) {
  Idx num_heads = gdata.num_heads;
  Idx hidden_xlen = gdata.feat_src_xlen / num_heads;
  for (Idx dst_vid = blockIdx.y; dst_vid < num_rows; dst_vid += gridDim.y) {
    Idx start_off = *(row_offsets + dst_vid);
    Idx end_off = *(row_offsets + dst_vid + 1);
    for (Idx head_idx = threadIdx.y; head_idx < num_heads;
         head_idx += blockDim.y) {
      for (Idx feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
           feat_idx < hidden_xlen; feat_idx += blockDim.x * gridDim.x) {
        DType s = 0.;
        for (Idx eidx = start_off; eidx < end_off; eidx++) {
          Idx src_vid = column_indices[eidx];
          Idx feat_src_entry_id = -1;
          Idx edata_idx = gdata.eids[eidx];
          if constexpr (RelationalFlag) {
            // Idx sum_idx = -1;
            Idx etype = -1;
            if constexpr (ETypeRelPtrFlag) {
              etype = binary_search(num_relations, etypes, eidx);
            } else {
              etype = etypes[eidx];
            }
            if constexpr (IsCompact(kind)) {
              feat_src_entry_id = find_relational_compact_as_of_node_index(
                  etype, src_vid, unique_srcs_and_dests_rel_ptr,
                  unique_srcs_and_dests_node_indices);

            } else {
              // NB: we need to use edata_idx instead of eidx here
              feat_src_entry_id = edata_idx;
            }
            // TODO: actually full cartesian can be applied both to
            // feat_src_entry_id and sum_idx, in future we may need to add an
            // additional FullCartesianFlag to cover all cases
            if constexpr (FullCartesianFlag) {
              // NB: This is the case where we have the data stored in
              // (relation, node) but do not compress the (relation, node)
              // matrix. It could be a case in subgraph where compressing along
              // the node dimension may not be worth it.
              CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
                  FullCartesianFlag, "should be non-reachable not implemented");
            }  // else {
               // sum_idx = find_relational_compact_as_of_node_index(
               //     etype, dst_vid, unique_srcs_and_dests_node_indices,
               //     unique_srcs_and_dests_rel_ptr);
            //}

            s += (gdata.exp[edata_idx * num_heads + head_idx] /
                  gdata.sum[dst_vid * num_heads + head_idx] *
                  gdata.feat_src[feat_src_entry_id * gdata.feat_src_xlen +
                                 head_idx * hidden_xlen + feat_idx]);
          } else {  // !RelationalFlag
            // NB: feat_src_entry_id varies between edata_idx and src_vid
            // depending on compactasofnodeflag
            if constexpr (IsCompact(kind)) {
              feat_src_entry_id = src_vid;
            } else {
              feat_src_entry_id = edata_idx;
            }
            s += gdata.exp[edata_idx * num_heads + head_idx] /
                 gdata.sum[dst_vid * num_heads + head_idx] *
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

template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag>
__global__ void HET_gatSumProdZipDivKernel(
    GatFusedData<Idx, DType> gdata, const Idx *row_offsets,
    const Idx *column_indices, const Idx *etypes, int64_t num_rows,
    const Idx *unique_srcs_and_dests_rel_ptr,
    const Idx *unique_srcs_and_dests_node_indices) {
  _gatSumProdZipDivKernel<Idx, DType, kind, RelationalFlag, false, false>(
      gdata, row_offsets, column_indices, etypes, num_rows,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices, -1);
}

// from seastar dgl-hack src/kernel/cuda/binary_reduce_impl.cu
// NB: when CompactAsOfNodeFlag is false, gdata.el, gdata.er, gdata.feat_src are
// edge-wise data instead of node-wise.
template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag, bool ETypeRelPtrFlag, bool FullCartesianFlag>
__device__ __forceinline__ void _gatExpLeakyReluSumKernel(
    GatFusedData<Idx, DType> gdata, const Idx *row_offsets,
    const Idx *column_indices, const Idx *etypes, int64_t num_rows,
    const Idx *unique_srcs_and_dests_rel_ptr,
    const Idx *unique_srcs_and_dests_node_indices, int64_t num_relations) {
  // extern __shared__ DType er[];
  Idx tx = blockIdx.x * blockDim.x + threadIdx.x;
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;

  Idx num_heads = gdata.num_heads;
  for (Idx dst_vid = ty; dst_vid < num_rows;
       dst_vid += blockDim.y * gridDim.y) {
    Idx start_off = *(row_offsets + dst_vid);
    Idx end_off = *(row_offsets + dst_vid + 1);

    for (Idx feat_idx = tx;
         feat_idx <
         num_heads;  // NB: e_xlen is set as num_head.
                     // NB: this is calculating attention and sum of attention
                     // thus no need for the gdata.feat_src_xlen, i.e.,
                     // num_feat_per_head, for-loop level
         feat_idx += blockDim.x * gridDim.x) {
      // 1. Load destination vertex into shared memory
      Idx feat_off_dst = -1;
      if constexpr (IsCompact(kind)) {
        feat_off_dst = dst_vid * num_heads + feat_idx;
      }
      // er[threadIdx.x] = gdata.er[feat_off_dst];
      //__syncthreads();
      // 2. Do the computation
      DType sum = 0.;
      for (Idx eidx = start_off; eidx < end_off; ++eidx) {
        Idx src_id = *(column_indices + eidx);
        Idx feat_off_src = -1;
        Idx edata_idx = gdata.eids[eidx];
        Idx dst_vid_relational = -1;
        Idx etype = -1;
        if constexpr (RelationalFlag) {
          if constexpr (ETypeRelPtrFlag) {
            etype = binary_search(num_relations, etypes, eidx);
          } else {
            etype = etypes[eidx];
          }
          dst_vid_relational = find_relational_compact_as_of_node_index(
              etype, dst_vid, unique_srcs_and_dests_rel_ptr,
              unique_srcs_and_dests_node_indices);
        }
        if constexpr (IsCompact(kind)) {
          if constexpr (RelationalFlag) {
            // Idx etype = etypes[eidx];
            if constexpr (FullCartesianFlag) {
              // NB: This is the case where we have the data stored in
              // (relation, node) but do not compress the (relation, node)
              // matrix. It could be a case in subgraph where compressing along
              // the node dimension may not be worth it.
              CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
                  IsCompact(kind) && RelationalFlag && FullCartesianFlag,
                  "should be non-reachable not implemented");
            }
            Idx src_vid_relational = find_relational_compact_as_of_node_index(
                etype, src_id, unique_srcs_and_dests_rel_ptr,
                unique_srcs_and_dests_node_indices);
            feat_off_src = src_vid_relational * num_heads + feat_idx;
            feat_off_dst = dst_vid_relational * num_heads + feat_idx;
          } else {
            feat_off_src = src_id * num_heads + feat_idx;
          }
        } else {
          // per edge
          feat_off_src = edata_idx * num_heads + feat_idx;
          feat_off_dst = edata_idx * num_heads + feat_idx;
        }
        // DType tmp = gatLeakyReluExp(gdata.el[feat_off_src] + er[threadIdx.x],
        // gdata.leaky_relu_slope);
        DType tmp =
            gatLeakyReluExp(gdata.el[feat_off_src] + gdata.er[feat_off_dst],
                            gdata.leaky_relu_slope);
        gdata.exp[Idx(edata_idx * num_heads) + feat_idx] = tmp;
        if constexpr (RelationalFlag) {
          // NB: double check dst_vid_relational is defined when
          // !CompactAsOfNodeFlag && RelationalFlag
          // TODO: fix this and align dst_vid_relational definition with
          // _fusedGatBackwardGradElErFeatSrcFused
          atomicAdd(&gdata.sum[Idx(dst_vid * num_heads) + feat_idx], tmp);
        }
        sum += tmp;
      }
      if constexpr (!RelationalFlag) {
        gdata.sum[Idx(dst_vid * num_heads) + feat_idx] = sum;
      }
    }
  }
}

template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag>
__global__ void HET_gatExpLeakyReluSumKernel(
    GatFusedData<Idx, DType> gdata, const Idx *row_offsets,
    const Idx *column_indices, const Idx *etypes, int64_t num_rows,
    const Idx *unique_srcs_and_dests_rel_ptr,
    const Idx *unique_srcs_and_dests_node_indices) {
  _gatExpLeakyReluSumKernel<Idx, DType, kind, RelationalFlag, false, false>(
      gdata, row_offsets, column_indices, etypes, num_rows,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices, -1);
}

template <typename Idx, typename DType>
constexpr auto relational_gatExpLeakyReluSumKernel_per_edge =
    HET_gatExpLeakyReluSumKernel<Idx, DType, false, false, false>;
template <typename Idx, typename DType>
constexpr auto relational_gatSumProdZipDivKernel_per_edge =
    HET_gatSumProdZipDivKernel<Idx, DType, false, false>;