#pragma once

#include <cuda_runtime.h>

#include "kernel_enums.h"

template <typename Idx, typename DType> struct InnerProductData {
  // feat_size size along feature dimension
  Idx feat_src_xlen{0};
  Idx num_heads{0};
  Idx *__restrict__ eids{nullptr};
  // Inputs
  DType *__restrict__ feat_src{nullptr}, *__restrict__ feat_dst{nullptr};
  // Output
  DType *__restrict__ edge_inner_product{nullptr};
};

template <typename Idx, typename DType> struct BackwardInnerProductData {
  // feat_size size along feature dimension
  Idx feat_src_xlen{0};
  Idx num_heads{0};
  Idx *__restrict__ eids{nullptr};
  // Inputs
  DType *__restrict__ feat_src{nullptr}, *__restrict__ feat_dst{nullptr};
  DType *__restrict__ grad_inner_product{nullptr};
  // Output
  DType *__restrict__ grad_feat_dst{nullptr}, *__restrict__ grad_feat_src{
                                                  nullptr};
};

// adapted from _gatSumProdZipDivKernel in
// [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGAT.cu.h]]
template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag, bool ETypeRelPtrFlag, bool FullCartesianFlag>
__global__ void HET_inner_product_fw_kernel(
    InnerProductData<Idx, DType> gdata, const Idx *row_offsets,
    const Idx *column_indices, const ETypeData<Idx, ETypeRelPtrFlag> etype_data,
    int64_t num_rows, const ETypeMapperData<Idx, kind> etype_mapper_data) {
  Idx num_heads = gdata.num_heads;
  Idx hidden_xlen = gdata.feat_src_xlen / num_heads;
  for (Idx dst_vid = blockIdx.y; dst_vid < num_rows; dst_vid += gridDim.y) {
    Idx start_off = *(row_offsets + dst_vid);
    Idx end_off = *(row_offsets + dst_vid + 1);
    for (Idx head_idx = threadIdx.y; head_idx < num_heads;
         head_idx += blockDim.y) {
      for (Idx feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
           feat_idx < hidden_xlen; feat_idx += blockDim.x * gridDim.x) {
        for (Idx eidx = start_off; eidx < end_off; eidx++) {
          DType s = 0.;
          Idx src_vid = column_indices[eidx];
          Idx feat_src_entry_id = -1;
          Idx edata_idx = gdata.eids[eidx];
          if constexpr (RelationalFlag) {
            if constexpr (IsCompact(kind)) {
              Idx etype = -1;
              if constexpr (ETypeRelPtrFlag) {
                etype = binary_search(etype_data.num_relations,
                                      etype_data.etypes, eidx);
              } else {
                etype = etype_data.etypes[eidx];
              }

              feat_src_entry_id = find_relational_compact_as_of_node_index(
                  etype, src_vid, edata_idx, etype_mapper_data);

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
            }

            s += gdata.feat_dst[dst_vid * gdata.feat_src_xlen +
                                head_idx * hidden_xlen + feat_idx] *
                 gdata.feat_src[feat_src_entry_id * gdata.feat_src_xlen +
                                head_idx * hidden_xlen + feat_idx];
          } else { // !RelationalFlag
            // NB: feat_src_entry_id varies between edata_idx and src_vid
            // depending on compactasofnodeflag
            if constexpr (IsCompact(kind)) {
              feat_src_entry_id = src_vid;
            } else {
              feat_src_entry_id = edata_idx;
            }
            s += gdata.feat_dst[dst_vid * gdata.feat_src_xlen +
                                head_idx * hidden_xlen + feat_idx] *
                 gdata.feat_src[feat_src_entry_id * gdata.feat_src_xlen +
                                head_idx * hidden_xlen + feat_idx];
          }
          atomicAdd(&gdata.edge_inner_product[edata_idx * num_heads + head_idx],
                    s);
        }
      }
    }
  }
}

// adapted from _fusedGatBackwardGradElErFeatSrcFused in
// [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGATBackward.cu.h]]
template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag, bool ETypeRelPtrFlag>
__global__ void HET_inner_product_bck_kernel(
    BackwardInnerProductData<Idx, DType> gdata, const Idx *row_offsets,
    const Idx *column_indices, const ETypeData<Idx, ETypeRelPtrFlag> etype_data,
    int64_t num_rows, const ETypeMapperData<Idx, kind> etype_mapper_data) {
  Idx num_heads = gdata.num_heads;
  Idx hidden_xlen = gdata.feat_src_xlen / num_heads;
  for (Idx src_vid = blockIdx.y; src_vid < num_rows; src_vid += gridDim.y) {
    Idx start_off = row_offsets[src_vid];
    Idx end_off = row_offsets[src_vid + 1];
    for (Idx head_idx = threadIdx.y; head_idx < num_heads;
         head_idx += blockDim.y) {
      for (Idx feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
           feat_idx < hidden_xlen; feat_idx += blockDim.x * gridDim.x) {
        DType sfeatsrc = 0.;
        Idx feat_src_offset = -1;
        if constexpr (IsCompact(kind) && !RelationalFlag) {
          // in this case, feat_src_offset is the same regardless of which
          // outgoing edge we deal with
          feat_src_offset =
              src_vid * gdata.feat_src_xlen + head_idx * hidden_xlen + feat_idx;
        }
        for (Idx e = start_off; e < end_off; ++e) {
          Idx edata_idx = gdata.eids[e];
          Idx dst_vid = column_indices[e];
          if constexpr (!IsCompact(kind)) {
            // in this case, feat_src_offset, er_idx and el_idx are related to
            // edge id, regardless of the type of the edge
            feat_src_offset = edata_idx * gdata.feat_src_xlen +
                              head_idx * hidden_xlen + feat_idx;
          } else { // CompactAsOfNodeFlag
            if constexpr (RelationalFlag) {
              // in this case, er_idx (sum's index) is related to (relation,
              // unique node index) el_idx is related to (relation, unique node
              // index) feat_src_offset is related to (relation, unique node
              // index)
              Idx etype = -1;
              if constexpr (ETypeRelPtrFlag) {
                etype = binary_search(etype_data.num_relations,
                                      etype_data.etypes, e);
              } else {
                etype = etype_data.etypes[e];
              }
              Idx src_vid_relational = find_relational_compact_as_of_node_index(
                  etype, src_vid, edata_idx, etype_mapper_data);

              feat_src_offset = src_vid_relational * gdata.feat_src_xlen +
                                head_idx * hidden_xlen + feat_idx;
            }
          }

          Idx edge_offset = edata_idx * num_heads + head_idx;

          Idx dst_out_offset =
              dst_vid * gdata.feat_src_xlen + head_idx * hidden_xlen + feat_idx;
          gdata.grad_feat_dst + (dst_vid * gdata.feat_src_xlen +
                                 head_idx * hidden_xlen + feat_idx) =
              gdata.grad_inner_product[edata_idx * num_heads + head_idx] *
              gdata.feat_src[feat_src_offset];
          if constexpr (!IsCompact(kind) || RelationalFlag) {
            gdata.grad_feat_src + feat_src_offset =
                gdata.grad_inner_product[edata_idx * num_heads + head_idx] *
                gdata.feat_dst[dst_vid * gdata.feat_src_xlen +
                               head_idx * hidden_xlen + feat_idx];
          } else {
            sfeatsrc +=
                gdata.grad_inner_product[edata_idx * num_heads + head_idx] *
                gdata.feat_dst[dst_vid * gdata.feat_src_xlen +
                               head_idx * hidden_xlen + feat_idx];
          }
        } // for Idx e
        if constexpr (IsCompact(kind) && !RelationalFlag) {
          gdata.grad_feat_src[feat_src_offset] = sfeatsrc;
        }
      } // while feat_idx
    }   // while head_idx
  }     // while src_vid
}
