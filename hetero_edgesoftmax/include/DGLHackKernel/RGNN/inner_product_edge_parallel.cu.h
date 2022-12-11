#pragma once
#include "DGLHackKernel/DGLHackKernel.h"
#include "inner_product.cu.h"

// adapted from _gatSumProdZipDivKernel in
// [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGAT.cu.h]]
template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool RelationalFlag, bool ETypeRelPtrFlag, bool FullCartesianFlag,
          bool FeatSizeNoLessThanWarpSize>
__global__ void HET_inner_product_fw_kernel_edge_parallel(
    InnerProductData<Idx, DType> gdata, const Idx* row_indices,
    const Idx* column_indices, const Idx* etypes, int64_t num_edges,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  Idx num_heads = gdata.num_heads;
  Idx hidden_xlen = gdata.feat_src_xlen / num_heads;
  for (Idx eidx = blockIdx.y; eidx < num_edges; eidx += gridDim.y) {
    Idx dst_vid = row_indices[eidx];
    Idx edge_id = gdata.eids[eidx];
    Idx src_vid = column_indices[eidx];
    // Idx start_off = *(row_offsets + dst_vid);
    // Idx end_off = *(row_offsets + dst_vid + 1);

    for (Idx head_idx = threadIdx.y; head_idx < num_heads;
         head_idx += blockDim.y) {
      for (Idx feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
           feat_idx < hidden_xlen; feat_idx += blockDim.x * gridDim.x) {
        // for (Idx eidx = start_off; eidx < end_off; eidx++) {
        DType s = 0.;

        Idx feat_src_entry_id = -1;
        if constexpr (RelationalFlag) {
          // Idx sum_idx = -1;
          if constexpr (CompactAsOfNodeFlag) {
            Idx etype = -1;
            if constexpr (ETypeRelPtrFlag) {
              etype = binary_search(num_relations, etypes, eidx);
            } else {
              etype = etypes[eidx];
            }

            feat_src_entry_id = find_relational_compact_as_of_node_index(
                etype, src_vid, unique_srcs_and_dests_node_indices,
                unique_srcs_and_dests_rel_ptr);

          } else {
            // NB: we need to use edge_id instead of eidx here
            feat_src_entry_id = edge_id;
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

          s += gdata.feat_dst[dst_vid * gdata.feat_src_xlen +
                              head_idx * hidden_xlen + feat_idx] *
               gdata.feat_src[feat_src_entry_id * gdata.feat_src_xlen +
                              head_idx * hidden_xlen + feat_idx];

        } else {  // !RelationalFlag
          // NB: feat_src_entry_id varies between edge_id and src_vid
          // depending on compactasofnodeflag
          if constexpr (CompactAsOfNodeFlag) {
            feat_src_entry_id = src_vid;
          } else {
            feat_src_entry_id = edge_id;
          }
          s += gdata.feat_dst[dst_vid * gdata.feat_src_xlen +
                              head_idx * hidden_xlen + feat_idx] *
               gdata.feat_src[feat_src_entry_id * gdata.feat_src_xlen +
                              head_idx * hidden_xlen + feat_idx];
        }
        // warp reduction
        if constexpr (FeatSizeNoLessThanWarpSize) {
          s += __shfl_xor_sync(0xffffffff, s, 16);
          s += __shfl_xor_sync(0xffffffff, s, 8);
          s += __shfl_xor_sync(0xffffffff, s, 4);
          s += __shfl_xor_sync(0xffffffff, s, 2);
          s += __shfl_xor_sync(0xffffffff, s, 1);
          if (threadIdx.x % 32 == 0) {
            atomicAdd(&gdata.edge_inner_product[edge_id * num_heads + head_idx],
                      s);
          }
        } else {
          CONSTEXPR_FALSE_CLAUSE_UNREACHABLE(
              FeatSizeNoLessThanWarpSize,
              "should be non-reachable not implemented");
        }
      }
      //}
    }
  }
}

// adapted from _fusedGatBackwardGradElErFeatSrcFused in
// [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGATBackward.cu.h]]
template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool RelationalFlag, bool ETypeRelPtrFlag>
__global__ void HET_inner_product_bck_kernel_edge_parallel(
    BackwardInnerProductData<Idx, DType> gdata, const Idx* row_indices,
    const Idx* column_indices, const Idx* etypes, int64_t num_edges,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  Idx num_heads = gdata.num_heads;
  Idx hidden_xlen = gdata.feat_src_xlen / num_heads;
  for (Idx e = blockIdx.y; e < num_edges; e += gridDim.y) {
    // Idx start_off = row_offsets[src_vid];
    // Idx end_off = row_offsets[src_vid + 1];
    Idx src_vid = row_indices[e];
    for (Idx head_idx = threadIdx.y; head_idx < num_heads;
         head_idx += blockDim.y) {
      for (Idx feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
           feat_idx < hidden_xlen; feat_idx += blockDim.x * gridDim.x) {
        // DType s = 0.;
        DType sfeatsrc = 0.;
        Idx feat_src_offset = -1;
        // Idx el_idx = -1;
        if constexpr (CompactAsOfNodeFlag && !RelationalFlag) {
          // in this case, feat_src_offset is the same regardless of which
          // outgoing edge we deal with
          feat_src_offset =
              src_vid * gdata.feat_src_xlen + head_idx * hidden_xlen + feat_idx;
          // el_idx = src_vid * num_heads + head_idx;
        }
        // for (Idx e = start_off; e < end_off; ++e) {
        Idx eid = gdata.eids[e];
        Idx dst_vid = column_indices[e];
        // Idx er_idx = -1;
        // Idx dst_vid_relational = -1;
        if constexpr (!CompactAsOfNodeFlag) {
          // in this case, feat_src_offset, er_idx and el_idx are related to
          // edge id, regardless of the type of the edge
          feat_src_offset =
              eid * gdata.feat_src_xlen + head_idx * hidden_xlen + feat_idx;
          // er_idx = eid * num_heads + head_idx;
          // el_idx = eid * num_heads + head_idx;
        } else {  // CompactAsOfNodeFlag
          // if constexpr (!RelationalFlag) {
          //   er_idx = dst_vid * num_heads + head_idx;
          // } else {  // RelationalFlag
          if constexpr (RelationalFlag) {
            // in this case, er_idx (sum's index) is related to (relation,
            // unique node index) el_idx is related to (relation, unique node
            // index) feat_src_offset is related to (relation, unique node
            // index)
            // Idx etype = etypes[e];
            Idx etype = -1;
            if constexpr (ETypeRelPtrFlag) {
              etype = binary_search(num_relations, etypes, e);
            } else {
              etype = etypes[e];
            }
            // dst_vid_relational = find_relational_compact_as_of_node_index(
            //     etype, dst_vid, unique_srcs_and_dests_rel_ptr,
            //     unique_srcs_and_dests_node_indices);
            // er_idx = dst_vid_relational * num_heads + head_idx;
            Idx src_vid_relational = find_relational_compact_as_of_node_index(
                etype, src_vid, unique_srcs_and_dests_rel_ptr,
                unique_srcs_and_dests_node_indices);
            // el_idx = src_vid_relational * num_heads + head_idx;

            feat_src_offset = src_vid_relational * gdata.feat_src_xlen +
                              head_idx * hidden_xlen + feat_idx;
            // printf(
            //     "src_vid %ld dst_vid %ld etype %ld src_vid_relational %ld "
            //     "dst_vid_relational %ld \n",
            //     src_vid, dst_vid, etype, src_vid_relational,
            //     dst_vid_relational);
          }
        }

        Idx edge_offset = eid * num_heads + head_idx;

        Idx dst_out_offset =
            dst_vid * gdata.feat_src_xlen + head_idx * hidden_xlen + feat_idx;
        // DType grad_exp =
        //     gdata.grad_out[dst_out_offset] *
        //     (gdata.feat_src[feat_src_offset] - gdata.ret[dst_out_offset]) /
        //     gdata.sum[dst_vid * num_heads + head_idx];
        // DType tmp_sum = gdata.el[el_idx] + gdata.er[er_idx];
        // DType tmp2 = grad_exp * gdata.exp[edge_offset] *
        //              gradLeaky(tmp_sum, gdata.leaky_relu_slope);

        // Idx sum_vid = dst_vid;
        // if constexpr (RelationalFlag && CompactAsOfNodeFlag) {
        //   sum_vid = dst_vid_relational;
        // }
        gdata.grad_feat_dst[(dst_vid * gdata.feat_src_xlen +
                             head_idx * hidden_xlen + feat_idx)] =
            gdata.grad_inner_product[eid * num_heads + head_idx] *
            gdata.feat_src[feat_src_offset];
        if constexpr (!CompactAsOfNodeFlag || RelationalFlag) {
          gdata.grad_feat_src[feat_src_offset] =
              gdata.grad_inner_product[eid * num_heads + head_idx] *
              gdata.feat_dst[dst_vid * gdata.feat_src_xlen +
                             head_idx * hidden_xlen + feat_idx];
        } else {
          sfeatsrc += gdata.grad_inner_product[eid * num_heads + head_idx] *
                      gdata.feat_dst[dst_vid * gdata.feat_src_xlen +
                                     head_idx * hidden_xlen + feat_idx];

        }  // if constexpr (!CompactAsOfNodeFlag)
        if constexpr (CompactAsOfNodeFlag && !RelationalFlag) {
          gdata.grad_feat_src[feat_src_offset] = sfeatsrc;
        }
        //}    // for Idx e

      }  // while feat_idx
    }    // while head_idx
  }      // while src_vid
}
