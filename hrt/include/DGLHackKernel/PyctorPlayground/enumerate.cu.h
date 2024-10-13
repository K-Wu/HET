#pragma once
#include <cuda_runtime.h>

#include "kernel_enums.h"
#include "utils.cu.h"

template <typename Idx, typename DType>
struct PYCTORInnerProductData {
  // feat_size size along feature dimension
  Idx feat_src_xlen{0};
  Idx num_heads{0};
  Idx *__restrict__ eids{nullptr};
  // Inputs
  DType *__restrict__ feat_src{nullptr}, *__restrict__ feat_dst{nullptr};
  // Output
  DType *__restrict__ edge_inner_product{nullptr};
};

// simplified adaption from HET_inner_product_fw_kernel_edge_parallel in
// [[hrt/include/DGLHackKernel/RGNN/InnerProductEdgeParallel.cu.h]]
template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool ETypeRelPtrFlag>
__global__ void HET_PYCTOR_inner_product_fw_kernel_edge_parallel(
    PYCTORInnerProductData<Idx, DType> gdata, const Idx *row_indices,
    const Idx *column_indices, const ETypeData<Idx, ETypeRelPtrFlag> etype_data,
    int64_t num_edges, const ETypeMapperData<Idx, kind> etype_mapper_data) {
  /// PYCTOR: Prologue
  constexpr bool EtypeRelPtrIndexSearch = true;
  Idx resume_from = 0;
  Idx num_heads = gdata.num_heads;
  // PYCTOR: Only for type 2 schedule
  Idx hidden_xlen = gdata.feat_src_xlen / num_heads;

  /// PYCTOR: Main loop
  /// PYCTOR: Loop header and prologue
  for (Idx eidx = blockIdx.y; eidx < num_edges; eidx += gridDim.y) {
    /// PYCTOR: Edge enumeration loop body prologue
    Idx dst_vid = row_indices[eidx];
    Idx edata_idx = gdata.eids[eidx];
    Idx src_vid = column_indices[eidx];

    for (Idx head_idx = threadIdx.y; head_idx < num_heads;
         head_idx += blockDim.y) {
      for (Idx feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
           feat_idx < hidden_xlen; feat_idx += blockDim.x * gridDim.x) {
        /// PYCTOR: Loop scalar template definition
        DType s = 0.;

        /// PYCTOR: Index scheme for edgewise vector
        Idx etype = -1;
        if constexpr (IsCompact(kind)) {
          // TODO: etype is not needed if etype_mapper_data's kind is subject
          // to !IsBinarySearch(kind)
          if constexpr (ETypeRelPtrFlag) {
            if constexpr (EtypeRelPtrIndexSearch) {
              etype = linear_search(etype_data.num_relations, etype_data.etypes,
                                    eidx, resume_from);
              resume_from = etype;
            } else {
              etype = binary_search(etype_data.num_relations, etype_data.etypes,
                                    eidx);
            }
          } else {
            etype = etype_data.etypes[eidx];
          }
        }
        Idx feat_src_entry_id = -1;
        if constexpr (IsCompact(kind)) {
          // TODO: etype is not needed if etype_mapper_data's kind is subject
          // to !IsBinarySearch(kind)
          feat_src_entry_id = find_relational_compact_as_of_node_index(
              etype, src_vid, edata_idx, etype_mapper_data);

        } else {
          // NB: we need to use edata_idx instead of eidx here
          feat_src_entry_id = edata_idx;
        }

        /// PYCTOR: operation statement
        s += gdata.feat_dst[dst_vid * gdata.feat_src_xlen +
                            head_idx * hidden_xlen + feat_idx] *
             gdata.feat_src[feat_src_entry_id * gdata.feat_src_xlen +
                            head_idx * hidden_xlen + feat_idx];

        /// PYCTOR: Warp reduction
        for (Idx sum_idx = hidden_xlen; sum_idx > 0; sum_idx >>= 1) {
          s += __shfl_down_sync(0xffffffff, s, sum_idx);
        }
        if (threadIdx.x % hidden_xlen == 0) {
          gdata.edge_inner_product[edata_idx * num_heads + head_idx] = s;
        }
      }
    }
  }
}
