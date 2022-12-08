#pragma once
#include "DGLHackKernel/DGLHackKernel.h"

// bgs:
template <typename Idx, typename DType, int ELL_logical_width,
          int ELL_physical_width>
__global__ void HET_RgcnLayer1MyHYBKernelImpl(
    const Idx* ellcolidx_data, const Idx* ellreltype_data,
    const Idx* elleids_data, const Idx* ranges, const Idx* src_ids,
    const Idx* eids, const Idx* types, const DType* hidden, const DType* weight,
    const DType* norm, DType* ret, Idx num_nodes, Idx feat_len_y,
    Idx feat_len_x, Idx ntypes) {
  // ell portion
  if (blockIdx.x < num_nodes) {
    Idx ell_beg = ELL_physical_width * blockIdx.x;
    Idx ell_end = ELL_physical_width * blockIdx.x + ELL_logical_width;
    Idx tx = threadIdx.x;
    Idx ty = threadIdx.x / feat_len_x;
    Idx th = threadIdx.x % feat_len_x;
    DType agg_val = 0.;
    DType w = 0.;
    Idx cur_type_id = -1;
    for (; ell_beg < ell_end; ell_beg++) {
      Idx src_id = __ldg(ellcolidx_data + ell_beg);
      if (src_id == MyHyb_NONEXISTENT_ELEMENT) break;
      Idx eid = __ldg(elleids_data + ell_beg);
      Idx type_id = __ldg(ellreltype_data + ell_beg);
      if (type_id != cur_type_id) {
        w = __ldg(weight + type_id * feat_len_y * feat_len_x + tx);
      }
      DType h = __ldg(hidden + src_id * feat_len_y + ty);
      DType n = __ldg(norm + eid);
      agg_val += h * w * n;
    }
    // atomicAdd(ret + blockIdx.x*feat_len_x + th, agg_val);
    //}

    // csr portion
    // if (blockIdx.x < num_nodes) {
    Idx beg = __ldg(ranges + blockIdx.x);
    Idx end = __ldg(ranges + blockIdx.x + 1);
    // Idx tx = threadIdx.x;
    // Idx ty = threadIdx.x / feat_len_x;
    // Idx th = threadIdx.x % feat_len_x;
    // DType agg_val = 0.;
    // DType w = 0.;
    // Idx cur_type_id = -1;
    for (; beg < end; beg++) {
      Idx src_id = __ldg(src_ids + beg);
      Idx eid = __ldg(eids + beg);
      Idx type_id = __ldg(types + beg);
      if (type_id != cur_type_id) {
        w = __ldg(weight + type_id * feat_len_y * feat_len_x + tx);
      }
      DType h = __ldg(hidden + src_id * feat_len_y + ty);
      DType n = __ldg(norm + eid);
      agg_val += h * w * n;
    }
    atomicAdd(ret + blockIdx.x * feat_len_x + th, agg_val);
  }
}
