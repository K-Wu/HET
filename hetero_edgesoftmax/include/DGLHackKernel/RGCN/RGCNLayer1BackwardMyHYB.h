#pragma once

#include <cuda_runtime.h>

template <typename Idx, typename DType, int ELL_logical_width,
          int ELL_physical_width>
__global__ void HET_Seastar_RgcnLayer1BackwardMyHYBKernelImpl(
    const Idx* ellcolidx_data, const Idx* ellreltype_data,
    const Idx* elleids_data, Idx* ranges, Idx* dst_ids, Idx* eids, Idx* types,
    DType* hidden, DType* weight, DType* norm, DType* grad_out,
    DType* grad_hidden, DType* grad_weight, Idx num_nodes, Idx feat_len_y,
    Idx feat_len_x, Idx ntypes) {
  if (blockIdx.x < num_nodes) {
    // ell portion
    Idx ellbeg = ELL_physical_width * blockIdx.x;
    Idx ellend = ELL_physical_width * blockIdx.x + ELL_logical_width;
    Idx tx = threadIdx.x;
    for (; tx < feat_len_x * feat_len_y; tx += blockDim.x) {
      Idx ty = tx / feat_len_x;
      Idx th = tx % feat_len_x;
      DType h = __ldg(hidden + blockIdx.x * feat_len_y + ty);
      DType agg = 0.;
      for (; ellbeg < ellend; ellbeg++) {
        Idx dst_id = __ldg(ellcolidx_data + ellbeg);
        if (dst_id ==
            MyHyb_NONEXISTENT_ELEMENT)  // TODO: check if in transposed hyb
                                        // dst_id is uninitalized and is
                                        // MyHyb_NONEXISTENT_ELEMENT
          break;
        Idx eid = __ldg(elleids_data + ellbeg);
        Idx type_id = __ldg(ellreltype_data + ellbeg);
        DType g = __ldg(grad_out + dst_id * feat_len_x + th);
        DType w = __ldg(weight + type_id * feat_len_y * feat_len_x + tx);
        DType n = __ldg(norm + eid);
        agg += g * w * n;
        atomicAdd(grad_weight + type_id * feat_len_y * feat_len_x + tx,
                  g * h * n);
      }
      atomicAdd(grad_hidden + blockIdx.x * feat_len_y + ty, agg);
    }

    // csr portion
    Idx beg = __ldg(ranges + blockIdx.x);
    Idx end = __ldg(ranges + blockIdx.x + 1);
    // Idx tx = threadIdx.x;
    for (; tx < feat_len_x * feat_len_y; tx += blockDim.x) {
      Idx ty = tx / feat_len_x;
      Idx th = tx % feat_len_x;
      DType h = __ldg(hidden + blockIdx.x * feat_len_y + ty);
      DType agg = 0.;
      for (; beg < end; beg++) {
        Idx dst_id = __ldg(dst_ids + beg);
        Idx eid = __ldg(eids + beg);
        Idx type_id = __ldg(types + beg);
        DType g = __ldg(grad_out + dst_id * feat_len_x + th);
        DType w = __ldg(weight + type_id * feat_len_y * feat_len_x + tx);
        DType n = __ldg(norm + eid);
        agg += g * w * n;
        atomicAdd(grad_weight + type_id * feat_len_y * feat_len_x + tx,
                  g * h * n);
      }
      atomicAdd(grad_hidden + blockIdx.x * feat_len_y + ty, agg);
    }
  }
}
