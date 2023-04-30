#pragma once
#include <tuple>
#include "../../utils.cu.h"
#include "../DGLHackUtils.h"
#include "device_launch_parameters.h"

// C++ 17: auto [nblks, nthrs] = get_type1_schedule();
std::tuple<dim3, dim3> get_type1_schedule(int64_t num_heads,
                                          int64_t num_rows_or_edges) {
  // Type 1 Schedule:
  // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-069c3c2c5a9041df2c9a0b01c9f28044c4d519d86c5ed2f859d0d74282967062L232-R233
  // head -> blockIdx.x * blockDim.x + threadIdx.x;
  // edge|node -> blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t MAX_NBLKS = 65535;
  const int64_t MAX_NTHRS = 1024;
  unsigned int nthrs_x = 1;
  unsigned int nthrs_y = 32;
  unsigned int nblks_x = (num_heads + nthrs_x - 1) / (nthrs_x);

  unsigned int nblks_y =
      std::min(ceil_div(num_rows_or_edges, (int64_t)nthrs_y), MAX_NBLKS);
  dim3 nblks(nblks_x, nblks_y);
  dim3 nthrs(nthrs_x, nthrs_y);
  return std::make_tuple(nblks, nthrs);
}

// C++ 17: auto [nblks, nthrs] = get_type2_schedule();
// feat_src_xlen is obtained by applying SeastarComputeXLength to the feature
// tensor
std::tuple<dim3, dim3> get_type2_schedule(int64_t num_heads,
                                          int64_t feat_src_xlen,
                                          int64_t num_rows_or_edges) {
  // NB: updated to Type 2 Schedule:
  // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
  // head -> threadIdx.y
  // node -> blockIdx.y
  // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x
  const int64_t MAX_NBLKS = 65535;
  const int64_t MAX_NTHRS = 1024;
  unsigned nthrs_y = SeastarFindNumThreads(num_heads, 64);
  unsigned nthrs_x =
      SeastarFindNumThreads(feat_src_xlen / num_heads, MAX_NTHRS / nthrs_y);
  unsigned nblks_x = 1;
  unsigned nblks_y = std::min(num_rows_or_edges, MAX_NBLKS);
  dim3 nthrs(nthrs_x, nthrs_y);
  dim3 nblks(nblks_x, nblks_y);
  return std::make_tuple(nblks, nthrs);
}
