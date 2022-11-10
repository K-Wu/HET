#pragma once
#include <assert.h>
#include "DGLHackKernel/DGLHackKernel.h"
#define CUDA_MAX_NUM_THREADS 1024

inline int SeastarFindNumThreads(int dim,
                                 int max_nthrs = CUDA_MAX_NUM_THREADS) {
  assert(dim >= 0);
  if (dim == 0) return 1;
  int ret = max_nthrs;
  while (ret > dim) {
    ret = ret >> 1;
  }
  return ret;
}

int64_t SeastarComputeXLength(at::Tensor& tensor) {
  int64_t ret = 1;
  for (int i = 1; i < tensor.dim(); ++i) {
    ret *= tensor.size(i);
  }
  return ret;
}
