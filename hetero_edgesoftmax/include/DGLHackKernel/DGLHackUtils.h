#pragma once
#include "DGLHackKernel.h"
#include <assert.h>
#define CUDA_MAX_NUM_THREADS 1024



inline int FindNumThreads(int dim, int max_nthrs = CUDA_MAX_NUM_THREADS) {
  assert(dim>= 0);
  if (dim == 0)
    return 1;
  int ret = max_nthrs;
  while (ret > dim) {
    ret = ret >> 1;
  }
  return ret;
}