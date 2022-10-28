#pragma once
#include "cuda.h"
#include "cuda_runtime.h"

__device__ __forceinline__ int binary_search(int num_elements,
                                             int *__restrict__ arr,
                                             int target) {
  int lo = 0, hi = num_elements;
  // find element in arr[i] where i in [lo, hi)
  // This below check covers all cases , so need to check
  // for mid=lo-(hi-lo)/2
  while (hi - lo > 1) {
    int mid = (hi + lo) / 2;
    if (arr[mid] <= target) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}
