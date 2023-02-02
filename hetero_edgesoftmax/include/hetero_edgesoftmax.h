#pragma once
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
namespace cg = cooperative_groups;

#include <curand.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/transpose.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <chrono>
#include <npy.hpp>
#include <vector>

#include "utils.cu.h"

template <typename Iterator>
void print_range(const std::string &name, Iterator first, Iterator last) {
  // from thrust example
  typedef typename std::iterator_traits<Iterator>::value_type T;

  std::cout << name << ": (" << std::distance(first, last) << ")";
  thrust::copy(first, last, std::ostream_iterator<T>(std::cout, " "));
  std::cout << "\n";
}

#define WARP_SIZE (32)

#define RTX_3090_MAX_BLOCKSIZE 1024
#define RTX_3090_SM_NUM 82
#define NUM_MAX_RELATIONS 8

#define RTX_2070MQ_SM_NUM 36

#define RTX_3090_BLOCKSIZE (RTX_3090_MAX_BLOCKSIZE / 4)
#define RTX_3090_GRIDSIZE (RTX_3090_SM_NUM * 6)
