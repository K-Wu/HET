#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/transpose.h>
#include <thrust/copy.h>
#include <npy.hpp>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <chrono>
#include <curand.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#define RTX_3090_MAX_BLOCKSIZE 1024
#define RTX_3090_SM_NUM 82
#define NUM_MAX_RELATIONS 8

#define RTX_2070MQ_SM_NUM 36

#define RTX_3090_BLOCKSIZE (RTX_3090_MAX_BLOCKSIZE / 4)
#define RTX_3090_GRIDSIZE (RTX_3090_SM_NUM * 4)
