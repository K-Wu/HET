#pragma once
#include "hrt.h"

__device__ __forceinline__ void
_perRow_EdgeSoftmax_4FirstStageConcatenatedCOOKernel(
    int edge_idx, float4 **__restrict__ outNodes_per_relation, int nnz,
    int *__restrict__ matCols, int *__restrict__ relation,
    float4 *__restrict__ edge_input_data, float4 *__restrict__ mus) {
  //@@ insert spmv kernel for coo format

  int col = matCols[edge_idx];

  float val1 = expf(edge_input_data[edge_idx].x) + 1e-10f;
  atomicAdd(&outNodes_per_relation[relation[edge_idx]][col].x, val1);

  float val2 = expf(edge_input_data[edge_idx].y) + 1e-10f;
  atomicAdd(&outNodes_per_relation[relation[edge_idx]][col].y, val2);

  float val3 = expf(edge_input_data[edge_idx].z) + 1e-10f;
  atomicAdd(&outNodes_per_relation[relation[edge_idx]][col].z, val3);

  float val4 = expf(edge_input_data[edge_idx].w) + 1e-10f;
  atomicAdd(&outNodes_per_relation[relation[edge_idx]][col].w, val4);
}

__device__ __forceinline__ void
_perRow_EdgeSoftmax_4SecondStageConcatenatedCOOKernel(
    int edge_idx, float4 *__restrict__ outEdge,
    float4 **__restrict__ outNodes_per_relation, int nnz,
    int *__restrict__ matCols, int *__restrict__ relation,
    float4 *__restrict__ edge_input_data, float4 *__restrict__ mus) {
  //@@ insert spmv kernel for coo format

  int col = matCols[edge_idx];
  float4 vals;
  float4 edge_input_data_curr =
      *reinterpret_cast<float4 *>(&edge_input_data[edge_idx]);
  float4 mus_curr = *reinterpret_cast<float4 *>(&mus[relation[edge_idx]]);
  float4 outNodes_curr = *reinterpret_cast<float4 *>(
      &outNodes_per_relation[relation[edge_idx]][col]);
  // val = mus[relation[edge_idx]] * expf(edge_input_data[edge_idx]) /
  // outNodes_per_relation[relation[edge_idx]][col];
  vals.x = mus_curr.x * expf(edge_input_data_curr.x) / outNodes_curr.x;
  vals.y = mus_curr.y * expf(edge_input_data_curr.y) / outNodes_curr.y;
  vals.z = mus_curr.z * expf(edge_input_data_curr.z) / outNodes_curr.z;
  vals.w = mus_curr.w * expf(edge_input_data_curr.w) / outNodes_curr.w;
  outEdge[edge_idx] = vals;
}

__device__ __forceinline__ void _EdgeSoftmax_4FirstStageConcatenatedCOOKernel(
    int beg_edge_idx, int stride, float4 **__restrict__ outNodes_per_relation,
    int nnz, int *__restrict__ matCols, int *__restrict__ matRelation,
    float4 *__restrict__ edge_input_data, float4 *__restrict__ mus) {
  for (int edge_idx = beg_edge_idx; edge_idx < nnz; edge_idx += stride) {
    _perRow_EdgeSoftmax_4FirstStageConcatenatedCOOKernel(
        edge_idx, outNodes_per_relation, nnz, matCols, matRelation,
        edge_input_data, mus);
  }
}

__device__ __forceinline__ void _EdgeSoftmax_4SecondStageConcatenatedCOOKernel(
    int beg_edge_idx, int stride, float4 *__restrict__ outEdge,
    float4 **__restrict__ outNodes_per_relation, int nnz,
    int *__restrict__ matCols, int *__restrict__ matRelation,
    float4 *__restrict__ edge_input_data, float4 *__restrict__ mus) {
  for (int edge_idx = beg_edge_idx; edge_idx < nnz; edge_idx += stride) {
    _perRow_EdgeSoftmax_4SecondStageConcatenatedCOOKernel(
        edge_idx, outEdge, outNodes_per_relation, nnz, matCols, matRelation,
        edge_input_data, mus);
  }
}

__global__ void HET_EdgeSoftmax_4ConcatenatedCOOKernel(
    float4 *__restrict__ outEdges, float4 **__restrict__ outNodes_per_relation,
    int nnz, int *__restrict__ matCols, int *__restrict__ matRelation,
    float4 *__restrict__ edge_input_data, float4 *__restrict__ mus) {
  int beg_edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
  _EdgeSoftmax_4FirstStageConcatenatedCOOKernel(
      beg_edge_idx, blockDim.x * gridDim.x, outNodes_per_relation, nnz, matCols,
      matRelation, edge_input_data, mus);
  _EdgeSoftmax_4SecondStageConcatenatedCOOKernel(
      beg_edge_idx, blockDim.x * gridDim.x, outEdges, outNodes_per_relation,
      nnz, matCols, matRelation, edge_input_data, mus);
}

std::vector<thrust::device_vector<float4>>
_doGPUEdgeSoftmax_4ConcatenatedCOOKernel(
    int num_nodes,
    cusp::coo_matrix<int, int, cusp::device_memory>::column_indices_array_type
        concatenated_coo_matrix_column_indices,
    cusp::coo_matrix<int, int, cusp::device_memory>::values_array_type
        concatenated_coo_matrix_values,
    int num_relations, bool FlagInitWithRandomValue) {
  std::vector<thrust::device_vector<float4>> outNodes_per_relation_vect_vect(
      num_relations, thrust::device_vector<float4>(
                         num_nodes, make_float4(0.0f, 0.0f, 0.0f, 0.0f)));
  thrust::device_vector<float4 *> outNodes_per_relation_vect;

  for (int idx_relation = 0; idx_relation < num_relations; idx_relation++) {
    // thrust::device_vector<float>
    // outEdge_vect_for_curr_relation(concatenated_coo_matrix.num_rows, 0);
    // outNodes_per_relation_vect_vect.push_back(outEdge_vect_for_curr_relation);
    // outNodes_per_relation_vect.push_back(thrust::raw_pointer_cast(outEdge_vect_for_curr_relation.data()));
    std::cout << thrust::raw_pointer_cast(
                     outNodes_per_relation_vect_vect[idx_relation].data())
              << std::endl;
    outNodes_per_relation_vect.push_back(thrust::raw_pointer_cast(
        outNodes_per_relation_vect_vect[idx_relation].data()));
  }

  float4 *outEdges;
  float4 *mus;
  float4 *edge_input_data;
  cudaMalloc((void **)&edge_input_data,
             sizeof(float4) * concatenated_coo_matrix_column_indices.size());
  cudaMalloc((void **)&mus, sizeof(float4) * num_relations);
  cudaMalloc((void **)&outEdges,
             sizeof(float4) * concatenated_coo_matrix_column_indices.size());

  if (FlagInitWithRandomValue) {
    curandGenerator_t m_prng;
    // Create a new generator
    curandCreateGenerator(&m_prng, CURAND_RNG_PSEUDO_DEFAULT);
    // Set the generator options
    curandSetPseudoRandomGeneratorSeed(m_prng, (unsigned long)0);
    // Generate random numbers
    curandGenerateUniform(m_prng, reinterpret_cast<float *>(mus),
                          4 * num_relations);
    curandGenerateUniform(m_prng, reinterpret_cast<float *>(edge_input_data),
                          4 * concatenated_coo_matrix_column_indices.size());
  } else {
    cudaMemset(mus, 1, sizeof(float4) * num_relations);
    cudaMemset(edge_input_data, 1,
               sizeof(float4) * concatenated_coo_matrix_column_indices.size());
  }
  dim3 block(RTX_3090_BLOCKSIZE, 1, 1);
  dim3 grid(RTX_3090_GRIDSIZE, 1, 1);
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  HET_EdgeSoftmax_4ConcatenatedCOOKernel<<<grid, block>>>(
      outEdges, thrust::raw_pointer_cast(outNodes_per_relation_vect.data()),
      concatenated_coo_matrix_column_indices.size(),
      thrust::raw_pointer_cast(concatenated_coo_matrix_column_indices.data()),
      thrust::raw_pointer_cast(concatenated_coo_matrix_values.data()),
      edge_input_data, mus);
  cuda_err_chk(cudaPeekAtLastError());
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::cout
      << "GPU doGPUEdgeSoftmax_4ConcatenatedCOOKernel time: "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << " us" << std::endl;
  cudaFree(outEdges);
  cudaFree(edge_input_data);
  cudaFree(mus);
  return outNodes_per_relation_vect_vect;
}

std::vector<thrust::device_vector<float4>>
doGPUEdgeSoftmax_4ConcatenatedCOOKernel(
    cusp::coo_matrix<int, int, cusp::device_memory> concatenated_coo_matrix,
    int num_relations, bool FlagInitWithRandomValue) {
  return _doGPUEdgeSoftmax_4ConcatenatedCOOKernel(
      concatenated_coo_matrix.num_rows, concatenated_coo_matrix.column_indices,
      concatenated_coo_matrix.values, num_relations, FlagInitWithRandomValue);
}

std::vector<thrust::device_vector<float4>>
doGPUEdgeSoftmax_4ConcatenatedCOOKernel(
    cusp::csr_matrix<int, int, cusp::device_memory> concatenated_csr_matrix,
    int num_relations, bool FlagInitWithRandomValue) {
  return _doGPUEdgeSoftmax_4ConcatenatedCOOKernel(
      concatenated_csr_matrix.num_rows, concatenated_csr_matrix.column_indices,
      concatenated_csr_matrix.values, num_relations, FlagInitWithRandomValue);
}