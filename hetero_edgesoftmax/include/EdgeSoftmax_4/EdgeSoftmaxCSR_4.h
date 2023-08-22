#pragma once
#include "hetero_edgesoftmax.h"

__device__ __forceinline__ void
_perRow_EdgeSoftmax_4FirstStageConcatenatedCSRKernel(
    int row_idx, float **__restrict__ outNodes_per_relation, int num_rows,
    int nnz, int *__restrict__ matCols, int *__restrict__ matRows,
    int *__restrict__ relation, float *__restrict__ edge_input_data,
    float *__restrict__ mus) {
  //@@ insert spmv kernel for csr format
  if (row_idx >= num_rows)
    return;
  int row_start = matRows[row_idx];
  int row_end = matRows[row_idx + 1];
  // float row_sum = 1e-10f;
  for (int edge_idx = row_start; edge_idx < row_end; edge_idx++) {
    int col = matCols[edge_idx];
    float val = expf(edge_input_data[edge_idx]) + 1e-10f;
    atomicAdd(&outNodes_per_relation[relation[edge_idx]][col], val);
  }
}

__device__ __forceinline__ void
_perRow_EdgeSoftmax_4SecondStageConcatenatedCSRKernel(
    int row_idx, float *__restrict__ outEdge,
    float **__restrict__ outNodes_per_relation, int num_rows, int nnz,
    int *__restrict__ matCols, int *__restrict__ matRows,
    int *__restrict__ relation, float *__restrict__ edge_input_data,
    float *__restrict__ mus) {
  //@@ insert spmv kernel for csr format
  if (row_idx >= num_rows)
    return;
  int row_start = matRows[row_idx];
  int row_end = matRows[row_idx + 1];

  for (int edge_idx = row_start; edge_idx < row_end; edge_idx++) {
    int col = matCols[edge_idx];
    float val = mus[relation[edge_idx]] * expf(edge_input_data[edge_idx]) /
                outNodes_per_relation[relation[edge_idx]][col];
    outEdge[edge_idx] = val;
  }
}

__device__ __forceinline__ void _EdgeSoftmax_4FirstStageConcatenatedCSRKernel(
    int beg_row_idx, int stride, float **__restrict__ outNodes_per_relation,
    int num_rows, int nnz, int *__restrict__ matCols, int *__restrict__ matRows,
    int *__restrict__ matRelation, float *__restrict__ edge_input_data,
    float *__restrict__ mus) {
  for (int row_idx = beg_row_idx; row_idx < num_rows; row_idx += stride) {
    _perRow_EdgeSoftmax_4FirstStageConcatenatedCSRKernel(
        row_idx, outNodes_per_relation, num_rows, nnz, matCols, matRows,
        matRelation, edge_input_data, mus);
  }
}

__device__ __forceinline__ void _EdgeSoftmax_4SecondStageConcatenatedCSRKernel(
    int beg_row_idx, int stride, float *__restrict__ outEdge,
    float **__restrict__ outNodes_per_relation, int num_rows, int nnz,
    int *__restrict__ matCols, int *__restrict__ matRows,
    int *__restrict__ matRelation, float *__restrict__ edge_input_data,
    float *__restrict__ mus) {
  for (int row_idx = beg_row_idx; row_idx < num_rows; row_idx += stride) {
    _perRow_EdgeSoftmax_4SecondStageConcatenatedCSRKernel(
        row_idx, outEdge, outNodes_per_relation, num_rows, nnz, matCols,
        matRows, matRelation, edge_input_data, mus);
  }
}

__global__ void HET_EdgeSoftmax_4ConcatenatedCSRKernel(
    float *__restrict__ outEdges, float **__restrict__ outNodes_per_relation,
    int num_rows, int nnz, int *__restrict__ matCols, int *__restrict__ matRows,
    int *__restrict__ matRelation, float *__restrict__ edge_input_data,
    float *__restrict__ mus) {
  int beg_row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  _EdgeSoftmax_4FirstStageConcatenatedCSRKernel(
      beg_row_idx, blockDim.x * gridDim.x, outNodes_per_relation, num_rows, nnz,
      matCols, matRows, matRelation, edge_input_data, mus);
  _EdgeSoftmax_4SecondStageConcatenatedCSRKernel(
      beg_row_idx, blockDim.x * gridDim.x, outEdges, outNodes_per_relation,
      num_rows, nnz, matCols, matRows, matRelation, edge_input_data, mus);
}

std::vector<thrust::device_vector<float>>
doGPUEdgeSoftmax_4ConcatenatedCSRKernel(
    cusp::csr_matrix<int, int, cusp::device_memory> concatenated_csr_matrix,
    int num_relations, bool FlagInitWithRandomValue) {
  std::vector<thrust::device_vector<float>> outNodes_per_relation_vect_vect(
      num_relations,
      thrust::device_vector<float>(concatenated_csr_matrix.num_rows, 0));
  thrust::device_vector<float *> outNodes_per_relation_vect;

  for (int idx_relation = 0; idx_relation < num_relations; idx_relation++) {
    // thrust::device_vector<float>
    // outEdge_vect_for_curr_relation(concatenated_csr_matrix.num_rows, 0);
    // outNodes_per_relation_vect_vect.push_back(outEdge_vect_for_curr_relation);
    // outNodes_per_relation_vect.push_back(thrust::raw_pointer_cast(outEdge_vect_for_curr_relation.data()));
    std::cout << thrust::raw_pointer_cast(
                     outNodes_per_relation_vect_vect[idx_relation].data())
              << std::endl;
    outNodes_per_relation_vect.push_back(thrust::raw_pointer_cast(
        outNodes_per_relation_vect_vect[idx_relation].data()));
  }

  float *outEdges;
  float *mus;
  float *edge_input_data;
  cudaMalloc((void **)&edge_input_data,
             sizeof(float) * concatenated_csr_matrix.column_indices.size());
  cudaMalloc((void **)&mus, sizeof(float) * num_relations);
  cudaMalloc((void **)&outEdges,
             sizeof(float) * concatenated_csr_matrix.column_indices.size());

  if (FlagInitWithRandomValue) {
    curandGenerator_t m_prng;
    // Create a new generator
    curandCreateGenerator(&m_prng, CURAND_RNG_PSEUDO_DEFAULT);
    // Set the generator options
    curandSetPseudoRandomGeneratorSeed(m_prng, (unsigned long)0);
    // Generate random numbers
    curandGenerateUniform(m_prng, mus, num_relations);
    curandGenerateUniform(m_prng, edge_input_data,
                          concatenated_csr_matrix.column_indices.size());
  } else {
    cudaMemset(mus, 1, sizeof(float) * num_relations);
    cudaMemset(edge_input_data, 1,
               sizeof(float) * concatenated_csr_matrix.column_indices.size());
  }
  dim3 block(RTX_3090_BLOCKSIZE, 1, 1);
  dim3 grid(RTX_3090_GRIDSIZE, 1, 1);
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  HET_EdgeSoftmax_4ConcatenatedCSRKernel<<<grid, block>>>(
      outEdges, thrust::raw_pointer_cast(outNodes_per_relation_vect.data()),
      concatenated_csr_matrix.num_rows,
      concatenated_csr_matrix.column_indices.size(),
      thrust::raw_pointer_cast(concatenated_csr_matrix.column_indices.data()),
      thrust::raw_pointer_cast(concatenated_csr_matrix.row_offsets.data()),
      thrust::raw_pointer_cast(concatenated_csr_matrix.values.data()),
      edge_input_data, mus);
  cuda_err_chk(cudaPeekAtLastError());
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::cout
      << "GPU doGPUEdgeSoftmax_4ConcatenatedCSRKernel time: "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << " us" << std::endl;
  cudaFree(outEdges);
  cudaFree(edge_input_data);
  cudaFree(mus);
  return outNodes_per_relation_vect_vect;
}