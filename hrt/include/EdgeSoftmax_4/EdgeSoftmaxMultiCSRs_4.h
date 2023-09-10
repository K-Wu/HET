#pragma once
#include "hrt.h"

__device__ __forceinline__ void _perRow_EdgeSoftmax_4FirstStageCSRKernel(
    int row_idx, float *__restrict__ outNode, int num_rows, int nnz,
    int *__restrict__ matCols, int *__restrict__ matRows,
    float *__restrict__ edge_input_data, float mu) {
  //@@ insert spmv kernel for csr format
  if (row_idx >= num_rows) return;
  int row_start = matRows[row_idx];
  int row_end = matRows[row_idx + 1];
  // float row_sum = 1e-10f;
  for (int edge_idx = row_start; edge_idx < row_end; edge_idx++) {
    int col = matCols[edge_idx];
    float val = expf(edge_input_data[edge_idx]) + 1e-10f;
    atomicAdd(&outNode[col], val);
  }
}

__device__ __forceinline__ void _perRow_EdgeSoftmax_4SecondStageCSRKernel(
    int row_idx, float *__restrict__ outEdge, float *__restrict__ outNode,
    int num_rows, int nnz, int *__restrict__ matCols, int *__restrict__ matRows,
    float *__restrict__ edge_input_data, float mu) {
  //@@ insert spmv kernel for csr format
  if (row_idx >= num_rows) return;
  int row_start = matRows[row_idx];
  int row_end = matRows[row_idx + 1];

  for (int edge_idx = row_start; edge_idx < row_end; edge_idx++) {
    int col = matCols[edge_idx];
    float val = mu * expf(edge_input_data[edge_idx]) / outNode[col];
    outEdge[edge_idx] = val;
  }
}

__device__ __forceinline__ void _EdgeSoftmax_4FirstStageCSRKernel(
    int beg_row_idx, int stride, float *__restrict__ outNode, int num_rows,
    int nnz, int *__restrict__ matCols, int *__restrict__ matRows,
    float *__restrict__ edge_input_data, float mu) {
  for (int row_idx = beg_row_idx; row_idx < num_rows; row_idx += stride) {
    _perRow_EdgeSoftmax_4FirstStageCSRKernel(
        row_idx, outNode, num_rows, nnz, matCols, matRows, edge_input_data, mu);
  }
}

__device__ __forceinline__ void _EdgeSoftmax_4SecondStageCSRKernel(
    int beg_row_idx, int stride, float *__restrict__ outEdge,
    float *__restrict__ outNode, int num_rows, int nnz,
    int *__restrict__ matCols, int *__restrict__ matRows,
    float *__restrict__ edge_input_data, float mu) {
  for (int row_idx = beg_row_idx; row_idx < num_rows; row_idx += stride) {
    _perRow_EdgeSoftmax_4SecondStageCSRKernel(row_idx, outEdge, outNode,
                                              num_rows, nnz, matCols, matRows,
                                              edge_input_data, mu);
  }
}

// template <int BLOCK_PER_SM>
__global__ void HET_EdgeSoftmax_4MultiCSRsKernel(
    int *__restrict__ blockid_relation_id, int *__restrict__ beg_row_idxes,
    int *__restrict__ num_block_for_same_relation,
    float **__restrict__ outNode_per_relation, float *__restrict__ outEdge,
    int num_relations, int num_rows, int *__restrict__ nnzs,
    int **__restrict__ matCols, int **__restrict__ matRows,
    float *__restrict__ edge_input_data, float *__restrict__ mus) {
  // int smid = getsmid();
  // int relation_id = smid_relation_id[smid];
  int relation_id = blockid_relation_id[blockIdx.x];
  int offset = 0;
  for (int idx = 0; idx < relation_id; idx++) {
    offset += nnzs[idx];
  }
  _EdgeSoftmax_4FirstStageCSRKernel(
      beg_row_idxes[blockIdx.x] + threadIdx.x,
      num_block_for_same_relation[blockIdx.x] * blockDim.x,
      outNode_per_relation[relation_id], num_rows, nnzs[relation_id],
      matCols[relation_id], matRows[relation_id], &edge_input_data[offset],
      mus[relation_id]);
  _EdgeSoftmax_4SecondStageCSRKernel(
      beg_row_idxes[blockIdx.x] + threadIdx.x,
      num_block_for_same_relation[blockIdx.x] * blockDim.x, &outEdge[offset],
      outNode_per_relation[relation_id], num_rows, nnzs[relation_id],
      matCols[relation_id], matRows[relation_id], &edge_input_data[offset],
      mus[relation_id]);
}

std::vector<thrust::device_vector<float>> doGPUEdgeSoftmax_4MultiCSRsKernel(
    std::vector<cusp::csr_matrix<int, int, cusp::device_memory>> csr_matrices,
    bool FlagInitWithRandomValue) {
  thrust::device_vector<int *> matCols_vect =
      thrust::device_vector<int *>(csr_matrices.size());
  thrust::device_vector<int *> matRows_vect =
      thrust::device_vector<int *>(csr_matrices.size());
  thrust::device_vector<int> nnzs_vect;
  thrust::device_vector<int> blockid_relation_id_vect;
  thrust::device_vector<int> beg_row_idxes_vect;
  thrust::device_vector<int> num_blocks_for_same_relation_per_block_vect;
  std::vector<thrust::device_vector<float>> outNodes_per_relation_vect_vect(
      csr_matrices.size(),
      thrust::device_vector<float>(csr_matrices[0].num_rows, 0));
  thrust::device_vector<float *> outNodes_per_relation_vect;
  size_t total_nnzs = 0;
  for (int idx_relation = 0; idx_relation < csr_matrices.size();
       idx_relation++) {
    matCols_vect[idx_relation] = thrust::raw_pointer_cast(
        csr_matrices[idx_relation].column_indices.data());
    matRows_vect[idx_relation] =
        thrust::raw_pointer_cast(csr_matrices[idx_relation].row_offsets.data());
    nnzs_vect.push_back(csr_matrices[idx_relation].column_indices.size());
    total_nnzs += csr_matrices[idx_relation].column_indices.size();
  }

  for (int idx_relation = 0; idx_relation < csr_matrices.size();
       idx_relation++) {
    // thrust::device_vector<float>
    // outEdge_vect_for_curr_relation(csr_matrices[0].num_rows, 0);
    // outNodes_per_relation_vect_vect.push_back(outEdge_vect_for_curr_relation);
    // printf("%x\n",thrust::raw_pointer_cast(outEdge_vect_for_curr_relation.data()));
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
  cudaMalloc((void **)&edge_input_data, sizeof(float) * total_nnzs);
  cudaMalloc((void **)&mus, sizeof(float) * csr_matrices.size());
  cudaMalloc((void **)&outEdges, sizeof(float) * total_nnzs);

  if (FlagInitWithRandomValue) {
    curandGenerator_t m_prng;
    // Create a new generator
    curandCreateGenerator(&m_prng, CURAND_RNG_PSEUDO_DEFAULT);
    // Set the generator options
    curandSetPseudoRandomGeneratorSeed(m_prng, (unsigned long)0);
    // Generate random numbers
    curandGenerateUniform(m_prng, mus, csr_matrices.size());
    curandGenerateUniform(m_prng, edge_input_data, total_nnzs);
  } else {
    cudaMemset(mus, 1, sizeof(float) * csr_matrices.size());
    cudaMemset(edge_input_data, 1, sizeof(float) * total_nnzs);
  }

  std::vector<int> num_blocks_for_same_relation_vect;
  std::vector<int> num_blocks_for_all_prev_relation_vect;
  num_blocks_for_all_prev_relation_vect.push_back(0);
  for (int idx_relationship = 0; idx_relationship < csr_matrices.size();
       idx_relationship++) {
    int num_blocks_for_this_and_prev_relation = (idx_relationship + 1 + 0.0) /
                                                (csr_matrices.size() + 0.0) *
                                                RTX_3090_GRIDSIZE;
    num_blocks_for_all_prev_relation_vect.push_back(
        num_blocks_for_this_and_prev_relation);
  }
  for (int idx_relationship = 0; idx_relationship < csr_matrices.size();
       idx_relationship++) {
    num_blocks_for_same_relation_vect.push_back(
        num_blocks_for_all_prev_relation_vect[idx_relationship + 1] -
        num_blocks_for_all_prev_relation_vect[idx_relationship]);
  }
  num_blocks_for_all_prev_relation_vect.erase(
      num_blocks_for_all_prev_relation_vect.begin());
  int idx_curr_relation = 0;
  int curr_beg_row_idx = 0;
  for (int idx_block = 0; idx_block < RTX_3090_GRIDSIZE; idx_block++) {
    if (idx_curr_relation < num_blocks_for_all_prev_relation_vect.size() - 1 &&
        idx_block >= num_blocks_for_all_prev_relation_vect[idx_curr_relation]) {
      assert(curr_beg_row_idx / RTX_3090_BLOCKSIZE ==
             num_blocks_for_same_relation_vect[idx_curr_relation]);
      idx_curr_relation++;
      curr_beg_row_idx = 0;
    }
    blockid_relation_id_vect.push_back(idx_curr_relation);
    beg_row_idxes_vect.push_back(curr_beg_row_idx);
    curr_beg_row_idx += RTX_3090_BLOCKSIZE;
    num_blocks_for_same_relation_per_block_vect.push_back(
        num_blocks_for_same_relation_vect[idx_curr_relation]);
  }

  dim3 block(RTX_3090_BLOCKSIZE, 1, 1);
  dim3 grid(RTX_3090_GRIDSIZE, 1, 1);
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  HET_EdgeSoftmax_4MultiCSRsKernel<<<grid, block>>>(
      thrust::raw_pointer_cast(blockid_relation_id_vect.data()),
      thrust::raw_pointer_cast(beg_row_idxes_vect.data()),
      thrust::raw_pointer_cast(
          num_blocks_for_same_relation_per_block_vect.data()),
      thrust::raw_pointer_cast(outNodes_per_relation_vect.data()), outEdges,
      nnzs_vect.size(), csr_matrices[0].num_rows,
      thrust::raw_pointer_cast(nnzs_vect.data()),
      thrust::raw_pointer_cast(matCols_vect.data()),
      thrust::raw_pointer_cast(matRows_vect.data()), edge_input_data, mus);
  cuda_err_chk(cudaPeekAtLastError());
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::cout
      << "GPU HET_EdgeSoftmax_4MultiCSRsKernel time: "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << " us" << std::endl;
  cudaFree(outEdges);
  cudaFree(edge_input_data);
  cudaFree(mus);
  return outNodes_per_relation_vect_vect;
}