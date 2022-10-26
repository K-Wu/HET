#pragma once
#include "hetero_edgesoftmax.h"

template <int num_max_relations>
__device__ __forceinline__ void _perRow_EdgeSoftmaxFirstStageConcatenatedCSCKernel(int num_relations, int col_idx, float **__restrict__ outNodes_per_relation, int num_cols, int nnz, int *__restrict__ matCols, int *__restrict__ matRows, int *__restrict__ relation,
                                                                                   float *__restrict__ edge_input_data, float *__restrict__ mus)
{
    //@@ insert spmv kernel for csc format
    if (col_idx >= num_cols)
        return;
    int col_start = matCols[col_idx];
    int col_end = matCols[col_idx + 1];
    //float row_sum = 1e-10f;
    assert(num_relations < num_max_relations);
    float val[num_max_relations] = {0.0f};
    for (int edge_idx = col_start; edge_idx < col_end; edge_idx++)
    {
        val[relation[edge_idx]] += expf(edge_input_data[edge_idx]) + 1e-10f;
    }
    for (int idx_relation = 0; idx_relation < num_relations; idx_relation++)
    {
        outNodes_per_relation[idx_relation][col_idx] = val[idx_relation];
    }
}

__device__ __forceinline__ void _perRow_EdgeSoftmaxSecondStageConcatenatedCSCKernel(int col_idx, float *__restrict__ outEdge, float **__restrict__ outNodes_per_relation, int num_cols, int nnz, int *__restrict__ matCols, int *__restrict__ matRows,
                                                                                    int *__restrict__ relation, float *__restrict__ edge_input_data, float *__restrict__ mus)
{
    //@@ insert spmv kernel for csc format
    if (col_idx >= num_cols)
        return;
    int col_start = matCols[col_idx];
    int col_end = matCols[col_idx + 1];

    for (int edge_idx = col_start; edge_idx < col_end; edge_idx++)
    {
        float val = mus[relation[edge_idx]] * expf(edge_input_data[edge_idx]) / outNodes_per_relation[relation[edge_idx]][col_idx];
        outEdge[edge_idx] = val;
    }
}

template <int num_max_relations>
__device__ __forceinline__ void _EdgeSoftmaxFirstStageConcatenatedCSCKernel(int num_relations, int beg_col_idx, int stride, float **__restrict__ outNodes_per_relation, int num_cols, int nnz, int *__restrict__ matCols, int *__restrict__ matRows, int *__restrict__ matRelation,
                                                                            float *__restrict__ edge_input_data, float *__restrict__ mus)
{
    for (int col_idx = beg_col_idx; col_idx < num_cols; col_idx += stride)
    {
        _perRow_EdgeSoftmaxFirstStageConcatenatedCSCKernel<num_max_relations>(num_relations, col_idx, outNodes_per_relation, num_cols, nnz, matCols, matRows, matRelation, edge_input_data, mus);
    }
}

__device__ __forceinline__ void _EdgeSoftmaxSecondStageConcatenatedCSCKernel(int beg_col_idx, int stride, float *__restrict__ outEdge, float **__restrict__ outNodes_per_relation, int num_cols, int nnz, int *__restrict__ matCols, int *__restrict__ matRows, int *__restrict__ matRelation,
                                                                             float *__restrict__ edge_input_data, float *__restrict__ mus)
{
    for (int col_idx = beg_col_idx; col_idx < num_cols; col_idx += stride)
    {
        _perRow_EdgeSoftmaxSecondStageConcatenatedCSCKernel(col_idx, outEdge, outNodes_per_relation, num_cols, nnz, matCols, matRows, matRelation, edge_input_data, mus);
    }
}

template <int num_max_relations>
__global__ void EdgeSoftmaxConcatenatedCSCKernel(int num_relations, float *__restrict__ outEdges, float **__restrict__ outNodes_per_relation, int num_cols, int nnz, int *__restrict__ matCols, int *__restrict__ matRows, int *__restrict__ matRelation,
                                                 float *__restrict__ edge_input_data, float *__restrict__ mus)
{
    int beg_col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    _EdgeSoftmaxFirstStageConcatenatedCSCKernel<num_max_relations>(num_relations, beg_col_idx, blockDim.x * gridDim.x, outNodes_per_relation, num_cols, nnz, matCols, matRows, matRelation, edge_input_data, mus);
    _EdgeSoftmaxSecondStageConcatenatedCSCKernel(beg_col_idx, blockDim.x * gridDim.x, outEdges, outNodes_per_relation, num_cols, nnz, matCols, matRows, matRelation, edge_input_data, mus);
}

std::vector<thrust::device_vector<float>> doGPUEdgeSoftmaxConcatenatedCSCKernel(cusp::csr_matrix<int, int, cusp::device_memory> concatenated_csc_matrix, int num_relations, bool FlagInitWithRandomValue)
{

    std::vector<thrust::device_vector<float>> outNodes_per_relation_vect_vect(num_relations, thrust::device_vector<float>(concatenated_csc_matrix.num_cols, 0));
    thrust::device_vector<float *> outNodes_per_relation_vect;

    for (int idx_relation = 0; idx_relation < num_relations; idx_relation++)
    {
        //thrust::device_vector<float> outEdge_vect_for_curr_relation(concatenated_csc_matrix.num_cols, 0);
        //outNodes_per_relation_vect_vect.push_back(outEdge_vect_for_curr_relation);
        //outNodes_per_relation_vect.push_back(thrust::raw_pointer_cast(outEdge_vect_for_curr_relation.data()));
        std::cout << thrust::raw_pointer_cast(outNodes_per_relation_vect_vect[idx_relation].data()) << std::endl;
        outNodes_per_relation_vect.push_back(thrust::raw_pointer_cast(outNodes_per_relation_vect_vect[idx_relation].data()));
    }

    float *outEdges;
    float *mus;
    float *edge_input_data;
    cudaMalloc((void **)&edge_input_data, sizeof(float) * concatenated_csc_matrix.column_indices.size());
    cudaMalloc((void **)&mus, sizeof(float) * num_relations);
    cudaMalloc((void **)&outEdges, sizeof(float) * concatenated_csc_matrix.column_indices.size());

    if (FlagInitWithRandomValue)
    {
        curandGenerator_t m_prng;
        //Create a new generator
        curandCreateGenerator(&m_prng, CURAND_RNG_PSEUDO_DEFAULT);
        //Set the generator options
        curandSetPseudoRandomGeneratorSeed(m_prng, (unsigned long)0);
        //Generate random numbers
        curandGenerateUniform(m_prng, mus, num_relations);
        curandGenerateUniform(m_prng, edge_input_data, concatenated_csc_matrix.column_indices.size());
    }
    else
    {
        cudaMemset(mus, 1, sizeof(float) * num_relations);
        cudaMemset(edge_input_data, 1, sizeof(float) * concatenated_csc_matrix.column_indices.size());
    }
    dim3 block(RTX_3090_BLOCKSIZE, 1, 1);
    dim3 grid(RTX_3090_GRIDSIZE, 1, 1);
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    EdgeSoftmaxConcatenatedCSCKernel<NUM_MAX_RELATIONS><<<grid, block>>>(num_relations, outEdges, thrust::raw_pointer_cast(outNodes_per_relation_vect.data()), concatenated_csc_matrix.num_rows, concatenated_csc_matrix.column_indices.size(), thrust::raw_pointer_cast(concatenated_csc_matrix.row_offsets.data()), thrust::raw_pointer_cast(concatenated_csc_matrix.column_indices.data()), thrust::raw_pointer_cast(concatenated_csc_matrix.values.data()), edge_input_data, mus);
    cuda_err_chk(cudaPeekAtLastError());
    cuda_err_chk(cudaDeviceSynchronize());
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "GPU doGPUEdgeSoftmaxConcatenatedCSCKernel time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " us" << std::endl;
    cudaFree(outEdges);
    cudaFree(edge_input_data);
    cudaFree(mus);
    return outNodes_per_relation_vect_vect;
}