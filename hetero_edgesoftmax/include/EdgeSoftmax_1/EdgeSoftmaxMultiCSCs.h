#pragma once
#include "hetero_edgesoftmax.h"

template <int num_max_relations>
__device__ __forceinline__ void _perRow_EdgeSoftmaxFirstStageCSCKernel(int num_relations, int col_idx, float *__restrict__ outNode, int num_cols, int nnz, int *__restrict__ matCols, int *__restrict__ matRows,
                                                                       float *__restrict__ edge_input_data, float mu)
{
    //@@ insert spmv kernel for csr format
    if (col_idx >= num_cols)
        return;
    int col_start = matCols[col_idx];
    int col_end = matCols[col_idx + 1];
    //float col_sum = 1e-10f;
    assert(num_relations < num_max_relations);
    float val = 0.0f;
    for (int edge_idx = col_start; edge_idx < col_end; edge_idx++)
    {
        //int col = matCols[edge_idx];
        //float val = expf(edge_input_data[edge_idx]) + 1e-10f;
        //atomicAdd(&outNode[col], val);
        val += (expf(edge_input_data[edge_idx]) + 1e-10f);
    }
    outNode[col_idx] = val;
}

__device__ __forceinline__ void _perRow_EdgeSoftmaxSecondStageCSCKernel(int col_idx, float *__restrict__ outEdge, float *__restrict__ outNode, int num_cols, int nnz, int *__restrict__ matCols, int *__restrict__ matRows,
                                                                        float *__restrict__ edge_input_data, float mu)
{
    //@@ insert spmv kernel for csr format
    if (col_idx >= num_cols)
        return;
    int col_start = matCols[col_idx];
    int col_end = matCols[col_idx + 1];

    for (int edge_idx = col_start; edge_idx < col_end; edge_idx++)
    {
        float val = mu * expf(edge_input_data[edge_idx]) / outNode[col_idx];
        outEdge[edge_idx] = val;
    }
}

template <int num_max_relations>
__device__ __forceinline__ void _EdgeSoftmaxFirstStageCSCKernel(int num_relations, int beg_col_idx, int stride, float *__restrict__ outNode, int num_cols, int nnz, int *__restrict__ matCols, int *__restrict__ matRows,
                                                                float *__restrict__ edge_input_data, float mu)
{
    for (int col_idx = beg_col_idx; col_idx < num_cols; col_idx += stride)
    {
        _perRow_EdgeSoftmaxFirstStageCSCKernel<num_max_relations>(num_relations, col_idx, outNode, num_cols, nnz, matCols, matRows, edge_input_data, mu);
    }
}

__device__ __forceinline__ void _EdgeSoftmaxSecondStageCSCKernel(int beg_col_idx, int stride, float *__restrict__ outEdge, float *__restrict__ outNode, int num_cols, int nnz, int *__restrict__ matCols, int *__restrict__ matRows,
                                                                 float *__restrict__ edge_input_data, float mu)
{
    for (int col_idx = beg_col_idx; col_idx < num_cols; col_idx += stride)
    {
        _perRow_EdgeSoftmaxSecondStageCSCKernel(col_idx, outEdge, outNode, num_cols, nnz, matCols, matRows, edge_input_data, mu);
    }
}

//template <int BLOCK_PER_SM>
template <int num_max_relations>
__global__ void EdgeSoftmaxMultiCSCsKernel(int *__restrict__ blockid_relation_id, int *__restrict__ beg_col_idxes, int *__restrict__ num_block_for_same_relation, float **__restrict__ outNode_per_relation, float *__restrict__ outEdge, int num_relations, int num_cols, int *__restrict__ nnzs, int **__restrict__ matCols, int **__restrict__ matRows,
                                           float *__restrict__ edge_input_data, float *__restrict__ mus)
{
    //int smid = getsmid();
    //int relation_id = smid_relation_id[smid];
    int relation_id = blockid_relation_id[blockIdx.x];
    int offset = 0;
    for (int idx = 0; idx < relation_id; idx++)
    {
        offset += nnzs[idx];
    }
    _EdgeSoftmaxFirstStageCSCKernel<num_max_relations>(num_relations, beg_col_idxes[blockIdx.x] + threadIdx.x, num_block_for_same_relation[blockIdx.x] * blockDim.x, outNode_per_relation[relation_id], num_cols, nnzs[relation_id], matCols[relation_id], matRows[relation_id], &edge_input_data[offset], mus[relation_id]);
    _EdgeSoftmaxSecondStageCSCKernel(beg_col_idxes[blockIdx.x] + threadIdx.x, num_block_for_same_relation[blockIdx.x] * blockDim.x, &outEdge[offset], outNode_per_relation[relation_id], num_cols, nnzs[relation_id], matCols[relation_id], matRows[relation_id], &edge_input_data[offset], mus[relation_id]);
}

std::vector<thrust::device_vector<float>> doGPUEdgeSoftmaxMultiCSCsKernel(std::vector<cusp::csr_matrix<int, int, cusp::device_memory>> csc_matrices, bool FlagInitWithRandomValue)
{
    thrust::device_vector<int *> matCols_vect = thrust::device_vector<int *>(csc_matrices.size());
    thrust::device_vector<int *> matRows_vect = thrust::device_vector<int *>(csc_matrices.size());
    thrust::device_vector<int> nnzs_vect;
    thrust::device_vector<int> blockid_relation_id_vect;
    thrust::device_vector<int> beg_col_idxes_vect;
    thrust::device_vector<int> num_blocks_for_same_relation_per_block_vect;
    std::vector<thrust::device_vector<float>> outNodes_per_relation_vect_vect(csc_matrices.size(), thrust::device_vector<float>(csc_matrices[0].num_rows, 0));
    thrust::device_vector<float *> outNodes_per_relation_vect;
    size_t total_nnzs = 0;
    for (int idx_relation = 0; idx_relation < csc_matrices.size(); idx_relation++)
    {

        matCols_vect[idx_relation] = thrust::raw_pointer_cast(csc_matrices[idx_relation].column_indices.data());
        matRows_vect[idx_relation] = thrust::raw_pointer_cast(csc_matrices[idx_relation].row_offsets.data());
        nnzs_vect.push_back(csc_matrices[idx_relation].column_indices.size());
        total_nnzs += csc_matrices[idx_relation].column_indices.size();
    }

    for (int idx_relation = 0; idx_relation < csc_matrices.size(); idx_relation++)
    {
        //thrust::device_vector<float> outEdge_vect_for_curr_relation(csc_matrices[0].num_cols, 0);
        //outNodes_per_relation_vect_vect.push_back(outEdge_vect_for_curr_relation);
        //printf("%x\n",thrust::raw_pointer_cast(outEdge_vect_for_curr_relation.data()));
        //outNodes_per_relation_vect.push_back(thrust::raw_pointer_cast(outEdge_vect_for_curr_relation.data()));

        std::cout << thrust::raw_pointer_cast(outNodes_per_relation_vect_vect[idx_relation].data()) << std::endl;
        outNodes_per_relation_vect.push_back(thrust::raw_pointer_cast(outNodes_per_relation_vect_vect[idx_relation].data()));
    }

    float *outEdges;
    float *mus;
    float *edge_input_data;
    cudaMalloc((void **)&edge_input_data, sizeof(float) * total_nnzs);
    cudaMalloc((void **)&mus, sizeof(float) * csc_matrices.size());
    cudaMalloc((void **)&outEdges, sizeof(float) * total_nnzs);

    if (FlagInitWithRandomValue)
    {
        curandGenerator_t m_prng;
        //Create a new generator
        curandCreateGenerator(&m_prng, CURAND_RNG_PSEUDO_DEFAULT);
        //Set the generator options
        curandSetPseudoRandomGeneratorSeed(m_prng, (unsigned long)0);
        //Generate random numbers
        curandGenerateUniform(m_prng, mus, csc_matrices.size());
        curandGenerateUniform(m_prng, edge_input_data, total_nnzs);
    }
    else
    {
        cudaMemset(mus, 1, sizeof(float) * csc_matrices.size());
        cudaMemset(edge_input_data, 1, sizeof(float) * total_nnzs);
    }

    std::vector<int> num_blocks_for_same_relation_vect;
    std::vector<int> num_blocks_for_all_prev_relation_vect;
    num_blocks_for_all_prev_relation_vect.push_back(0);
    for (int idx_relationship = 0; idx_relationship < csc_matrices.size(); idx_relationship++)
    {
        int num_blocks_for_this_and_prev_relation = (idx_relationship + 1 + 0.0) / (csc_matrices.size() + 0.0) * RTX_3090_GRIDSIZE;
        num_blocks_for_all_prev_relation_vect.push_back(num_blocks_for_this_and_prev_relation);
    }
    for (int idx_relationship = 0; idx_relationship < csc_matrices.size(); idx_relationship++)
    {
        num_blocks_for_same_relation_vect.push_back(num_blocks_for_all_prev_relation_vect[idx_relationship + 1] - num_blocks_for_all_prev_relation_vect[idx_relationship]);
    }
    num_blocks_for_all_prev_relation_vect.erase(num_blocks_for_all_prev_relation_vect.begin());
    int idx_curr_relation = 0;
    int curr_beg_col_idx = 0;
    for (int idx_block = 0; idx_block < RTX_3090_GRIDSIZE; idx_block++)
    {
        if (idx_curr_relation < num_blocks_for_all_prev_relation_vect.size() - 1 && idx_block >= num_blocks_for_all_prev_relation_vect[idx_curr_relation])
        {
            assert(curr_beg_col_idx / RTX_3090_BLOCKSIZE == num_blocks_for_same_relation_vect[idx_curr_relation]);
            idx_curr_relation++;
            curr_beg_col_idx = 0;
        }
        blockid_relation_id_vect.push_back(idx_curr_relation);
        beg_col_idxes_vect.push_back(curr_beg_col_idx);
        curr_beg_col_idx += RTX_3090_BLOCKSIZE;
        num_blocks_for_same_relation_per_block_vect.push_back(num_blocks_for_same_relation_vect[idx_curr_relation]);
    }

    dim3 block(RTX_3090_BLOCKSIZE, 1, 1);
    dim3 grid(RTX_3090_GRIDSIZE, 1, 1);
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    EdgeSoftmaxMultiCSCsKernel<NUM_MAX_RELATIONS><<<grid, block>>>(thrust::raw_pointer_cast(blockid_relation_id_vect.data()), thrust::raw_pointer_cast(beg_col_idxes_vect.data()), thrust::raw_pointer_cast(num_blocks_for_same_relation_per_block_vect.data()), thrust::raw_pointer_cast(outNodes_per_relation_vect.data()), outEdges, nnzs_vect.size(), csc_matrices[0].num_rows, thrust::raw_pointer_cast(nnzs_vect.data()), thrust::raw_pointer_cast(matRows_vect.data()), thrust::raw_pointer_cast(matCols_vect.data()), edge_input_data, mus);
    cuda_err_chk(cudaPeekAtLastError());
    cuda_err_chk(cudaDeviceSynchronize());
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "GPU EdgeSoftmaxMultiCSCsKernel time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " us" << std::endl;
    cudaFree(outEdges);
    cudaFree(edge_input_data);
    cudaFree(mus);
    return outNodes_per_relation_vect_vect;
}