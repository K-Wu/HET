#pragma once
#include "DGLHackKernel/DGLHackKernel.h"

// TODO: we can reuse the first stage of sWt to calculate heterogensous message. The only difference is that heteroegneous message is associated with source node rather than destination node. The unique trick to reduce computation may still apply but the unique op is applied to the source node rather than destination node.
// TODO: For sWt, we may do the unique op on the fly: each warp is assigned a large chunk of consecutive source node index, and each warp work coorperatively to figure out the non-zero-outgoing-edge source node before calculating sW.

// For now, we put MySimpleNDArray as members of each of the structure. Whenever we want to reduce redundant deep copy, we may use std::shared_ptr as examplified in the answer here https://stackoverflow.com/a/395158
// An example of initializing shared_ptr: https://godbolt.org/z/Yj86q3fEP backup: https://gist.github.com/K-Wu/141d949fd467ec7ff32e003ad0a5c5ce
struct HGTLayerExecPreprocessData{

};

struct HGTLayerIntermediateData{

};

struct HGTLayerInputData{

};

struct HGTLayerOutputData{

};

// extract this kernel with mysgemm_ into template specialization
// template <int NODE_INPUT_DIM_PER_HEAD/*derived from OUT_DIM and NUM_HEADS*/, NUM_HEADS, OUT_DIM, COARSE_SGEMM_NODES_PER_BLOCK>
template <int TILE_SZ_A, int TILE_SZ_B, int OUT_DIM, int NUM_HEADS>
__global__ void _global_EdgeMessageConcatenatedCOOKernel(float **__restrict__ intermediate_node_vect, int nnz, int *__restrict__ matCols, int *__restrict__ matRelation,
                                                                          float *__restrict__ node_input_data, float *__restrict__ relation_message_matrices, int **__restrict__ dest_node_to_unique_index_per_relation,  int *__restrict__ sizes_unique_index_to_dest_node_per_relation, int num_relations, int *__restrict__ num_blocks_xdim_for_same_relation_per_block_vect, int *__restrict__ beg_node_entry_idxes_vect, int *__restrict__ blockid_relation_id_vect)
{
    constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
    constexpr int COARSE_SGEMM_NODES_PER_BLOCK = (TILE_SZ_B);
    int beg_node_entry_idx = beg_node_entry_idxes_vect[blockIdx.x];
    int stride = num_blocks_xdim_for_same_relation_per_block_vect[blockIdx.x] * COARSE_SGEMM_NODES_PER_BLOCK;
    int relation_idx = blockid_relation_id_vect[blockIdx.x];

    for (int node_entry_idx = beg_node_entry_idx; node_entry_idx < sizes_unique_index_to_dest_node_per_relation[relation_idx]; node_entry_idx += stride)
    {
        mysgemm_functor<TILE_SZ_A, TILE_SZ_B, OUT_DIM, NUM_HEADS, false>::exec_function(OUT_DIM, sizes_unique_index_to_dest_node_per_relation[relation_idx], NODE_INPUT_DIM_PER_HEAD, &relation_message_matrices[relation_idx * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD * NODE_INPUT_DIM_PER_HEAD], node_input_data, intermediate_node_vect[relation_idx], nullptr, node_entry_idx);
    }
}

HGTLayerExecPreprocessData HGTLayerPreprocessing(int num_nodes, cusp::coo_matrix<int, int, cusp::device_memory>::row_indices_array_type concatenated_coo_matrix_row_indices, cusp::coo_matrix<int, int, cusp::device_memory>::column_indices_array_type concatenated_coo_matrix_column_indices, std::vector<cusp::coo_matrix<int, int, cusp::device_memory>::column_indices_array_type> coo_matrices_column_indices, cusp::coo_matrix<int, int, cusp::device_memory>::values_array_type concatenated_coo_matrix_values, int num_relations, bool FlagInitWithRandomValue){
    HGTLayerExecPreprocessData ret;

    constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
    constexpr int COARSE_SGEMM_BLOCKSIZE = (TILE_SZ_A);
    constexpr int COARSE_SGEMM_NODES_PER_BLOCK = (TILE_SZ_B);

    return ret;
}

// TODO: collect input data into a struct; malloc intermediate and output data.

// assume nodes indices are currently sorted according to their node type, or even only monotype exists.
// We use naive for loop outside kernel launch to do the linear layer for now.
// TODO: implement more general case where nodes may not be sorted according to node type, thus indirection needed
// TODO: optimize the for loop by fusing multiple kernels into one
// work for both k-linear and q-linear
LinearByNodeType(){

}




// NB: In this implementation, message generation is done for each (source node, relationship this node is involved) where each (source node, relationship this node is involved) is mapped to a unique (relationship id, unique node index) and referred to in the next stage. Notice getting this unique index mapping is O(|R||V|) complexity and stays the same throughout the whole execution. We can do this mapping in the first step and reuse it thereafter. In this case, the it is dense operation first with scatter operation implicitly done by succeeding operations.
// TODO: an alternative implementation is message generation for each edge where there might be redundant computation of (source node, relationship this node is involved) pairs. In this case, only the relationship type and source node index for each edge is needed. This is explicit scatter operation done first and then dense operation.
template <int TILE_SZ_A /*128*/, int TILE_SZ_B /*8*/, int OUT_DIM /*256*/, int NUM_HEADS /*4*/>
thrust::device_vector<float4> EdgeMessageConcatenatedCOOKernel(int num_nodes, cusp::coo_matrix<int, int, cusp::device_memory>::row_indices_array_type concatenated_coo_matrix_row_indices, cusp::coo_matrix<int, int, cusp::device_memory>::column_indices_array_type concatenated_coo_matrix_column_indices, std::vector<cusp::coo_matrix<int, int, cusp::device_memory>::column_indices_array_type> coo_matrices_column_indices, cusp::coo_matrix<int, int, cusp::device_memory>::values_array_type concatenated_coo_matrix_values, int num_relations, bool FlagInitWithRandomValue)
{
    constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
    constexpr int COARSE_SGEMM_BLOCKSIZE = (TILE_SZ_A);
    constexpr int COARSE_SGEMM_NODES_PER_BLOCK = (TILE_SZ_B);

    
    // preprocessed metadata: dest_node_to_unique_index_per_relation_d, unique_indices_to_column_indices_per_relation_d, num_unique_indices_to_column_indices_per_relation
    thrust::device_vector<float4> outEdges_vect(concatenated_coo_matrix_column_indices.size(), make_float4(0.0f, 0.0f, 0.0f, 0.0f));
    std::vector<cusp::coo_matrix<int, int, cusp::device_memory>::column_indices_array_type> coo_matrices_column_indices_unique(num_relations);
    thrust::device_vector<int *> unique_indices_to_column_indices_per_relation_d;
    thrust::device_vector<int> num_unique_indices_to_column_indices_per_relation(num_relations, -1);

    std::vector<thrust::device_vector<int>> dest_node_to_unique_index_per_relation = std::vector<thrust::device_vector<int>>(num_relations, thrust::device_vector<int>(num_nodes, -1));
    thrust::device_vector<int *> dest_node_to_unique_index_per_relation_d;
    for (int idx_relation = 0; idx_relation < num_relations; idx_relation++)
    {
        dest_node_to_unique_index_per_relation_d.push_back(thrust::raw_pointer_cast(dest_node_to_unique_index_per_relation[idx_relation].data()));
    }

    for (int idx_relation = 0; idx_relation < num_relations; idx_relation++)
    {
        coo_matrices_column_indices_unique[idx_relation] = coo_matrices_column_indices[idx_relation];
        std::cout << "curr_coo_matrix_column_indices_unique address" << thrust::raw_pointer_cast(coo_matrices_column_indices_unique[idx_relation].data()) << " original address: " << thrust::raw_pointer_cast(coo_matrices_column_indices[idx_relation].data()) << std::endl;
        thrust::sort(thrust::device, coo_matrices_column_indices_unique[idx_relation].begin(), coo_matrices_column_indices_unique[idx_relation].end());
        auto curr_unique_vector_end = thrust::unique(thrust::device, coo_matrices_column_indices_unique[idx_relation].begin(), coo_matrices_column_indices_unique[idx_relation].end());
        coo_matrices_column_indices_unique[idx_relation].resize(thrust::distance(coo_matrices_column_indices_unique[idx_relation].begin(), curr_unique_vector_end));
    }


    for (int idx_relation = 0; idx_relation < num_relations; idx_relation++)
    {
        thrust::counting_iterator<int> first_counting_iter(0);
        thrust::counting_iterator<int> last_counting_iter = first_counting_iter + coo_matrices_column_indices_unique[idx_relation].size();

        int *curr_dest_node_to_unique_index_per_relation_d = dest_node_to_unique_index_per_relation_d[idx_relation];
        thrust::for_each(thrust::device, thrust::make_zip_iterator(thrust::make_tuple(coo_matrices_column_indices_unique[idx_relation].begin(), first_counting_iter)),
                         thrust::make_zip_iterator(thrust::make_tuple(coo_matrices_column_indices_unique[idx_relation].end(), last_counting_iter)),
                         [=] __host__ __device__(thrust::tuple<int, int> t)
                         {
                             curr_dest_node_to_unique_index_per_relation_d[thrust::get<0>(t)] = thrust::get<1>(t);
                         });
        std::cout << "relation " << idx_relation << " dest_node_to_unique_index_per_relation_d[idx_relation] address: " << dest_node_to_unique_index_per_relation_d[idx_relation] << std::endl;
    }
    for (int idx_relation = 0; idx_relation < num_relations; idx_relation++)
    {
        unique_indices_to_column_indices_per_relation_d.push_back(thrust::raw_pointer_cast(coo_matrices_column_indices_unique[idx_relation].data()));
        num_unique_indices_to_column_indices_per_relation[idx_relation] = coo_matrices_column_indices_unique[idx_relation].size();
    }


    // preparing intermediate data: intermediate_node_vect_d
    std::vector<thrust::device_vector<float>> intermediate_node_vect(num_relations);
    thrust::device_vector<float *> intermediate_node_vect_d;
    for (int idx_relation = 0; idx_relation < num_relations; idx_relation++)
    {
        intermediate_node_vect[idx_relation].resize(coo_matrices_column_indices_unique[idx_relation].size() * OUT_DIM);
    }

    for (int idx_relation = 0; idx_relation < num_relations; idx_relation++)
    {
        intermediate_node_vect_d.push_back(thrust::raw_pointer_cast(intermediate_node_vect[idx_relation].data()));
    }



    // weight initialization
    float *node_input_data;
    float *relation_attention_matrices;
    cudaMalloc((void **)&node_input_data, sizeof(float) * num_nodes * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD);
    cudaMalloc((void **)&relation_attention_matrices, sizeof(float) * num_relations * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD * NODE_INPUT_DIM_PER_HEAD);

    if (FlagInitWithRandomValue)
    {
        curandGenerator_t m_prng;
        // Create a new generator
        curandCreateGenerator(&m_prng, CURAND_RNG_PSEUDO_DEFAULT);
        // Set the generator options
        curandSetPseudoRandomGeneratorSeed(m_prng, (unsigned long)0);
        // Generate random numbers
        // curandGenerateUniform(m_prng, mus, num_relations);
        curandGenerateUniform(m_prng, node_input_data, num_nodes * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD);
        curandGenerateUniform(m_prng, relation_attention_matrices, num_relations * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD * NODE_INPUT_DIM_PER_HEAD);
    }
    else
    {
        cudaMemset(node_input_data, 64, sizeof(float) * num_nodes * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD);
        cudaMemset(relation_attention_matrices, 64, sizeof(float) * num_relations * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD * NODE_INPUT_DIM_PER_HEAD);
    }


    // preparing op kernel launch specific preprocessed metadata: num_blocks_xdim_for_same_relation_per_block_vect, beg_node_entry_idxes_vect, blockid_relation_id_vect 
    thrust::device_vector<int> num_blocks_xdim_for_same_relation_per_block_vect;
    thrust::device_vector<int> blockid_relation_id_vect;
    thrust::device_vector<int> beg_node_entry_idxes_vect;
    std::vector<int> num_blocks_xdim_for_same_relation_vect;
    std::vector<int> num_blocks_xdim_for_all_prev_relation_vect;
    num_blocks_xdim_for_all_prev_relation_vect.push_back(0);

    // for ease of programming equally partition the workload to different blocks at this moment.
    for (int idx_relationship = 0; idx_relationship < num_relations; idx_relationship++)
    {
        int num_blocks_xdim_for_this_and_prev_relation = (idx_relationship + 1 + 0.0) / (num_relations + 0.0) * RTX_3090_GRIDSIZE;
        num_blocks_xdim_for_all_prev_relation_vect.push_back(num_blocks_xdim_for_this_and_prev_relation);
    }
    for (int idx_relationship = 0; idx_relationship < num_relations; idx_relationship++)
    {
        num_blocks_xdim_for_same_relation_vect.push_back(num_blocks_xdim_for_all_prev_relation_vect[idx_relationship + 1] - num_blocks_xdim_for_all_prev_relation_vect[idx_relationship]);
    }
    num_blocks_xdim_for_all_prev_relation_vect.erase(num_blocks_xdim_for_all_prev_relation_vect.begin());
    int idx_curr_relation = 0;
    int curr_beg_node_entry_idx = 0;

    // grid and thread configuration of the first stage
    //   block (0,0): (head0 (64 element), 16 nodes), (head1 (64 element), 16 nodes); block(1,0): (head0 (64 element), 16 nodes), (head1 (64 element), 16 nodes); ... block(BLOCKDIM_X-1,0): (head0 (64 element), 16 nodes), (head1 (64 element), 16 nodes);
    //   block (0,1): (head2 (64 element), 16 nodes), (head3 (64 element), 16 nodes); block(1,1): (head2 (64 element), 16 nodes), (head3 (64 element), 16 nodes); ... block(BLOCKDIM_X-1,1): (head2 (64 element), 16 nodes), (head3 (64 element), 16 nodes);

    for (int idx_block = 0; idx_block < RTX_3090_GRIDSIZE; idx_block++)
    {
        if (idx_curr_relation < num_blocks_xdim_for_all_prev_relation_vect.size() - 1 && idx_block >= num_blocks_xdim_for_all_prev_relation_vect[idx_curr_relation])
        {
            assert(curr_beg_node_entry_idx / COARSE_SGEMM_NODES_PER_BLOCK == num_blocks_xdim_for_same_relation_vect[idx_curr_relation]);
            idx_curr_relation++;
            curr_beg_node_entry_idx = 0;
        }
        blockid_relation_id_vect.push_back(idx_curr_relation);
        beg_node_entry_idxes_vect.push_back(curr_beg_node_entry_idx);
        curr_beg_node_entry_idx += COARSE_SGEMM_NODES_PER_BLOCK;
        num_blocks_xdim_for_same_relation_per_block_vect.push_back(num_blocks_xdim_for_same_relation_vect[idx_curr_relation]);
    }

    

    dim3 block(COARSE_SGEMM_BLOCKSIZE, 1, 1);
    dim3 grid(RTX_3090_GRIDSIZE, 2, 1);
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    EdgeAttentionConcatenatedFirstStageWeightMulDestCOOKernel<TILE_SZ_A, TILE_SZ_B, OUT_DIM, NUM_HEADS><<<grid, block>>>(thrust::raw_pointer_cast(intermediate_node_vect_d.data()), concatenated_coo_matrix_column_indices.size(), thrust::raw_pointer_cast(concatenated_coo_matrix_column_indices.data()), thrust::raw_pointer_cast(concatenated_coo_matrix_values.data()),
                                                                                     node_input_data, relation_attention_matrices, thrust::raw_pointer_cast(dest_node_to_unique_index_per_relation_d.data()), thrust::raw_pointer_cast(unique_indices_to_column_indices_per_relation_d.data()), thrust::raw_pointer_cast(num_unique_indices_to_column_indices_per_relation.data()), num_relations, thrust::raw_pointer_cast(num_blocks_xdim_for_same_relation_per_block_vect.data()), thrust::raw_pointer_cast(beg_node_entry_idxes_vect.data()), thrust::raw_pointer_cast(blockid_relation_id_vect.data()));
    

    

    thrust::copy(intermediate_node_vect[0].begin(), intermediate_node_vect[0].begin() + 1, std::ostream_iterator<float>(std::cout, ","));
    cuda_err_chk(cudaPeekAtLastError());
    cuda_err_chk(cudaDeviceSynchronize());
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "GPU doGPUEdgeAttentionConcatenatedCOO_128_8 Kernel time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " us" << std::endl;
    cudaFree(node_input_data);
    cudaFree(relation_attention_matrices);
    return outEdges_vect;
}