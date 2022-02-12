#pragma once
#include "hetero_edgesoftmax.h"

// __device__ __forceinline__ void _perRow_EdgeAttentionConcatenatedCOOKernel(int edge_idx, float **__restrict__ outEdges_per_relation, int nnz, int *__restrict__ matCols, int *__restrict__ relation,
//                                                                                    float *__restrict__ node_input_data, float *__restrict__ relation_attention_matrices)
// {
//     //@@ insert spmv kernel for coo format

//     int col = matCols[edge_idx];
//     float val = expf(edge_input_data[edge_idx]) + 1e-10f;
//     atomicAdd(&outEdges_per_relation[relation[edge_idx]][col], val);
// }

// __device__ __forceinline__ void _EdgeAttentionConcatenatedCOOKernel(int beg_edge_idx, int stride, float4 **__restrict__ outEdges_per_relation, int nnz, int *__restrict__ matCols, int *__restrict__ matRelation,
//                                                                             float *__restrict__ node_input_data, float *__restrict__ relation_attention_matrices)
// {
//     for (int edge_idx = beg_edge_idx; edge_idx < nnz; edge_idx += stride)
//     {
//         _perRow_EdgeAttentionConcatenatedCOOKernel(edge_idx, outEdges_per_relation, nnz, matCols, matRelation, node_input_data, relation_attention_matrices);
//     }
// }

// __global__ void EdgeAttentionConcatenatedCOOKernel(float4 **__restrict__ outEdges_per_relation, int nnz, int *__restrict__ matCols, int *__restrict__ matRelation,
//                                                  float *__restrict__ node_input_data, float * __restrict__ relation_attention_matrices)
// {
//     int beg_edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
//     _EdgeAttentionConcatenatedCOOKernel(beg_edge_idx, blockDim.x * gridDim.x, outEdges_per_relation, nnz, matCols, matRelation, node_input_data, relation_attention_matrices);
// }

// grid and thread configuration of the first stage
//   block (0,0): (head0 (64 element), 16 nodes), (head1 (64 element), 16 nodes); block(1,0): (head0 (64 element), 16 nodes), (head1 (64 element), 16 nodes); ... block(BLOCKDIM_X-1,0): (head0 (64 element), 16 nodes), (head1 (64 element), 16 nodes);
//   block (0,1): (head2 (64 element), 16 nodes), (head3 (64 element), 16 nodes); block(1,1): (head2 (64 element), 16 nodes), (head3 (64 element), 16 nodes); ... block(BLOCKDIM_X-1,1): (head2 (64 element), 16 nodes), (head3 (64 element), 16 nodes);

const int OUT_DIM = 256;
const int NUM_HEADS = 4;
const int NODE_INPUT_DIM_PER_HEAD = OUT_DIM / NUM_HEADS; // d_k

#define TILE_SZ_A 128
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A / TILE_SZ_B)
#define TILE_NUM_HEAD (TILE_SZ_A / NODE_INPUT_DIM_PER_HEAD)

#define COARSE_SGEMM_BLOCKSIZE (TILE_SZ_A)
#define COARSE_SGEMM_NODES_PER_BLOCK (TILE_SZ_B)
static_assert(TILE_SZ_RATIO % TILE_NUM_HEAD == 0, "");

__device__ __forceinline__ void mysgemm(int m, int n, int k, float *A, float *B, float *C, int *dest_node_index_unique, int BcolBias)
{

/********************************************************************
 *
 * Compute C = A x B
 *   where A is a (m x k) matrix
 *   where B is a (k x n) matrix
 *   where C is a (m x n) matrix
 *
 * Use register and shared memory tiling and thread coarsening
 *
 * NOTE: A and C are column major, B is column major as well
 * m == 256, n == number of unique dest nodes in this relation, k == 64
 * m stands for OUT_DIM, k stands for NODE_INPUT_DIM_PER_HEAD
 ********************************************************************/
// layout of B (column major)
// |- node 0 head 0, node 1 head 0, node 2 head 0, ... -|
// |  node 0 head 0, node 1 head 0, node 2 head 0, ...  |
// |  node 0 head 1, node 1 head 1, node 2 head 1, ...  |
// |  node 0 head 1, node 1 head 1, node 2 head 1, ...  |
// |  node 0 head 2, node 1 head 2, node 2 head 2, ...  |
// |  node 0 head 2, node 1 head 2, node 2 head 2, ...  |
// |  node 0 head 3, node 1 head 3, node 2 head 3, ...  |
// |- node 0 head 3, node 1 head 3, node 2 head 3, ... -|

// layout of A (column major)
// |- src head 0 dest head 0, src head 0 dest head 0, src head 0 dest head 1, ... -|
// |  src head 0 dest head 0, src head 0 dest head 0, src head 0 dest head 1, ...  |
// |  src head 1 dest head 0, src head 1 dest head 0, src head 1 dest head 1, ...  |
// |  src head 1 dest head 0, src head 1 dest head 0, src head 1 dest head 1, ...  |
// |  src head 2 dest head 0, src head 2 dest head 0, src head 2 dest head 1, ...  |
// |  src head 2 dest head 0, src head 2 dest head 0, src head 2 dest head 1, ...  |
// |  src head 3 dest head 0, src head 3 dest head 0, src head 3 dest head 1, ...  |
// |- src head 3 dest head 0, src head 3 dest head 0, src head 3 dest head 1, ... -|

// layout of C (column major)
// |- intermediate node 0 head 0, intermediate node 1 head 0, intermediate node 2 head 0, ... -|
// |  intermediate node 0 head 0, intermediate node 1 head 0, intermediate node 2 head 0, ...  |
// |  intermediate node 0 head 1, intermediate node 1 head 1, intermediate node 2 head 1, ...  |
// |  intermediate node 0 head 1, intermediate node 1 head 1, intermediate node 2 head 1, ...  |
// |  intermediate node 0 head 2, intermediate node 1 head 2, intermediate node 2 head 2, ...  |
// |  intermediate node 0 head 2, intermediate node 1 head 2, intermediate node 2 head 2, ...  |
// |  intermediate node 0 head 3, intermediate node 1 head 3, intermediate node 2 head 3, ...  |
// |- intermediate node 0 head 3, intermediate node 1 head 3, intermediate node 2 head 3, ... -|

// Macros for accessing flattened matrices
#define A(idx_head, row, col) A[(idx_head * k) + (row) + (col)*m]
#define B(idx_head, row, col) B[(idx_head * k) + (row) + (dest_node_index_unique[col]) * m]
#define C(idx_head, row, col) C[(idx_head * k) + (row) + (col)*m]
    __shared__ float shmem[TILE_NUM_HEAD][TILE_SZ_RATIO / TILE_NUM_HEAD][TILE_SZ_B];

    // INSERT KERNEL CODE HERE

    int ArowIdx = blockIdx.y * TILE_SZ_A + threadIdx.x;

    for (int i = 0; i < (k + TILE_SZ_RATIO / TILE_NUM_HEAD - 1) / (TILE_SZ_RATIO / TILE_NUM_HEAD); i++)
    {
        // load A in registers
        float reg0 = 0.0f;
        float reg1 = 0.0f;
        float reg2 = 0.0f;
        float reg3 = 0.0f;
        /*float reg4=0.0f;
        float reg5=0.0f;
        float reg6=0.0f;
        float reg7=0.0f;*/
        if (ArowIdx < m)
        {
            reg0 = (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD) ? A(ArowIdx / k, ArowIdx % k, i * TILE_SZ_RATIO / TILE_NUM_HEAD) : 0.0f;
            reg1 = (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD + 1) ? A(ArowIdx / k, ArowIdx % k, i * TILE_SZ_RATIO / TILE_NUM_HEAD + 1) : 0.0f;
            reg2 = (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD + 2) ? A(ArowIdx / k, ArowIdx % k, i * TILE_SZ_RATIO / TILE_NUM_HEAD + 2) : 0.0f;
            reg3 = (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD + 3) ? A(ArowIdx / k, ArowIdx % k, i * TILE_SZ_RATIO / TILE_NUM_HEAD + 3) : 0.0f;
            /*reg4 = (k>i*TILE_SZ_RATIO/TILE_NUM_HEAD+4)?A(blockIdx.y*TILE_NUM_HEAD+ (TILE_NUM_HEAD-1 - threadIdx.x / k), ArowIdx % k,i*TILE_SZ_RATIO/TILE_NUM_HEAD):0.0f;
            reg5 = (k>i*TILE_SZ_RATIO/TILE_NUM_HEAD+5)?A(blockIdx.y * TILE_NUM_HEAD + (TILE_NUM_HEAD - 1 - threadIdx.x / k), ArowIdx % k,i*TILE_SZ_RATIO/TILE_NUM_HEAD):0.0f;
            reg6 = (k>i*TILE_SZ_RATIO/TILE_NUM_HEAD+6)?A(blockIdx.y * TILE_NUM_HEAD + (TILE_NUM_HEAD - 1 - threadIdx.x / k), ArowIdx % k,i*TILE_SZ_RATIO/TILE_NUM_HEAD):0.0f;
            reg7 = (k>i*TILE_SZ_RATIO/TILE_NUM_HEAD+7)?A(blockIdx.y * TILE_NUM_HEAD + (TILE_NUM_HEAD - 1 - threadIdx.x / k), ArowIdx % k,i*TILE_SZ_RATIO/TILE_NUM_HEAD):0.0f;*/
        }
        // load B in shared memory
        // the loading scheme is adjusted to fit B's column-major layout
        int shdmemLDBrowIdx = i * TILE_SZ_RATIO / TILE_NUM_HEAD + (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (TILE_SZ_RATIO / TILE_NUM_HEAD);
        int shdmemLDBcolIdx = /*blockIdx.x * TILE_SZ_B*/ BcolBias + (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (TILE_SZ_RATIO / TILE_NUM_HEAD);
        int shdmemLDBheadIdx = blockIdx.y * TILE_NUM_HEAD + threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD);
        shmem[threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (TILE_SZ_RATIO / TILE_NUM_HEAD)][(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (TILE_SZ_RATIO / TILE_NUM_HEAD)] = (shdmemLDBrowIdx < k && shdmemLDBcolIdx < n) ? B(shdmemLDBheadIdx, shdmemLDBrowIdx, shdmemLDBcolIdx) : 0.0f;

        __syncthreads();
        // compute C
        if (ArowIdx < m)
        {
            for (int shdmemColIdx = 0; shdmemColIdx < TILE_SZ_B; shdmemColIdx++)
            {
                int CcolIdx = shdmemColIdx + /*blockIdx.x * TILE_SZ_B*/ BcolBias;
                if (CcolIdx < n)
                {
                    C(ArowIdx / k, ArowIdx % k, CcolIdx) += reg0 * shmem[threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][0][shdmemColIdx];
                    C(ArowIdx / k, ArowIdx % k, CcolIdx) += reg1 * shmem[threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][1][shdmemColIdx];
                    C(ArowIdx / k, ArowIdx % k, CcolIdx) += reg2 * shmem[threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][2][shdmemColIdx];
                    C(ArowIdx / k, ArowIdx % k, CcolIdx) += reg3 * shmem[threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][3][shdmemColIdx];
                    /*C(ArowIdx / k, ArowIdx % k, CcolIdx)+=reg4*shmem[1-threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][0][shdmemColIdx];
                    C(ArowIdx / k, ArowIdx % k, CcolIdx)+=reg5*shmem[1-threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][1][shdmemColIdx];
                    C(ArowIdx / k, ArowIdx % k, CcolIdx)+=reg6*shmem[1-threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][2][shdmemColIdx];
                    C(ArowIdx / k, ArowIdx % k, CcolIdx)+=reg7*shmem[1-threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][3][shdmemColIdx];*/
                }
            }
        }
        __syncthreads();
    }

    // SSL Hint (9/6/21): try using just one register for the tile of A
    // rather than several--in other words, load one value (per thread)
    // from A and compute using that value rather than loading all values
    // before doing the computation.  This approach seems to be slightly
    // faster than the alternative.
}

__global__ void EdgeAttentionConcatenatedFirstStageWeightMulDestCOOKernel(float **__restrict__ intermediate_node_vect, int nnz, int *__restrict__ matCols, int *__restrict__ matRelation,
                                                                          float *__restrict__ node_input_data, float *__restrict__ relation_attention_matrices, int **__restrict__ dest_node_to_unique_index_per_relation, int **__restrict__ unique_index_to_dest_node_per_relation, int *__restrict__ sizes_unique_index_to_dest_node_per_relation, int num_relations, int *__restrict__ num_blocks_xdim_for_same_relation_per_block_vect, int *__restrict__ beg_node_entry_idxes_vect, int *__restrict__ blockid_relation_id_vect)
{

    int beg_node_entry_idx = beg_node_entry_idxes_vect[blockIdx.x];
    int stride = num_blocks_xdim_for_same_relation_per_block_vect[blockIdx.x] * COARSE_SGEMM_NODES_PER_BLOCK;
    int relation_idx = blockid_relation_id_vect[blockIdx.x];

    for (int node_entry_idx = beg_node_entry_idx; node_entry_idx < sizes_unique_index_to_dest_node_per_relation[relation_idx]; node_entry_idx += stride)
    {
        mysgemm(OUT_DIM, sizes_unique_index_to_dest_node_per_relation[relation_idx], NODE_INPUT_DIM_PER_HEAD, &relation_attention_matrices[relation_idx * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD * NODE_INPUT_DIM_PER_HEAD], node_input_data, intermediate_node_vect[relation_idx], unique_index_to_dest_node_per_relation[relation_idx], node_entry_idx);
    }
}

__global__ void EdgeAttentionConcatenatedSecondStageSrcInnerProductDestIntemediateCOOKernel(float4 *__restrict__ outEdges, int nnz, int *__restrict__ matCols, int *__restrict__ matRows, int *__restrict__ matRelation,
                                                                                            float *__restrict__ node_input_data, float **__restrict__ intermediate_node_vect_per_relation, int **__restrict__ dest_node_to_unique_index_per_relation)
{
    // each warp is in charge of an edge
    int beg_edge_idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_idx = (blockIdx.x * blockDim.x + threadIdx.x) % WARP_SIZE;
    for (int edge_idx = beg_edge_idx; edge_idx < nnz; edge_idx += (blockDim.x * gridDim.x) / WARP_SIZE)
    {
#define FULL_MASK 0xffffffff

        int col = matCols[edge_idx];
        int col_relation_idx = dest_node_to_unique_index_per_relation[matRelation[edge_idx]][col];
        int row = matRows[edge_idx];
        float src_1 = node_input_data[row * 256 + lane_idx];
        float src_2 = node_input_data[row * 256 + 32 + lane_idx];
        float dest_1 = intermediate_node_vect_per_relation[matRelation[edge_idx]][col_relation_idx * 256 + lane_idx];
        float dest_2 = intermediate_node_vect_per_relation[matRelation[edge_idx]][col_relation_idx * 256 + 32 + lane_idx];
        float product_1 = src_1 * dest_1 + src_2 * dest_2;
        for (int offset = 16; offset > 0; offset /= 2)
            product_1 += __shfl_down_sync(FULL_MASK, product_1, offset);

        float src_3 = node_input_data[row * 256 + 64 + lane_idx];
        float src_4 = node_input_data[row * 256 + 96 + lane_idx];
        float dest_3 = intermediate_node_vect_per_relation[matRelation[edge_idx]][col_relation_idx * 256 + 64 + lane_idx];
        float dest_4 = intermediate_node_vect_per_relation[matRelation[edge_idx]][col_relation_idx * 256 + 96 + lane_idx];
        float product_2 = src_3 * dest_3 + src_4 * dest_4;
        for (int offset = 16; offset > 0; offset /= 2)
            product_2 += __shfl_down_sync(FULL_MASK, product_2, offset);

        float src_5 = node_input_data[row * 256 + 128 + lane_idx];
        float src_6 = node_input_data[row * 256 + 160 + lane_idx];
        float dest_5 = intermediate_node_vect_per_relation[matRelation[edge_idx]][col_relation_idx * 256 + 128 + lane_idx];
        float dest_6 = intermediate_node_vect_per_relation[matRelation[edge_idx]][col_relation_idx * 256 + 160 + lane_idx];
        float product_3 = src_5 * dest_5 + src_6 * dest_6;
        for (int offset = 16; offset > 0; offset /= 2)
            product_3 += __shfl_down_sync(FULL_MASK, product_3, offset);

        float src_7 = node_input_data[row * 256 + 192 + lane_idx];
        float src_8 = node_input_data[row * 256 + 224 + lane_idx];
        float dest_7 = intermediate_node_vect_per_relation[matRelation[edge_idx]][col_relation_idx * 256 + 192 + lane_idx];
        float dest_8 = intermediate_node_vect_per_relation[matRelation[edge_idx]][col_relation_idx * 256 + 224 + lane_idx];
        float product_4 = src_7 * dest_7 + src_8 * dest_8;
        for (int offset = 16; offset > 0; offset /= 2)
            product_4 += __shfl_down_sync(FULL_MASK, product_4, offset);
        if (lane_idx == 0)
        {
            outEdges[edge_idx] = make_float4(product_1, product_2, product_3, product_4);
        }
    }
}

thrust::device_vector<float4> _doGPUEdgeAttentionConcatenatedCOOKernel(int num_nodes, cusp::coo_matrix<int, int, cusp::device_memory>::row_indices_array_type concatenated_coo_matrix_row_indices, cusp::coo_matrix<int, int, cusp::device_memory>::column_indices_array_type concatenated_coo_matrix_column_indices, std::vector<cusp::coo_matrix<int, int, cusp::device_memory>::column_indices_array_type> coo_matrices_column_indices, cusp::coo_matrix<int, int, cusp::device_memory>::values_array_type concatenated_coo_matrix_values, int num_relations, bool FlagInitWithRandomValue)
{

    std::vector<thrust::device_vector<float>> intermediate_node_vect(num_relations);
    thrust::device_vector<float *> intermediate_node_vect_d;
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

    /*for (int idx_relation = 0; idx_relation < num_relations; idx_relation++) {
        std::cout << "relation " << idx_relation << " coo_matrices_column_indices_unique" <<std::endl;
        thrust::copy(coo_matrices_column_indices_unique[idx_relation].begin(), coo_matrices_column_indices_unique[idx_relation].end(), std::ostream_iterator<int>(std::cout, ","));
    }*/

    for (int idx_relation = 0; idx_relation < num_relations; idx_relation++)
    {
        thrust::counting_iterator<int> first_counting_iter(0);
        thrust::counting_iterator<int> last_counting_iter = first_counting_iter + coo_matrices_column_indices_unique[idx_relation].size();

        int *curr_dest_node_to_unique_index_per_relation_d = dest_node_to_unique_index_per_relation_d[idx_relation];
        /*thrust::for_each(thrust::device, coo_matrices_column_indices_unique[idx_relation].begin(),
            coo_matrices_column_indices_unique[idx_relation].end(),
            [=]__host__ __device__(int t) {
            curr_dest_node_to_unique_index_per_relation_d[t] = 1;
        });*/
        thrust::for_each(thrust::device, thrust::make_zip_iterator(thrust::make_tuple(coo_matrices_column_indices_unique[idx_relation].begin(), first_counting_iter)),
                         thrust::make_zip_iterator(thrust::make_tuple(coo_matrices_column_indices_unique[idx_relation].end(), last_counting_iter)),
                         [=] __host__ __device__(thrust::tuple<int, int> t)
                         {
                             curr_dest_node_to_unique_index_per_relation_d[thrust::get<0>(t)] = thrust::get<1>(t);
                         });
        std::cout << "relation " << idx_relation << " dest_node_to_unique_index_per_relation_d[idx_relation] address: " << dest_node_to_unique_index_per_relation_d[idx_relation] << std::endl;
        /*thrust::copy(dest_node_to_unique_index_per_relation[idx_relation].begin(), dest_node_to_unique_index_per_relation[idx_relation].end(),std::ostream_iterator<int>(std::cout, ","));*/
    }

    for (int idx_relation = 0; idx_relation < num_relations; idx_relation++)
    {
        intermediate_node_vect[idx_relation].resize(coo_matrices_column_indices_unique[idx_relation].size() * OUT_DIM);
    }

    for (int idx_relation = 0; idx_relation < num_relations; idx_relation++)
    {
        intermediate_node_vect_d.push_back(thrust::raw_pointer_cast(intermediate_node_vect[idx_relation].data()));
    }

    for (int idx_relation = 0; idx_relation < num_relations; idx_relation++)
    {
        unique_indices_to_column_indices_per_relation_d.push_back(thrust::raw_pointer_cast(coo_matrices_column_indices_unique[idx_relation].data()));
        num_unique_indices_to_column_indices_per_relation[idx_relation] = coo_matrices_column_indices_unique[idx_relation].size();
    }

    // float *outEdges;
    // float *mus;

    float *node_input_data;
    float *relation_attention_matrices;
    cudaMalloc((void **)&node_input_data, sizeof(float) * num_nodes * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD);
    cudaMalloc((void **)&relation_attention_matrices, sizeof(float) * num_relations * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD * NODE_INPUT_DIM_PER_HEAD);
    // cudaMalloc((void **)&mus, sizeof(float) * num_relations);
    // cudaMalloc((void **)&outEdges, sizeof(float) * concatenated_coo_matrix_column_indices.size());

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
        // cudaMemset(mus, 1, sizeof(float) * num_relations);
        cudaMemset(node_input_data, 64, sizeof(float) * num_nodes * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD);
        cudaMemset(relation_attention_matrices, 64, sizeof(float) * num_relations * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD * NODE_INPUT_DIM_PER_HEAD);
    }

    thrust::device_vector<int> num_blocks_xdim_for_same_relation_per_block_vect;
    thrust::device_vector<int> blockid_relation_id_vect;
    thrust::device_vector<int> beg_node_entry_idxes_vect;
    std::vector<int> num_blocks_xdim_for_same_relation_vect;
    std::vector<int> num_blocks_xdim_for_all_prev_relation_vect;
    num_blocks_xdim_for_all_prev_relation_vect.push_back(0);
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
    // EdgeAttentionConcatenatedCOOKernel<<<grid, block>>>( thrust::raw_pointer_cast(outEdges_per_relation_vect.data()), concatenated_coo_matrix_column_indices.size(), thrust::raw_pointer_cast(concatenated_coo_matrix_column_indices.data()), thrust::raw_pointer_cast(concatenated_coo_matrix_values.data()), node_input_data);
    EdgeAttentionConcatenatedFirstStageWeightMulDestCOOKernel<<<grid, block>>>(thrust::raw_pointer_cast(intermediate_node_vect_d.data()), concatenated_coo_matrix_column_indices.size(), thrust::raw_pointer_cast(concatenated_coo_matrix_column_indices.data()), thrust::raw_pointer_cast(concatenated_coo_matrix_values.data()),
                                                                               node_input_data, relation_attention_matrices, thrust::raw_pointer_cast(dest_node_to_unique_index_per_relation_d.data()), thrust::raw_pointer_cast(unique_indices_to_column_indices_per_relation_d.data()), thrust::raw_pointer_cast(num_unique_indices_to_column_indices_per_relation.data()), num_relations, thrust::raw_pointer_cast(num_blocks_xdim_for_same_relation_per_block_vect.data()), thrust::raw_pointer_cast(beg_node_entry_idxes_vect.data()), thrust::raw_pointer_cast(blockid_relation_id_vect.data()));
    dim3 block2(RTX_3090_BLOCKSIZE, 1, 1);
    dim3 grid2(RTX_3090_GRIDSIZE, 1, 1);
    EdgeAttentionConcatenatedSecondStageSrcInnerProductDestIntemediateCOOKernel<<<grid2, block2>>>(thrust::raw_pointer_cast(outEdges_vect.data()), concatenated_coo_matrix_column_indices.size(), thrust::raw_pointer_cast(concatenated_coo_matrix_column_indices.data()), thrust::raw_pointer_cast(concatenated_coo_matrix_row_indices.data()), thrust::raw_pointer_cast(concatenated_coo_matrix_values.data()),
                                                                                                   node_input_data, thrust::raw_pointer_cast(intermediate_node_vect_d.data()), thrust::raw_pointer_cast(dest_node_to_unique_index_per_relation_d.data()));

    for (int idx_relation = 0; idx_relation < num_relations; idx_relation++)
    {
        std::cout << "intermediate_node_vect[" << idx_relation << "]" << std::endl;
        thrust::copy(intermediate_node_vect[idx_relation].begin(), intermediate_node_vect[idx_relation].end(), std::ostream_iterator<float>(std::cout, ","));
    }
    cuda_err_chk(cudaPeekAtLastError());
    cuda_err_chk(cudaDeviceSynchronize());
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "GPU doGPUEdgeAttentionConcatenatedCOOKernel time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " us" << std::endl;
    // cudaFree(outEdges);
    cudaFree(node_input_data);
    cudaFree(relation_attention_matrices);
    return outEdges_vect;
}

thrust::device_vector<float4> doGPUEdgeAttentionConcatenatedCOOKernel(std::vector<cusp::coo_matrix<int, int, cusp::device_memory>> coo_matrices, cusp::coo_matrix<int, int, cusp::device_memory> concatenated_coo_matrix, int num_relations, bool FlagInitWithRandomValue)
{
    std::vector<cusp::coo_matrix<int, int, cusp::device_memory>::column_indices_array_type> coo_matrices_column_indices;
    for (int idx_relation = 0; idx_relation < coo_matrices.size(); idx_relation++)
    {
        coo_matrices_column_indices.push_back(coo_matrices[idx_relation].column_indices);
    }
    return _doGPUEdgeAttentionConcatenatedCOOKernel(concatenated_coo_matrix.num_rows, concatenated_coo_matrix.row_indices, concatenated_coo_matrix.column_indices, coo_matrices_column_indices, concatenated_coo_matrix.values, num_relations, FlagInitWithRandomValue);
}
