#pragma once
#include "hetero_edgesoftmax.h"
#include "EdgeAttentionCOO_128_16.h"

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

#define OUT_DIM (256)
#define NUM_HEADS (4)
// d_k
#define NODE_INPUT_DIM_PER_HEAD (OUT_DIM / NUM_HEADS)

#define TILE_SZ_A 512
#define TILE_SZ_B 32
#define TILE_SZ_RATIO (TILE_SZ_A / TILE_SZ_B)
#define TILE_NUM_HEAD 4
#define K 64
//#define TILE_NUM_HEAD 1

#define COARSE_SGEMM_BLOCKSIZE (TILE_SZ_A)
#define COARSE_SGEMM_NODES_PER_BLOCK (TILE_SZ_B)
static_assert(TILE_SZ_RATIO % TILE_NUM_HEAD == 0, "");

__device__ __forceinline__ void mysgemm_512_32_asyncmemcpy(int m, int n, int k, float *A, float *B, float *C, int *dest_node_index_unique, int BcolBias)
{
    assert(k == 64);
    assert(m == OUT_DIM);
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
#define A(idx_head, row, col) A[(idx_head * K) + (row) + (col)*OUT_DIM]
#define B(idx_head, row, col) B[(idx_head * K) + (row) + (dest_node_index_unique[col]) * OUT_DIM]
#define C(idx_head, row, col) C[(idx_head * K) + (row) + (col)*OUT_DIM]

    __shared__ float shmem[2 /*double buffering*/][TILE_NUM_HEAD][TILE_SZ_B][8];
    __shared__ float shmem_output[16 /*node idx*/][16 /*element idx in 4 heads*/][2 /*node idx 2nd part*/][16 /*element idx in 4 heads 2nd part*/];
    for (int idx = 0; idx < 16; idx++)
    {
        shmem_output[idx][threadIdx.x / 32][threadIdx.x % 32 / 16][threadIdx.x % 16] = 0.0f;
    }
    static_assert(TILE_SZ_RATIO / TILE_NUM_HEAD == 4, "");
    static_assert(TILE_SZ_RATIO % TILE_NUM_HEAD == 0, "");
    // each thread should load 8/(TILE_SZ_RATIO / TILE_NUM_HEAD) times per iteration

    // INSERT KERNEL CODE HERE

    // int ArowIdx = blockIdx.y * TILE_SZ_A + threadIdx.x;
    int ArowIdx = threadIdx.x / 32 * 16 + ((threadIdx.x % 32) < 16 ? ((threadIdx.x % 32)) : ((threadIdx.x % 32) - 16));
    int shdmemColIdxBias = (threadIdx.x % 32) < 16 ? 0 : 16;

    int shdmemLDBrowIdx = 0 /*i*/ * 8 + (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8);
    int shdmemLDBcolIdx = /*blockIdx.x * TILE_SZ_B*/ BcolBias + (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8);
    int shdmemLDBheadIdx = blockIdx.y * TILE_NUM_HEAD + threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD);
    shmem[0][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8)][(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8)] = (shdmemLDBrowIdx < K && shdmemLDBcolIdx < n) ? B(shdmemLDBheadIdx, shdmemLDBrowIdx, shdmemLDBcolIdx) : 0.0f;
    shmem[0][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8) + 16][(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8)] = (shdmemLDBrowIdx < K && shdmemLDBcolIdx + 16 < n) ? B(shdmemLDBheadIdx, shdmemLDBrowIdx, shdmemLDBcolIdx + 16) : 0.0f;

    __syncthreads();

    float reg0;
    float reg1;
    float reg2;
    float reg3;
    float reg4;
    float reg5;
    float reg6;
    float reg7;
    __shared__ float shmem_Adata[256 * 8];
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cg::this_thread_block());
    // load A in registers; software pipelining
    cg::memcpy_async(tile32, &shmem_Adata[threadIdx.x / 32 * 32], &(A(0, 0, 0 * 8)) + (threadIdx.x / 32 * 32), sizeof(float) * 256 * 8 * 32 / 512);

    for (int i = 0; i < (K + 8 - 1) / (8); i++)
    {
        // shuffle: only half of the warp load the register
        // if (threadIdx.x%32<16){
        // TODO: async load A data to be used in the next iteration into the shared memory
        cg::wait(tile32);
        reg0 = shmem_Adata[ArowIdx + 0 * 8];
        reg1 = shmem_Adata[ArowIdx + 1 * 8];
        reg2 = shmem_Adata[ArowIdx + 2 * 8];
        reg3 = shmem_Adata[ArowIdx + 3 * 8];
        reg4 = shmem_Adata[ArowIdx + 4 * 8];
        reg5 = shmem_Adata[ArowIdx + 5 * 8];
        reg6 = shmem_Adata[ArowIdx + 6 * 8];
        reg7 = shmem_Adata[ArowIdx + 7 * 8];

        __syncthreads();
        if (i < (k + 8 - 1) / (8) - 1)
        {
            cg::memcpy_async(tile32, &shmem_Adata[threadIdx.x / 32 * 32], &(A(0, 0, (i + 1) * 8)) + (threadIdx.x / 32 * 32), sizeof(float) * 256 * 8 * 32 / 512);
        }
        // reg0 = (ArowIdx < OUT_DIM && K > i * 8) ? A(ArowIdx / K, ArowIdx % K, i * 8) : 0.0f;
        // reg1 = (ArowIdx < OUT_DIM && K > i * 8 + 1) ? A(ArowIdx / K, ArowIdx % K, i * 8 + 1) : 0.0f;
        // reg2 = (ArowIdx < OUT_DIM && K > i * 8 + 2) ? A(ArowIdx / K, ArowIdx % K, i * 8 + 2) : 0.0f;
        // reg3 = (ArowIdx < OUT_DIM && K > i * 8 + 3) ? A(ArowIdx / K, ArowIdx % K, i * 8 + 3) : 0.0f;
        // reg4 = (ArowIdx < OUT_DIM && K > i * 8 + 4) ? A(ArowIdx / K, ArowIdx % K, i * 8 + 4) : 0.0f;
        // reg5 = (ArowIdx < OUT_DIM && K > i * 8 + 5) ? A(ArowIdx / K, ArowIdx % K, i * 8 + 5) : 0.0f;
        // reg6 = (ArowIdx < OUT_DIM && K > i * 8 + 6) ? A(ArowIdx / K, ArowIdx % K, i * 8 + 6) : 0.0f;
        // reg7 = (ArowIdx < OUT_DIM && K > i * 8 + 7) ? A(ArowIdx / K, ArowIdx % K, i * 8 + 7) : 0.0f;

        // // shuffle: the second half of each warp get the register data from the first half through warp shuffling
        // reg0= __shfl_sync(0xffffffff, reg0, ((threadIdx.x%32)<16?((threadIdx.x%32)):((threadIdx.x%32)-16)));
        // reg1= __shfl_sync(0xffffffff, reg1, ((threadIdx.x%32)<16?((threadIdx.x%32)):((threadIdx.x%32)-16)));
        // reg2= __shfl_sync(0xffffffff, reg2, ((threadIdx.x%32)<16?((threadIdx.x%32)):((threadIdx.x%32)-16)));
        // reg3= __shfl_sync(0xffffffff, reg3, ((threadIdx.x%32)<16?((threadIdx.x%32)):((threadIdx.x%32)-16)));
        // reg4= __shfl_sync(0xffffffff, reg4, ((threadIdx.x%32)<16?((threadIdx.x%32)):((threadIdx.x%32)-16)));
        // reg5= __shfl_sync(0xffffffff, reg5, ((threadIdx.x%32)<16?((threadIdx.x%32)):((threadIdx.x%32)-16)));
        // reg6= __shfl_sync(0xffffffff, reg6, ((threadIdx.x%32)<16?((threadIdx.x%32)):((threadIdx.x%32)-16)));
        // reg7= __shfl_sync(0xffffffff, reg7, ((threadIdx.x%32)<16?((threadIdx.x%32)):((threadIdx.x%32)-16)));

        // load B in shared memory
        // the loading scheme is adjusted to fit B's column-major layout
        int shdmemLDBrowIdx = i * 8 + (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8);
        int shdmemLDBcolIdx = /*blockIdx.x * TILE_SZ_B*/ BcolBias + (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8);
        int shdmemLDBheadIdx = blockIdx.y * TILE_NUM_HEAD + threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD);

        float next_iter_shmem_val_0 = (shdmemLDBrowIdx + 8 < K && shdmemLDBcolIdx < n) ? B(shdmemLDBheadIdx, shdmemLDBrowIdx + 8, shdmemLDBcolIdx) : 0.0f;
        float next_iter_shmem_val_2 = (shdmemLDBrowIdx + 8 < K && shdmemLDBcolIdx + 16 < n) ? B(shdmemLDBheadIdx, shdmemLDBrowIdx + 8, shdmemLDBcolIdx + 16) : 0.0f;

        // compute C
        if (ArowIdx < OUT_DIM)
        {
            // float shmem_val0[2];
            // float shmem_val1[2];
            // float shmem_val2[2];
            // float shmem_val3[2];
            // float shmem_val4[2];
            // float shmem_val5[2];
            // float shmem_val6[2];
            // float shmem_val7[2];
            // // software pipelining to load data from shared memory for the next iteration
            // shmem_val0[0] = shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][0 + shdmemColIdxBias][0];
            // shmem_val1[0] = shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][0 + shdmemColIdxBias][1];
            // shmem_val2[0] = shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][0 + shdmemColIdxBias][2];
            // shmem_val3[0] = shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][0 + shdmemColIdxBias][3];
            // shmem_val4[0] = shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][0 + shdmemColIdxBias][4];
            // shmem_val5[0] = shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][0 + shdmemColIdxBias][5];
            // shmem_val6[0] = shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][0 + shdmemColIdxBias][6];
            // shmem_val7[0] = shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][0 + shdmemColIdxBias][7];
            for (int shdmemColIdx = 0; shdmemColIdx < 16; shdmemColIdx++)
            {
                // // software pipelining to load data from shared memory for the next iteration
                // if (shdmemColIdx < 15)
                // {
                //     shmem_val0[(shdmemColIdx + 1) % 2] = shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + 1 + shdmemColIdxBias][0];
                //     shmem_val1[(shdmemColIdx + 1) % 2] = shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + 1 + shdmemColIdxBias][1];
                //     shmem_val2[(shdmemColIdx + 1) % 2] = shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + 1 + shdmemColIdxBias][2];
                //     shmem_val3[(shdmemColIdx + 1) % 2] = shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + 1 + shdmemColIdxBias][3];
                //     shmem_val4[(shdmemColIdx + 1) % 2] = shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + 1 + shdmemColIdxBias][4];
                //     shmem_val5[(shdmemColIdx + 1) % 2] = shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + 1 + shdmemColIdxBias][5];
                //     shmem_val6[(shdmemColIdx + 1) % 2] = shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + 1 + shdmemColIdxBias][6];
                //     shmem_val7[(shdmemColIdx + 1) % 2] = shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + 1 + shdmemColIdxBias][7];
                // }
                int CcolIdx = shdmemColIdx + /*blockIdx.x * TILE_SZ_B*/ BcolBias + shdmemColIdxBias;
                
                    /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output[shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16][ArowIdx % 16] += reg0 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + shdmemColIdxBias][0];
                    /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output[shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16][ArowIdx % 16] += reg1 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + shdmemColIdxBias][1];
                    /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output[shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16][ArowIdx % 16] += reg2 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + shdmemColIdxBias][2];
                    /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output[shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16][ArowIdx % 16] += reg3 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + shdmemColIdxBias][3];
                    /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output[shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16][ArowIdx % 16] += reg4 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + shdmemColIdxBias][4];
                    /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output[shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16][ArowIdx % 16] += reg5 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + shdmemColIdxBias][5];
                    /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output[shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16][ArowIdx % 16] += reg6 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + shdmemColIdxBias][6];
                    /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output[shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16][ArowIdx % 16] += reg7 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + shdmemColIdxBias][7];
                
            }
        }
        shmem[(i + 1) % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8)][(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8)] = next_iter_shmem_val_0;
        shmem[(i + 1) % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8) + 16][(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8)] = next_iter_shmem_val_2;
    }
    __syncthreads();

    for (int store_iter = 0; store_iter < 256 * 32 / TILE_SZ_A; store_iter++)
    {
        int node_idx_1 = store_iter;
        int ele_idx_1 = ArowIdx / 16;
        int ele_idx_2 = ArowIdx % 16;
        int CcolIdx = store_iter + /*blockIdx.x * TILE_SZ_B*/ BcolBias + shdmemColIdxBias;
        if (CcolIdx < n)
        {
            C(ele_idx_1 * 16 / 64, ele_idx_2 + ele_idx_1 * 16 % 64, BcolBias + node_idx_1 + shdmemColIdxBias) = shmem_output[node_idx_1][ele_idx_1][shdmemColIdxBias / 16][ele_idx_2];
            // C(ArowIdx/k, ArowIdx%k, CcolIdx) = shmem_output[shdmemColIdx][ArowIdx/16][shdmemColIdxBias/16][ArowIdx%16];
        }
    }

    // SSL Hint (9/6/21): try using just one register for the tile of A
    // rather than several--in other words, load one value (per thread)
    // from A and compute using that value rather than loading all values
    // before doing the computation.  This approach seems to be slightly
    // faster than the alternative.
#undef A
#undef B
#undef C
}

__launch_bounds__(512, 2)
    __global__ void EdgeAttentionConcatenatedFirstStageWeightMulDestCOOKernel_512_32_asyncmemcpy(float **__restrict__ intermediate_node_vect, int nnz, int *__restrict__ matCols, int *__restrict__ matRelation,
                                                                                                 float *__restrict__ node_input_data, float *__restrict__ relation_attention_matrices, int **__restrict__ dest_node_to_unique_index_per_relation, int **__restrict__ unique_index_to_dest_node_per_relation, int *__restrict__ sizes_unique_index_to_dest_node_per_relation, int num_relations, int *__restrict__ num_blocks_xdim_for_same_relation_per_block_vect, int *__restrict__ beg_node_entry_idxes_vect, int *__restrict__ blockid_relation_id_vect)
{

    int beg_node_entry_idx = beg_node_entry_idxes_vect[blockIdx.x];
    int stride = num_blocks_xdim_for_same_relation_per_block_vect[blockIdx.x] * COARSE_SGEMM_NODES_PER_BLOCK;
    int relation_idx = blockid_relation_id_vect[blockIdx.x];

    for (int node_entry_idx = beg_node_entry_idx; node_entry_idx < sizes_unique_index_to_dest_node_per_relation[relation_idx]; node_entry_idx += stride)
    {
        mysgemm_512_32_asyncmemcpy(OUT_DIM, sizes_unique_index_to_dest_node_per_relation[relation_idx], NODE_INPUT_DIM_PER_HEAD, &relation_attention_matrices[relation_idx * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD * NODE_INPUT_DIM_PER_HEAD], node_input_data, intermediate_node_vect[relation_idx], unique_index_to_dest_node_per_relation[relation_idx], node_entry_idx);
    }
}

__launch_bounds__(512, 2)
    __global__ void EdgeAttentionConcatenatedFirstStageWeightMulDestCOOKernel_NonPersistentBlock_512_32_asyncmemcpy(float **__restrict__ intermediate_node_vect, int nnz, int *__restrict__ matCols, int *__restrict__ matRelation,
                                                                                                                    float *__restrict__ node_input_data, float *__restrict__ relation_attention_matrices, int **__restrict__ dest_node_to_unique_index_per_relation, int **__restrict__ unique_index_to_dest_node_per_relation, int *__restrict__ sizes_unique_index_to_dest_node_per_relation, int num_relations, int *__restrict__ beg_node_entry_idxes_vect, int *__restrict__ blockid_relation_id_vect)
{

    int node_entry_idx = beg_node_entry_idxes_vect[blockIdx.x];
    int relation_idx = blockid_relation_id_vect[blockIdx.x];

    mysgemm_512_32_asyncmemcpy(OUT_DIM, sizes_unique_index_to_dest_node_per_relation[relation_idx], NODE_INPUT_DIM_PER_HEAD, &relation_attention_matrices[relation_idx * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD * NODE_INPUT_DIM_PER_HEAD], node_input_data, intermediate_node_vect[relation_idx], unique_index_to_dest_node_per_relation[relation_idx], node_entry_idx);
}

thrust::device_vector<float4> EdgeAttentionConcatenatedSrcWeightMulDestCOOKernel_512_32_asyncmemcpy(int num_nodes, cusp::coo_matrix<int, int, cusp::device_memory>::row_indices_array_type concatenated_coo_matrix_row_indices, cusp::coo_matrix<int, int, cusp::device_memory>::column_indices_array_type concatenated_coo_matrix_column_indices, std::vector<cusp::coo_matrix<int, int, cusp::device_memory>::column_indices_array_type> coo_matrices_column_indices, cusp::coo_matrix<int, int, cusp::device_memory>::values_array_type concatenated_coo_matrix_values, int num_relations, bool FlagInitWithRandomValue, bool FlagEqualWorkPartitionForBlocks)
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

    std::chrono::high_resolution_clock::time_point t1;
    // for ease of programming equally partition the workload to different blocks at this moment.
    if (FlagEqualWorkPartitionForBlocks)
    {
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
        dim3 grid(RTX_3090_GRIDSIZE, 1, 1);
        t1 = std::chrono::high_resolution_clock::now();
        // EdgeAttentionConcatenatedCOOKernel<<<grid, block>>>( thrust::raw_pointer_cast(outEdges_per_relation_vect.data()), concatenated_coo_matrix_column_indices.size(), thrust::raw_pointer_cast(concatenated_coo_matrix_column_indices.data()), thrust::raw_pointer_cast(concatenated_coo_matrix_values.data()), node_input_data);
        // cudaFuncSetAttribute(EdgeAttentionConcatenatedFirstStageWeightMulDestCOOKernel_512_32_asyncmemcpy, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);

        EdgeAttentionConcatenatedFirstStageWeightMulDestCOOKernel_512_32_asyncmemcpy<<<grid, block>>>(thrust::raw_pointer_cast(intermediate_node_vect_d.data()), concatenated_coo_matrix_column_indices.size(), thrust::raw_pointer_cast(concatenated_coo_matrix_column_indices.data()), thrust::raw_pointer_cast(concatenated_coo_matrix_values.data()),
                                                                                                      node_input_data, relation_attention_matrices, thrust::raw_pointer_cast(dest_node_to_unique_index_per_relation_d.data()), thrust::raw_pointer_cast(unique_indices_to_column_indices_per_relation_d.data()), thrust::raw_pointer_cast(num_unique_indices_to_column_indices_per_relation.data()), num_relations, thrust::raw_pointer_cast(num_blocks_xdim_for_same_relation_per_block_vect.data()), thrust::raw_pointer_cast(beg_node_entry_idxes_vect.data()), thrust::raw_pointer_cast(blockid_relation_id_vect.data()));
    }
    else
    { // if not FlagEqualWorkPartitionForBlocks

        int non_persistent_block_num = 0;

        for (int idx_relationship = 0; idx_relationship < num_relations; idx_relationship++)
        {
            num_blocks_xdim_for_same_relation_vect.push_back((num_unique_indices_to_column_indices_per_relation[idx_relationship] + COARSE_SGEMM_NODES_PER_BLOCK - 1) / COARSE_SGEMM_NODES_PER_BLOCK);
            non_persistent_block_num += num_blocks_xdim_for_same_relation_vect[num_blocks_xdim_for_same_relation_vect.size() - 1];
            num_blocks_xdim_for_all_prev_relation_vect.push_back(non_persistent_block_num);
            std::cout << "(" << idx_relationship << ", " << non_persistent_block_num << "," << num_blocks_xdim_for_same_relation_vect[num_blocks_xdim_for_same_relation_vect.size() - 1] << std::endl;
        }

        int idx_curr_relation = 0;
        int curr_beg_node_entry_idx = 0;

        for (int idx_block = 0; idx_block < non_persistent_block_num; idx_block++)
        {
            if (idx_curr_relation < num_blocks_xdim_for_all_prev_relation_vect.size() - 1 && idx_block >= num_blocks_xdim_for_all_prev_relation_vect[idx_curr_relation])
            {
                std::cout << "[" << curr_beg_node_entry_idx << "," << num_blocks_xdim_for_same_relation_vect[idx_curr_relation] << "]" << std::endl;
                assert(curr_beg_node_entry_idx / COARSE_SGEMM_NODES_PER_BLOCK == num_blocks_xdim_for_same_relation_vect[idx_curr_relation]);
                idx_curr_relation++;
                curr_beg_node_entry_idx = 0;
            }
            blockid_relation_id_vect.push_back(idx_curr_relation);
            beg_node_entry_idxes_vect.push_back(curr_beg_node_entry_idx);
            curr_beg_node_entry_idx += COARSE_SGEMM_NODES_PER_BLOCK;
            t1 = std::chrono::high_resolution_clock::now();
            num_blocks_xdim_for_same_relation_per_block_vect.push_back(num_blocks_xdim_for_same_relation_vect[idx_curr_relation]);
        }

        dim3 block(COARSE_SGEMM_BLOCKSIZE, 1, 1);
        dim3 grid(non_persistent_block_num, 1, 1);
        // cudaFuncSetAttribute(EdgeAttentionConcatenatedFirstStageWeightMulDestCOOKernel_NonPersistentBlock_512_32_asyncmemcpy, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
        EdgeAttentionConcatenatedFirstStageWeightMulDestCOOKernel_NonPersistentBlock_512_32_asyncmemcpy<<<grid, block>>>(thrust::raw_pointer_cast(intermediate_node_vect_d.data()), concatenated_coo_matrix_column_indices.size(), thrust::raw_pointer_cast(concatenated_coo_matrix_column_indices.data()), thrust::raw_pointer_cast(concatenated_coo_matrix_values.data()),
                                                                                                                         node_input_data, relation_attention_matrices, thrust::raw_pointer_cast(dest_node_to_unique_index_per_relation_d.data()), thrust::raw_pointer_cast(unique_indices_to_column_indices_per_relation_d.data()), thrust::raw_pointer_cast(num_unique_indices_to_column_indices_per_relation.data()), num_relations, thrust::raw_pointer_cast(beg_node_entry_idxes_vect.data()), thrust::raw_pointer_cast(blockid_relation_id_vect.data()));
    }

    dim3 block2(RTX_3090_BLOCKSIZE, 1, 1);
    dim3 grid2(RTX_3090_GRIDSIZE, 1, 1);
    EdgeAttentionConcatenatedSecondStageSrcInnerProductDestIntemediateCOOKernel<<<grid2, block2>>>(thrust::raw_pointer_cast(outEdges_vect.data()), concatenated_coo_matrix_column_indices.size(), thrust::raw_pointer_cast(concatenated_coo_matrix_column_indices.data()), thrust::raw_pointer_cast(concatenated_coo_matrix_row_indices.data()), thrust::raw_pointer_cast(concatenated_coo_matrix_values.data()),
                                                                                                   node_input_data, thrust::raw_pointer_cast(intermediate_node_vect_d.data()), thrust::raw_pointer_cast(dest_node_to_unique_index_per_relation_d.data()));

    // for (int idx_relation = 0; idx_relation < num_relations; idx_relation++)
    // {
    //     std::cout << "intermediate_node_vect[" << idx_relation << "]" << std::endl;
    //     thrust::copy(intermediate_node_vect[idx_relation].begin(), intermediate_node_vect[idx_relation].end(), std::ostream_iterator<float>(std::cout, ","));
    // }
    cuda_err_chk(cudaPeekAtLastError());
    cuda_err_chk(cudaDeviceSynchronize());
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "GPU doGPUEdgeAttentionConcatenatedCOO_512_32_asyncmemcpy Kernel time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " us" << std::endl;
    // cudaFree(outEdges);
    cudaFree(node_input_data);
    cudaFree(relation_attention_matrices);
    return outEdges_vect;
}

thrust::device_vector<float4> doGPUEdgeAttentionConcatenatedCOOKernel_512_32_asyncmemcpy(std::vector<cusp::coo_matrix<int, int, cusp::device_memory>> coo_matrices, cusp::coo_matrix<int, int, cusp::device_memory> concatenated_coo_matrix, int num_relations, bool FlagInitWithRandomValue, bool FlagEqualWorkPartitionForBlocks)
{
    std::vector<cusp::coo_matrix<int, int, cusp::device_memory>::column_indices_array_type> coo_matrices_column_indices;
    for (int idx_relation = 0; idx_relation < coo_matrices.size(); idx_relation++)
    {
        coo_matrices_column_indices.push_back(coo_matrices[idx_relation].column_indices);
    }
    return EdgeAttentionConcatenatedSrcWeightMulDestCOOKernel_512_32_asyncmemcpy(concatenated_coo_matrix.num_rows, concatenated_coo_matrix.row_indices, concatenated_coo_matrix.column_indices, coo_matrices_column_indices, concatenated_coo_matrix.values, num_relations, FlagInitWithRandomValue, FlagEqualWorkPartitionForBlocks);
}

#undef A
#undef B
#undef C
#undef K
#undef TILE_SZ_A
#undef TILE_SZ_B
#undef TILE_SZ_RATIO
#undef TILE_NUM_HEAD
#undef COARSE_SGEMM_BLOCKSIZE
#undef COARSE_SGEMM_NODES_PER_BLOCK
#undef OUT_DIM
#undef NUM_HEADS
#undef NODE_INPUT_DIM_PER_HEAD