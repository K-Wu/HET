#pragma once
#include "DGLHackKernel.h"

// In this experiment, the fused kernel 1) do fused intermediate vector and attention calculation for the ELL portion, 2) output the intermediate vector in the fused kernel, and 3) do the intermediate vector and destinatiion vector multiplication in the second kernel for the residue csr
// Separate CSR format is used
// The fused kernel iterates all the relationship type, and do the fused logic based on a map between idxUSrcUniqueIdx per relationship to source node index

// adapted template <int OUT_DIM, int NUM_HEADS> class mysgemm_functor<512, 32, OUT_DIM, NUM_HEADS>.
// the logic is the same, except for renaming dest_node_index_unique and adding size of it.
// Let us assume here a = sWt where s is B here and W is A here.
// dim3 block(COARSE_SGEMM_BLOCKSIZE, 1, 1);
// dim3 grid(RTX_3090_GRIDSIZE, 1, 1);
template <int OUT_DIM, int NUM_HEADS>
__device__ __forceinline__ static void func512_32_mysgemm_exec(int m, int n, int k, float*  attention, float *Adata, float *Bdata, float *Cdata, int *node_indices, int* dense_edges_per_src_node, int* num_dense_edges_per_src_node, int* starting_pos_dense_edges_per_src_node, int* eids, int BcolBias)
    {
    assert(OUT_DIM == 256);
    assert(NUM_HEADS == 4);
        constexpr int TILE_SZ_A = 512;
        constexpr int TILE_SZ_B = 32;
        constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
        constexpr int TILE_NUM_HEAD = 4;
        constexpr int TILE_SZ_RATIO = TILE_SZ_A / TILE_SZ_B;
        constexpr int K = 64;
        static_assert(TILE_SZ_RATIO % TILE_NUM_HEAD == 0, "");
        assert(k == 64);
        assert(m == OUT_DIM);
        assert(m==256); // TODO: make the routine more general
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
// TODO: distinguish IN_DIM and OUT_DIM to generalize the kernel
#define A(idx_head, row, col) Adata[(idx_head * K) + (row) + (col)*OUT_DIM]
#define B(idx_head, row, col) Bdata[(idx_head * K) + (row) + (node_indices[col]) * OUT_DIM]
#define T(idx_head, row, col) Bdata[(idx_head * K) + (row) + (col) * OUT_DIM]
#define C(idx_head, row, col) Cdata[(idx_head * K) + (row) + (col)*OUT_DIM]

        __shared__ float shmem[2 /*double buffering*/][TILE_NUM_HEAD][TILE_SZ_B][8];
        __shared__ float shmem_output[16 /*node idx*/][16 /*element idx in 4 heads*/][2 /*node idx 2nd part*/][16 /*element idx in 4 heads 2nd part*/];
        for (int idx = 0; idx < 16; idx++)
        {
            shmem_output[idx][threadIdx.x / 32][threadIdx.x % 32 / 16][threadIdx.x % 16] = 0.0f;
        }
        static_assert(TILE_SZ_RATIO / TILE_NUM_HEAD == 4, "");
        static_assert(TILE_SZ_RATIO % TILE_NUM_HEAD == 0, "");
        // each thread should load 8/(TILE_SZ_RATIO / TILE_NUM_HEAD) times per iteration

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

        for (int i = 0; i < (K + 8 - 1) / (8); i++)
        {
            reg0 = (ArowIdx < OUT_DIM && K > i * 8) ? A(ArowIdx / K, ArowIdx % K, i * 8) : 0.0f;
            reg1 = (ArowIdx < OUT_DIM && K > i * 8 + 1) ? A(ArowIdx / K, ArowIdx % K, i * 8 + 1) : 0.0f;
            reg2 = (ArowIdx < OUT_DIM && K > i * 8 + 2) ? A(ArowIdx / K, ArowIdx % K, i * 8 + 2) : 0.0f;
            reg3 = (ArowIdx < OUT_DIM && K > i * 8 + 3) ? A(ArowIdx / K, ArowIdx % K, i * 8 + 3) : 0.0f;
            reg4 = (ArowIdx < OUT_DIM && K > i * 8 + 4) ? A(ArowIdx / K, ArowIdx % K, i * 8 + 4) : 0.0f;
            reg5 = (ArowIdx < OUT_DIM && K > i * 8 + 5) ? A(ArowIdx / K, ArowIdx % K, i * 8 + 5) : 0.0f;
            reg6 = (ArowIdx < OUT_DIM && K > i * 8 + 6) ? A(ArowIdx / K, ArowIdx % K, i * 8 + 6) : 0.0f;
            reg7 = (ArowIdx < OUT_DIM && K > i * 8 + 7) ? A(ArowIdx / K, ArowIdx % K, i * 8 + 7) : 0.0f;

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
#pragma unroll 2
                for (int shdmemColIdx = 0; shdmemColIdx < 16; shdmemColIdx++)
                {
                    int CcolIdx = shdmemColIdx + /*blockIdx.x * TILE_SZ_B*/ BcolBias + shdmemColIdxBias;

                    /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output[shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16][ArowIdx % 16] += reg0 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + shdmemColIdxBias][0] /*shmem_val0[shdmemColIdx % 2]*/;
                    /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output[shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16][ArowIdx % 16] += reg1 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + shdmemColIdxBias][1] /*shmem_val1[shdmemColIdx % 2]*/;
                    /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output[shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16][ArowIdx % 16] += reg2 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + shdmemColIdxBias][2] /*shmem_val2[shdmemColIdx % 2]*/;
                    /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output[shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16][ArowIdx % 16] += reg3 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + shdmemColIdxBias][3] /*shmem_val3[shdmemColIdx % 2]*/;
                    /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output[shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16][ArowIdx % 16] += reg4 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + shdmemColIdxBias][4] /*shmem_val4[shdmemColIdx % 2]*/;
                    /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output[shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16][ArowIdx % 16] += reg5 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + shdmemColIdxBias][5] /*shmem_val5[shdmemColIdx % 2]*/;
                    /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output[shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16][ArowIdx % 16] += reg6 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + shdmemColIdxBias][6] /*shmem_val6[shdmemColIdx % 2]*/;
                    /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output[shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16][ArowIdx % 16] += reg7 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + shdmemColIdxBias][7] /*shmem_val7[shdmemColIdx % 2]*/;
                }
            }
            shmem[(i + 1) % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8)][(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8)] = next_iter_shmem_val_0;
            shmem[(i + 1) % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8) + 16][(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8)] = next_iter_shmem_val_2;
            __syncthreads();
        }

        // TODO: optimize thread mapping to improve access pattern
        // Disabling storage in fused kernel 
        // for (int store_iter = 0; store_iter < 256 * 32 / TILE_SZ_A; store_iter++)
        // {
        //     int node_idx_1 = store_iter;
        //     int ele_idx_1 = ArowIdx / 16;
        //     int ele_idx_2 = ArowIdx % 16;
        //     int CcolIdx = store_iter + /*blockIdx.x * TILE_SZ_B*/ BcolBias + shdmemColIdxBias;
        //     if (CcolIdx < n)
        //     {
        //         // C(ele_idx_1 * 16 / 64, ele_idx_2 + ele_idx_1 * 16 % 64, BcolBias + node_idx_1 + shdmemColIdxBias) = shmem_output[node_idx_1][ele_idx_1][shdmemColIdxBias / 16][ele_idx_2];

        //     }
        // }
        

        // now, each 256 thread is in charge of one source node, wherein each warp is in charge of one destination node at each time.
        for (int second_stage_node_idx = threadIdx.x / (OUT_DIM); second_stage_node_idx < TILE_SZ_B; second_stage_node_idx+= (blockDim.x/OUT_DIM)){
            // load dest node indices
            if (BcolBias+second_stage_node_idx>=n)
                break;
            int curr_source_node_index = node_indices[BcolBias+second_stage_node_idx];
            
            
            int num_dense_edge_for_curr_source_node = num_dense_edges_per_src_node[curr_source_node_index];
            int starting_index_dense_edge_for_curr_source_node = starting_pos_dense_edges_per_src_node[curr_source_node_index];

            for (int idx_dense_edge_for_curr_source_node = (threadIdx.x % (OUT_DIM)) / 32; idx_dense_edge_for_curr_source_node< num_dense_edge_for_curr_source_node; idx_dense_edge_for_curr_source_node+=((OUT_DIM)/32)){
                int curr_eid = eids[starting_index_dense_edge_for_curr_source_node + idx_dense_edge_for_curr_source_node];
                int curr_dest_node_index = dense_edges_per_src_node[starting_index_dense_edge_for_curr_source_node + idx_dense_edge_for_curr_source_node];
                

                int src_node_shmem_idx_1 = second_stage_node_idx/2;
                int src_node_shmem_idx_2 = second_stage_node_idx%2;
                
                float att_val =0.; float att_val_2 = 0.; float att_val_3 = 0.; float att_val_4 = 0.;
                for (int idx_feat = 0; idx_feat<64; idx_feat+=32){
                    int element_shmem_idx_1 = idx_feat / 16;
                    int element_shmem_idx_2 = idx_feat % 16;
                    // asserting IN_DIM==OUT_DIM
                    att_val += shmem_output[src_node_shmem_idx_1][element_shmem_idx_1][src_node_shmem_idx_2][element_shmem_idx_2] * Bdata[curr_dest_node_index *OUT_DIM + idx_feat];
                    att_val_2 += shmem_output[src_node_shmem_idx_1][element_shmem_idx_1+4][src_node_shmem_idx_2][element_shmem_idx_2] * Bdata[curr_dest_node_index * OUT_DIM + idx_feat + 1 * (OUT_DIM /NUM_HEADS)];
                    att_val_3 += shmem_output[src_node_shmem_idx_1][element_shmem_idx_1+8][src_node_shmem_idx_2][element_shmem_idx_2] * Bdata[curr_dest_node_index * OUT_DIM + idx_feat + 2 * (OUT_DIM /NUM_HEADS)];
                    att_val_4 += shmem_output[src_node_shmem_idx_1][element_shmem_idx_1+12][src_node_shmem_idx_2][element_shmem_idx_2] * Bdata[curr_dest_node_index * OUT_DIM + idx_feat + 3 * (OUT_DIM /NUM_HEADS)];
                }

                // reduction
                for (int i_reduction=16; i_reduction>0; i_reduction=i_reduction/2){
                    att_val += __shfl_down_sync(-1, att_val, i_reduction);
                    att_val_2 += __shfl_down_sync(-1, att_val_2, i_reduction);
                    att_val_3 += __shfl_down_sync(-1, att_val_3, i_reduction);
                    att_val_4 += __shfl_down_sync(-1, att_val_4, i_reduction);
                }
                if (threadIdx.x%32==0){
                    attention[curr_eid *4+0] += att_val;
                    attention[curr_eid *4+1] += att_val_2;
                    attention[curr_eid *4+2] += att_val_3;
                    attention[curr_eid *4+3] += att_val_4;
                }
            }
        }


#undef A
#undef B
#undef C
    }

__device__ __forceinline__ int binary_search(int num_elements, int * __restrict__ arr, int target){
    int lo = 0, hi = num_elements - 1;
    int mid;
    // This below check covers all cases , so need to check
    // for mid=lo-(hi-lo)/2
    while (hi - lo > 1) {
        int mid = (hi + lo) / 2;
        if (arr[mid] < target) {
            lo = mid + 1;
        }
        else {
            hi = mid;
        }
    }
    return lo;
}

// grid size equals num
template <int OUT_DIM, int NUM_HEADS>
__global__ void HGTExperimentalEdgeAttentionFusedCOOKernel_512_32(int num_relations, float* __restrict__ attention, float *__restrict__ node_input_data, float *__restrict__ relation_attention_matrices, 
                                                                int *__restrict__ num_src_nodes_per_edge_type, int* __restrict__ exclusive_scan_num_src_nodes_per_edge_type, int *__restrict__ exclusive_scan_num_blocks_per_relation, int* __restrict__ src_node_per_edge_type,
                                                                int* __restrict__ dense_edges_per_src_node, int* __restrict__ num_dense_edges_per_src_node, int* __restrict__ starting_pos_dense_edges_per_src_node,int* __restrict__ eids)
{
    constexpr int TILE_SZ_A = 512;
    constexpr int TILE_SZ_B = 32;
    constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
    constexpr int COARSE_SGEMM_NODES_PER_BLOCK = (TILE_SZ_B);

    int relation_idx = binary_search(num_relations, exclusive_scan_num_blocks_per_relation, blockIdx.x);
    assert( blockIdx.x>=exclusive_scan_num_blocks_per_relation[relation_idx]&&(relation_idx== num_relations -1  || blockIdx.x<exclusive_scan_num_blocks_per_relation[relation_idx+1]));
    int node_entry_idx = blockIdx.x * TILE_SZ_B - num_src_nodes_per_edge_type[relation_idx];
    //TODO: check source node boundary during the logic
    func512_32_mysgemm_exec<OUT_DIM, NUM_HEADS>(OUT_DIM, num_src_nodes_per_edge_type[relation_idx], NODE_INPUT_DIM_PER_HEAD, 
    attention, &relation_attention_matrices[relation_idx * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD * NODE_INPUT_DIM_PER_HEAD], node_input_data, 
    nullptr,&src_node_per_edge_type[exclusive_scan_num_src_nodes_per_edge_type[relation_idx]],dense_edges_per_src_node,num_dense_edges_per_src_node,starting_pos_dense_edges_per_src_node,eids,
    node_entry_idx);
}

template <int OUT_DIM, int NUM_HEADS>
__global__ void HGTExperimentalEdgeAttentionResidueCSR(int num_relations, int num_rows, float* __restrict__ attention, float *__restrict__ node_input_data, float *__restrict__ weights, int* __restrict__ num_nnzs_per_relation, int* __restrict__ rel_ptr, int* __restrict__ row_ptr, int* __restrict__ col_idx, int* __restrict__ eids, int* __restrict__ exclusive_scan_numBlocks_per_relationship){
    assert(OUT_DIM==256);
    assert(NUM_HEADS==4);
    // asserting IN_DIM == OUT_DIM
    // each block is in charge of one edge at a time and 32 edges in total
    int relation_idx = binary_search(num_relations, exclusive_scan_numBlocks_per_relationship, blockIdx.x);
    assert( blockIdx.x>=exclusive_scan_numBlocks_per_relationship[relation_idx]&&(relation_idx== num_relations -1  || blockIdx.x<exclusive_scan_numBlocks_per_relationship[relation_idx+1]));

    int srcIdx = binary_search(num_rows,row_ptr,blockIdx.x * 32 - num_nnzs_per_relation[relation_idx]);
    bool flagSrcIdxChanged = true;

    float intermediate_val = 0.0f;
    

    for (int edgeIdx = blockIdx.x * 32 - rel_ptr[relation_idx]; edgeIdx < num_nnzs_per_relation[relation_idx]; edgeIdx++){
        float att_val = 0.0f;
        while (edgeIdx+rel_ptr[relation_idx]>=row_ptr[relation_idx*num_rows+srcIdx+1]){
            srcIdx++;
            flagSrcIdxChanged = true;
            intermediate_val = 0.0f;
        }
        int dest_node_idx = col_idx[edgeIdx+rel_ptr[relation_idx]];
        int curr_eid = eids[edgeIdx+rel_ptr[relation_idx]];
        //calculate the index of the destination node feature (also the index of the intermedaite vector element)
        int idxHead = threadIdx.x / (OUT_DIM/NUM_HEADS);
        int idxElementWithinHead = threadIdx.x % (OUT_DIM/NUM_HEADS);
        if (flagSrcIdxChanged){
            for (int k=0; k<OUT_DIM/NUM_HEADS;k++){
                // NB: all matrices are column major
                intermediate_val += (node_input_data[(srcIdx * OUT_DIM) + idxHead * NUM_HEADS + k] * weights[relation_idx * NUM_HEADS * (OUT_DIM / NUM_HEADS) * (OUT_DIM / NUM_HEADS) + (idxHead) * (OUT_DIM / NUM_HEADS) * (OUT_DIM / NUM_HEADS) + idxElementWithinHead * (OUT_DIM / NUM_HEADS) + k]);
            }
        }
        
        att_val += intermediate_val * node_input_data[dest_node_idx*OUT_DIM + idxHead*(OUT_DIM/NUM_HEADS) + idxElementWithinHead];
        for (int i_reduction=16; i_reduction>0; i_reduction=i_reduction/2){
            att_val += __shfl_down_sync(-1, att_val, i_reduction);
        }
        if (threadIdx.x%32==0){
            atomicAdd(&attention[curr_eid *4+idxHead], att_val); // TODO: needs to clear to zero at the beginning
        }
        flagSrcIdxChanged=false; // we may reuse the intermediate data
    }
    

}

// TODO: implement float4
__global__ void HGTExpermentalEdgeAttentionConcatenatedSecondStageSrcInnerProductDestIntemediateCOOKernel(float *__restrict__ outEdges, int nnz, int *__restrict__ matCols, int *__restrict__ matRows, int *__restrict__ matRelation,
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
            outEdges[edge_idx*4] = product_1;
            outEdges[edge_idx*4+1] = product_2;
            outEdges[edge_idx*4+2] = product_3;
            outEdges[edge_idx*4+3] = product_4;
        }
    }
}


void HGTForwardImpl(MySegmentCSR<int, thrust::device_allocator<int>, MyHeteroSeparateCSR<int, thrust::device_allocator<int>>>& graph,
const int num_heads, const int in_feat, const int out_feat,
MySimpleNDArray<float, thrust::device_allocator<float>>& node_features,
MySimpleNDArray<float, thrust::device_allocator<float>>& weight,
MySimpleNDArray<float, thrust::device_allocator<float>>& attention
){
    assert(num_heads == 4 && "other cases not implemented");
    std::cout<<"WARNING: notice that in|out_feat is per_head but OUT_DIM is out_feat * num_heads"<<std::endl;
    assert(in_feat ==64 && "other cases not implemented");
    assert(out_feat == 64 && "other cases not implemented");


    
    std::vector<int> num_blocks_per_relationship_h(graph.num_rels);
    std::vector<int> exclusive_scan_numBlocks_per_relationship_h(graph.num_rels+1);
    exclusive_scan_numBlocks_per_relationship_h[0] = 0;
    for (int IdxRelation = 0; IdxRelation<graph.num_rels;IdxRelation++){
        num_blocks_per_relationship_h[IdxRelation]= my_ceil_div<int>(graph.num_src_nodes_per_edge_type[IdxRelation],32);
        exclusive_scan_numBlocks_per_relationship_h[IdxRelation+1] = exclusive_scan_numBlocks_per_relationship_h[IdxRelation] + num_blocks_per_relationship_h[IdxRelation];
    }
    thrust::device_vector<int> exclusive_scan_numBlocks_per_relationship(exclusive_scan_numBlocks_per_relationship_h.begin(),exclusive_scan_numBlocks_per_relationship_h.end());
    
    dim3 block(512, 1, 1);
    dim3 grid(exclusive_scan_numBlocks_per_relationship_h[exclusive_scan_numBlocks_per_relationship_h.size()-1], 1, 1);
    
    HGTExperimentalEdgeAttentionFusedCOOKernel_512_32<256, 4><<<grid, block>>>(graph.num_rels, attention.Ptr(), node_features.Ptr(), weight.Ptr(),
    static_cast<int*>(thrust::raw_pointer_cast(graph.num_src_nodes_per_edge_type.data())), 
    static_cast<int*>(thrust::raw_pointer_cast(graph.exclusive_scan_num_src_nodes_per_edge_type.data())),
    static_cast<int*>(thrust::raw_pointer_cast(exclusive_scan_numBlocks_per_relationship.data())),
    static_cast<int*>(thrust::raw_pointer_cast(graph.src_node_per_edge_type.data())), 
    static_cast<int*>(thrust::raw_pointer_cast(graph.padded_dense_edges.data())), 
    static_cast<int*>(thrust::raw_pointer_cast(graph.maximal_edge_num_per_src_node.data())), 
    static_cast<int*>(thrust::raw_pointer_cast(graph.padded_exclusive_scan_maximal_edge_num_per_src_node.data())), 
    static_cast<int*>(thrust::raw_pointer_cast(graph.padded_dense_edges_eids.data())));
    //int num_relations, float *__restrict__ node_input_data, float *__restrict__ relation_attention_matrices, 
    //                                                            int *__restrict__ num_src_nodes_per_edge_type, int *__restrict__ exclusive_scan_num_blocks_per_relation, int* __restrict__ src_node_per_edge_type, 
    //                                                            int* dense_edges_per_src_node, int* num_dense_edges_per_src_node,int* starting_pos_dense_edges_per_src_node,int* eids

    std::vector<int> residual_num_blocks_per_relationship_h(graph.num_rels);
    std::vector<int> residual_exclusive_scan_numBlocks_per_relationship_h(graph.num_rels+1);
    for (int IdxRelation = 0; IdxRelation<graph.num_rels; IdxRelation++){
        residual_num_blocks_per_relationship_h[IdxRelation]=my_ceil_div<int>(graph.csr.num_nnzs[IdxRelation],32);
        residual_exclusive_scan_numBlocks_per_relationship_h[IdxRelation+1] = residual_exclusive_scan_numBlocks_per_relationship_h[IdxRelation] + residual_num_blocks_per_relationship_h[IdxRelation];
    }

    thrust::device_vector<int> residual_exclusive_scan_numBlocks_per_relationship(residual_exclusive_scan_numBlocks_per_relationship_h.begin(),residual_exclusive_scan_numBlocks_per_relationship_h.end());
    thrust::device_vector<int> residual_num_nnzs_per_relation_device(graph.csr.num_nnzs.begin(),graph.csr.num_nnzs.end());
    thrust::device_vector<int> exclusive_scan_residual_num_nnzs_per_relation_device(graph.csr.num_rels);
    thrust::exclusive_scan(residual_num_nnzs_per_relation_device.begin(),residual_num_nnzs_per_relation_device.end(),exclusive_scan_residual_num_nnzs_per_relation_device.begin());

    dim3 block_residual(256, 1, 1);
    dim3 grid_residual(residual_exclusive_scan_numBlocks_per_relationship_h[residual_exclusive_scan_numBlocks_per_relationship_h.size()-1],1,1);


    HGTExperimentalEdgeAttentionResidueCSR<256, 4><<<grid_residual, block_residual>>>(graph.csr.num_rels, graph.csr.num_rows, attention.Ptr(), node_features.Ptr(), weight.Ptr(), 
    static_cast<int*>(thrust::raw_pointer_cast(residual_num_nnzs_per_relation_device.data())),
    static_cast<int*>(thrust::raw_pointer_cast(exclusive_scan_residual_num_nnzs_per_relation_device.data())),
    static_cast<int*>(thrust::raw_pointer_cast(graph.csr.row_ptr.data())),
    static_cast<int*>(thrust::raw_pointer_cast(graph.csr.col_idx.data())),
    static_cast<int*>(thrust::raw_pointer_cast(graph.csr.eids.data())),
    static_cast<int*>(thrust::raw_pointer_cast(residual_exclusive_scan_numBlocks_per_relationship.data())));


}