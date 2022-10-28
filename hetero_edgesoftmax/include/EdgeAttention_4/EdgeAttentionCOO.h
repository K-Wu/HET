#pragma once
#include "hetero_edgesoftmax.h"
#include "mysgemm_functor.cu.h"
//#include "EdgeAttentionCOO_128_16.h"

// __device__ __forceinline__ void
// _perRow_EdgeAttentionConcatenatedCOOKernel(int edge_idx, float **__restrict__
// outEdges_per_relation, int nnz, int *__restrict__ matCols, int *__restrict__
// relation,
//                                                                                    float *__restrict__ node_input_data, float *__restrict__ relation_attention_matrices)
// {
//     //@@ insert spmv kernel for coo format

//     int col = matCols[edge_idx];
//     float val = expf(edge_input_data[edge_idx]) + 1e-10f;
//     atomicAdd(&outEdges_per_relation[relation[edge_idx]][col], val);
// }

// __device__ __forceinline__ void _EdgeAttentionConcatenatedCOOKernel(int
// beg_edge_idx, int stride, float4 **__restrict__ outEdges_per_relation, int
// nnz, int *__restrict__ matCols, int *__restrict__ matRelation,
//                                                                             float *__restrict__ node_input_data, float *__restrict__ relation_attention_matrices)
// {
//     for (int edge_idx = beg_edge_idx; edge_idx < nnz; edge_idx += stride)
//     {
//         _perRow_EdgeAttentionConcatenatedCOOKernel(edge_idx,
//         outEdges_per_relation, nnz, matCols, matRelation, node_input_data,
//         relation_attention_matrices);
//     }
// }

// __global__ void EdgeAttentionConcatenatedCOOKernel(float4 **__restrict__
// outEdges_per_relation, int nnz, int *__restrict__ matCols, int *__restrict__
// matRelation,
//                                                  float *__restrict__
//                                                  node_input_data, float *
//                                                  __restrict__
//                                                  relation_attention_matrices)
// {
//     int beg_edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
//     _EdgeAttentionConcatenatedCOOKernel(beg_edge_idx, blockDim.x * gridDim.x,
//     outEdges_per_relation, nnz, matCols, matRelation, node_input_data,
//     relation_attention_matrices);
// }

// grid and thread configuration of the first stage
//   block (0,0): (head0 (64 element), 16 nodes), (head1 (64 element), 16
//   nodes); block(1,0): (head0 (64 element), 16 nodes), (head1 (64 element), 16
//   nodes); ... block(BLOCKDIM_X-1,0): (head0 (64 element), 16 nodes), (head1
//   (64 element), 16 nodes); block (0,1): (head2 (64 element), 16 nodes),
//   (head3 (64 element), 16 nodes); block(1,1): (head2 (64 element), 16 nodes),
//   (head3 (64 element), 16 nodes); ... block(BLOCKDIM_X-1,1): (head2 (64
//   element), 16 nodes), (head3 (64 element), 16 nodes);
//  problem definition
//#define TILE_SZ_A 128
//#define TILE_SZ_B 8
//#define OUT_DIM (256)
//#define NUM_HEADS (4)
// d_k
// #define NODE_INPUT_DIM_PER_HEAD (OUT_DIM / NUM_HEADS)

// #define TILE_SZ_RATIO (TILE_SZ_A / TILE_SZ_B)
// #define TILE_NUM_HEAD (TILE_SZ_A / NODE_INPUT_DIM_PER_HEAD)
//#define TILE_NUM_HEAD 1

// #define COARSE_SGEMM_BLOCKSIZE (TILE_SZ_A)
// #define COARSE_SGEMM_NODES_PER_BLOCK (TILE_SZ_B)

// TODO: extract this kernel mysgemm_ into template specialization
// constant static __device__ class function is allowed by CUDA spec
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#const-variables

// The original use should be <float4, true, false, float**>
template <typename OutDType, bool ProductCompactAsOfNodeFlag,
          bool EidEnableFlag, typename IntermediateProductPointerType>
__global__ void GeneralEdgeMessageMultiplyNodeFeature(
    OutDType *__restrict__ outEdges, int nnz, int *__restrict__ matCols,
    int *__restrict__ matRows, int *__restrict__ matRelation,
    int *__restrict__ matEids, float *__restrict__ node_input_data,
    IntermediateProductPointerType __restrict__ intermediate_product_per_edge_or_per_relation,
    int **__restrict__ dest_node_to_unique_index_per_relation) {
  // each warp is in charge of an edge
  int beg_edge_idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int lane_idx = (blockIdx.x * blockDim.x + threadIdx.x) % WARP_SIZE;
  for (int edge_idx = beg_edge_idx; edge_idx < nnz;
       edge_idx += (blockDim.x * gridDim.x) / WARP_SIZE) {
#define FULL_MASK 0xffffffff

    int col = matCols[edge_idx];
    int col_relation_idx;
    if constexpr (ProductCompactAsOfNodeFlag) {
      col_relation_idx =
          dest_node_to_unique_index_per_relation[matRelation[edge_idx]][col];
    }
    int edge_output_storage_idx = edge_idx;
    if constexpr (EidEnableFlag) {
      edge_output_storage_idx = matEids[edge_idx];
    }
    int row = matRows[edge_idx];
    float src_1 = node_input_data[row * 256 + lane_idx];
    float src_2 = node_input_data[row * 256 + 32 + lane_idx];
    float dest_1, dest_2, dest_3, dest_4, dest_5, dest_6, dest_7, dest_8;
    if constexpr (ProductCompactAsOfNodeFlag) {
      dest_1 =
          intermediate_product_per_edge_or_per_relation[matRelation[edge_idx]]
                                                       [col_relation_idx * 256 +
                                                        lane_idx];
      dest_2 =
          intermediate_product_per_edge_or_per_relation[matRelation[edge_idx]]
                                                       [col_relation_idx * 256 +
                                                        32 + lane_idx];
    } else {
      dest_1 =
          intermediate_product_per_edge_or_per_relation[col * 256 + lane_idx];
      dest_2 = intermediate_product_per_edge_or_per_relation[col * 256 + 32 +
                                                             lane_idx];
    }
    float product_1 = src_1 * dest_1 + src_2 * dest_2;
    for (int offset = 16; offset > 0; offset /= 2)
      product_1 += __shfl_down_sync(FULL_MASK, product_1, offset);

    float src_3 = node_input_data[row * 256 + 64 + lane_idx];
    float src_4 = node_input_data[row * 256 + 96 + lane_idx];
    if constexpr (ProductCompactAsOfNodeFlag) {
      dest_3 =
          intermediate_product_per_edge_or_per_relation[matRelation[edge_idx]]
                                                       [col_relation_idx * 256 +
                                                        64 + lane_idx];
      dest_4 =
          intermediate_product_per_edge_or_per_relation[matRelation[edge_idx]]
                                                       [col_relation_idx * 256 +
                                                        96 + lane_idx];
    } else {
      dest_3 = intermediate_product_per_edge_or_per_relation[col * 256 + 64 +
                                                             lane_idx];
      dest_4 = intermediate_product_per_edge_or_per_relation[col * 256 + 96 +
                                                             lane_idx];
    }
    float product_2 = src_3 * dest_3 + src_4 * dest_4;
    for (int offset = 16; offset > 0; offset /= 2)
      product_2 += __shfl_down_sync(FULL_MASK, product_2, offset);

    float src_5 = node_input_data[row * 256 + 128 + lane_idx];
    float src_6 = node_input_data[row * 256 + 160 + lane_idx];
    if constexpr (ProductCompactAsOfNodeFlag) {
      dest_5 =
          intermediate_product_per_edge_or_per_relation[matRelation[edge_idx]]
                                                       [col_relation_idx * 256 +
                                                        128 + lane_idx];
      dest_6 =
          intermediate_product_per_edge_or_per_relation[matRelation[edge_idx]]
                                                       [col_relation_idx * 256 +
                                                        160 + lane_idx];
    } else {
      dest_5 = intermediate_product_per_edge_or_per_relation[col * 256 + 128 +
                                                             lane_idx];
      dest_6 = intermediate_product_per_edge_or_per_relation[col * 256 + 160 +
                                                             lane_idx];
    }
    float product_3 = src_5 * dest_5 + src_6 * dest_6;
    for (int offset = 16; offset > 0; offset /= 2)
      product_3 += __shfl_down_sync(FULL_MASK, product_3, offset);

    float src_7 = node_input_data[row * 256 + 192 + lane_idx];
    float src_8 = node_input_data[row * 256 + 224 + lane_idx];
    if constexpr (ProductCompactAsOfNodeFlag) {
      dest_7 =
          intermediate_product_per_edge_or_per_relation[matRelation[edge_idx]]
                                                       [col_relation_idx * 256 +
                                                        192 + lane_idx];
      dest_8 =
          intermediate_product_per_edge_or_per_relation[matRelation[edge_idx]]
                                                       [col_relation_idx * 256 +
                                                        224 + lane_idx];
    } else {
      dest_7 = intermediate_product_per_edge_or_per_relation[col * 256 + 192 +
                                                             lane_idx];
      dest_8 = intermediate_product_per_edge_or_per_relation[col * 256 + 224 +
                                                             lane_idx];
    }
    float product_4 = src_7 * dest_7 + src_8 * dest_8;
    for (int offset = 16; offset > 0; offset /= 2)
      product_4 += __shfl_down_sync(FULL_MASK, product_4, offset);
    if (lane_idx == 0) {
      if constexpr (std::is_same<OutDType, float4>::value) {
        outEdges[edge_output_storage_idx] =
            make_float4(product_1, product_2, product_3, product_4);
      } else if constexpr (std::is_same<OutDType, float>::value) {
        // outEdges[4*edge_idx] = product_1;
        // outEdges[4*edge_idx+1] = product_2;
        // outEdges[4*edge_idx+2] = product_3;
        // outEdges[4*edge_idx+3] = product_4;
        reinterpret_cast<float4 *>(outEdges)[edge_output_storage_idx] =
            make_float4(product_1, product_2, product_3, product_4);
      } else {
        assert(0);
      }
    }
  }
}

constexpr auto
    EdgeAttentionConcatenatedSecondStageSrcInnerProductDestIntemediateCOOKernel =
        GeneralEdgeMessageMultiplyNodeFeature<float4, true, false, float **>;

// extract this kernel with mysgemm_ into template specialization
// template <int NODE_INPUT_DIM_PER_HEAD/*derived from OUT_DIM and NUM_HEADS*/,
// NUM_HEADS, OUT_DIM, COARSE_SGEMM_NODES_PER_BLOCK>
template <int TILE_SZ_A, int TILE_SZ_B, int OUT_DIM, int NUM_HEADS>
__global__ void EdgeAttentionConcatenatedFirstStageWeightMulDestCOOKernel(
    float **__restrict__ intermediate_node_vect, int nnz,
    int *__restrict__ matCols, int *__restrict__ matRelation,
    float *__restrict__ node_input_data,
    float *__restrict__ relation_attention_matrices,
    int **__restrict__ dest_node_to_unique_index_per_relation,
    int **__restrict__ unique_index_to_dest_node_per_relation,
    int *__restrict__ sizes_unique_index_to_dest_node_per_relation,
    int num_relations,
    int *__restrict__ num_blocks_xdim_for_same_relation_per_block_vect,
    int *__restrict__ beg_node_entry_idxes_vect,
    int *__restrict__ blockid_relation_id_vect) {
  constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
  constexpr int COARSE_SGEMM_NODES_PER_BLOCK = (TILE_SZ_B);
  int beg_node_entry_idx = beg_node_entry_idxes_vect[blockIdx.x];
  int stride = num_blocks_xdim_for_same_relation_per_block_vect[blockIdx.x] *
               COARSE_SGEMM_NODES_PER_BLOCK;
  int relation_idx = blockid_relation_id_vect[blockIdx.x];

  for (int node_entry_idx = beg_node_entry_idx;
       node_entry_idx <
       sizes_unique_index_to_dest_node_per_relation[relation_idx];
       node_entry_idx += stride) {
    mysgemm_functor<TILE_SZ_A, TILE_SZ_B, OUT_DIM, NUM_HEADS, true, false>::
        exec_function(
            OUT_DIM, sizes_unique_index_to_dest_node_per_relation[relation_idx],
            NODE_INPUT_DIM_PER_HEAD,
            &relation_attention_matrices[relation_idx * NUM_HEADS *
                                         NODE_INPUT_DIM_PER_HEAD *
                                         NODE_INPUT_DIM_PER_HEAD],
            node_input_data, intermediate_node_vect[relation_idx],
            unique_index_to_dest_node_per_relation[relation_idx], nullptr,
            node_entry_idx);
  }
}

// template <int NODE_INPUT_DIM_PER_HEAD/*derived from OUT_DIM and NUM_HEADS*/,
// NUM_HEADS, OUT_DIM, COARSE_SGEMM_NODES_PER_BLOCK /*derived from  TILE_SZ_B*/,
// COARSE_SGEMM_BLOCKSIZE /*derived fromTILE_SZ_A*/>
template <int TILE_SZ_A /*128*/, int TILE_SZ_B /*8*/, int OUT_DIM /*256*/,
          int NUM_HEADS /*4*/>
thrust::device_vector<float4>
EdgeAttentionConcatenatedSrcWeightMulDestCOOKernel(
    int num_nodes,
    cusp::coo_matrix<int, int, cusp::device_memory>::row_indices_array_type
        concatenated_coo_matrix_row_indices,
    cusp::coo_matrix<int, int, cusp::device_memory>::column_indices_array_type
        concatenated_coo_matrix_column_indices,
    std::vector<cusp::coo_matrix<
        int, int, cusp::device_memory>::column_indices_array_type>
        coo_matrices_column_indices,
    cusp::coo_matrix<int, int, cusp::device_memory>::values_array_type
        concatenated_coo_matrix_values,
    int num_relations, bool FlagInitWithRandomValue) {
  constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
  constexpr int COARSE_SGEMM_BLOCKSIZE = (TILE_SZ_A);
  constexpr int COARSE_SGEMM_NODES_PER_BLOCK = (TILE_SZ_B);

  std::vector<thrust::device_vector<float>> intermediate_node_vect(
      num_relations);
  thrust::device_vector<float *> intermediate_node_vect_d;
  thrust::device_vector<float4> outEdges_vect(
      concatenated_coo_matrix_column_indices.size(),
      make_float4(0.0f, 0.0f, 0.0f, 0.0f));

  std::vector<cusp::coo_matrix<int, int,
                               cusp::device_memory>::column_indices_array_type>
      coo_matrices_column_indices_unique(num_relations);
  thrust::device_vector<int *> unique_indices_to_column_indices_per_relation_d;
  thrust::device_vector<int> num_unique_indices_to_column_indices_per_relation(
      num_relations, -1);

  std::vector<thrust::device_vector<int>>
      dest_node_to_unique_index_per_relation =
          std::vector<thrust::device_vector<int>>(
              num_relations, thrust::device_vector<int>(num_nodes, -1));
  thrust::device_vector<int *> dest_node_to_unique_index_per_relation_d;

  for (int idx_relation = 0; idx_relation < num_relations; idx_relation++) {
    dest_node_to_unique_index_per_relation_d.push_back(thrust::raw_pointer_cast(
        dest_node_to_unique_index_per_relation[idx_relation].data()));
  }

  for (int idx_relation = 0; idx_relation < num_relations; idx_relation++) {
    coo_matrices_column_indices_unique[idx_relation] =
        coo_matrices_column_indices[idx_relation];
    std::cout << "curr_coo_matrix_column_indices_unique address"
              << thrust::raw_pointer_cast(
                     coo_matrices_column_indices_unique[idx_relation].data())
              << " original address: "
              << thrust::raw_pointer_cast(
                     coo_matrices_column_indices[idx_relation].data())
              << std::endl;
    thrust::sort(thrust::device,
                 coo_matrices_column_indices_unique[idx_relation].begin(),
                 coo_matrices_column_indices_unique[idx_relation].end());
    auto curr_unique_vector_end =
        thrust::unique(thrust::device,
                       coo_matrices_column_indices_unique[idx_relation].begin(),
                       coo_matrices_column_indices_unique[idx_relation].end());
    coo_matrices_column_indices_unique[idx_relation].resize(thrust::distance(
        coo_matrices_column_indices_unique[idx_relation].begin(),
        curr_unique_vector_end));
  }

  /*for (int idx_relation = 0; idx_relation < num_relations; idx_relation++) {
      std::cout << "relation " << idx_relation << "
  coo_matrices_column_indices_unique" <<std::endl;
      thrust::copy(coo_matrices_column_indices_unique[idx_relation].begin(),
  coo_matrices_column_indices_unique[idx_relation].end(),
  std::ostream_iterator<int>(std::cout, ","));
  }*/

  for (int idx_relation = 0; idx_relation < num_relations; idx_relation++) {
    thrust::counting_iterator<int> first_counting_iter(0);
    thrust::counting_iterator<int> last_counting_iter =
        first_counting_iter +
        coo_matrices_column_indices_unique[idx_relation].size();

    int *curr_dest_node_to_unique_index_per_relation_d =
        dest_node_to_unique_index_per_relation_d[idx_relation];
    /*thrust::for_each(thrust::device,
    coo_matrices_column_indices_unique[idx_relation].begin(),
        coo_matrices_column_indices_unique[idx_relation].end(),
        [=]__host__ __device__(int t) {
        curr_dest_node_to_unique_index_per_relation_d[t] = 1;
    });*/
    thrust::for_each(
        thrust::device,
        thrust::make_zip_iterator(thrust::make_tuple(
            coo_matrices_column_indices_unique[idx_relation].begin(),
            first_counting_iter)),
        thrust::make_zip_iterator(thrust::make_tuple(
            coo_matrices_column_indices_unique[idx_relation].end(),
            last_counting_iter)),
        [=] __host__ __device__(thrust::tuple<int, int> t) {
          curr_dest_node_to_unique_index_per_relation_d[thrust::get<0>(t)] =
              thrust::get<1>(t);
        });
    std::cout
        << "relation " << idx_relation
        << " dest_node_to_unique_index_per_relation_d[idx_relation] address: "
        << dest_node_to_unique_index_per_relation_d[idx_relation] << std::endl;
    /*thrust::copy(dest_node_to_unique_index_per_relation[idx_relation].begin(),
     * dest_node_to_unique_index_per_relation[idx_relation].end(),std::ostream_iterator<int>(std::cout,
     * ","));*/
  }

  for (int idx_relation = 0; idx_relation < num_relations; idx_relation++) {
    intermediate_node_vect[idx_relation].resize(
        coo_matrices_column_indices_unique[idx_relation].size() * OUT_DIM);
  }

  for (int idx_relation = 0; idx_relation < num_relations; idx_relation++) {
    intermediate_node_vect_d.push_back(
        thrust::raw_pointer_cast(intermediate_node_vect[idx_relation].data()));
  }

  for (int idx_relation = 0; idx_relation < num_relations; idx_relation++) {
    unique_indices_to_column_indices_per_relation_d.push_back(
        thrust::raw_pointer_cast(
            coo_matrices_column_indices_unique[idx_relation].data()));
    num_unique_indices_to_column_indices_per_relation[idx_relation] =
        coo_matrices_column_indices_unique[idx_relation].size();
  }

  // float *outEdges;
  // float *mus;

  float *node_input_data;
  float *relation_attention_matrices;
  cudaMalloc((void **)&node_input_data,
             sizeof(float) * num_nodes * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD);
  cudaMalloc((void **)&relation_attention_matrices,
             sizeof(float) * num_relations * NUM_HEADS *
                 NODE_INPUT_DIM_PER_HEAD * NODE_INPUT_DIM_PER_HEAD);
  // cudaMalloc((void **)&mus, sizeof(float) * num_relations);
  // cudaMalloc((void **)&outEdges, sizeof(float) *
  // concatenated_coo_matrix_column_indices.size());

  if (FlagInitWithRandomValue) {
    curandGenerator_t m_prng;
    // Create a new generator
    curandCreateGenerator(&m_prng, CURAND_RNG_PSEUDO_DEFAULT);
    // Set the generator options
    curandSetPseudoRandomGeneratorSeed(m_prng, (unsigned long)0);
    // Generate random numbers
    // curandGenerateUniform(m_prng, mus, num_relations);
    curandGenerateUniform(m_prng, node_input_data,
                          num_nodes * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD);
    curandGenerateUniform(m_prng, relation_attention_matrices,
                          num_relations * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD *
                              NODE_INPUT_DIM_PER_HEAD);
  } else {
    // cudaMemset(mus, 1, sizeof(float) * num_relations);
    cudaMemset(node_input_data, 64,
               sizeof(float) * num_nodes * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD);
    cudaMemset(relation_attention_matrices, 64,
               sizeof(float) * num_relations * NUM_HEADS *
                   NODE_INPUT_DIM_PER_HEAD * NODE_INPUT_DIM_PER_HEAD);
  }

  thrust::device_vector<int> num_blocks_xdim_for_same_relation_per_block_vect;
  thrust::device_vector<int> blockid_relation_id_vect;
  thrust::device_vector<int> beg_node_entry_idxes_vect;
  std::vector<int> num_blocks_xdim_for_same_relation_vect;
  std::vector<int> num_blocks_xdim_for_all_prev_relation_vect;
  num_blocks_xdim_for_all_prev_relation_vect.push_back(0);

  // for ease of programming equally partition the workload to different blocks
  // at this moment.
  for (int idx_relationship = 0; idx_relationship < num_relations;
       idx_relationship++) {
    int num_blocks_xdim_for_this_and_prev_relation =
        (idx_relationship + 1 + 0.0) / (num_relations + 0.0) *
        RTX_3090_GRIDSIZE;
    num_blocks_xdim_for_all_prev_relation_vect.push_back(
        num_blocks_xdim_for_this_and_prev_relation);
  }
  for (int idx_relationship = 0; idx_relationship < num_relations;
       idx_relationship++) {
    num_blocks_xdim_for_same_relation_vect.push_back(
        num_blocks_xdim_for_all_prev_relation_vect[idx_relationship + 1] -
        num_blocks_xdim_for_all_prev_relation_vect[idx_relationship]);
  }
  num_blocks_xdim_for_all_prev_relation_vect.erase(
      num_blocks_xdim_for_all_prev_relation_vect.begin());
  int idx_curr_relation = 0;
  int curr_beg_node_entry_idx = 0;

  // grid and thread configuration of the first stage
  //   block (0,0): (head0 (64 element), 16 nodes), (head1 (64 element), 16
  //   nodes); block(1,0): (head0 (64 element), 16 nodes), (head1 (64 element),
  //   16 nodes); ... block(BLOCKDIM_X-1,0): (head0 (64 element), 16 nodes),
  //   (head1 (64 element), 16 nodes); block (0,1): (head2 (64 element), 16
  //   nodes), (head3 (64 element), 16 nodes); block(1,1): (head2 (64 element),
  //   16 nodes), (head3 (64 element), 16 nodes); ... block(BLOCKDIM_X-1,1):
  //   (head2 (64 element), 16 nodes), (head3 (64 element), 16 nodes);

  for (int idx_block = 0; idx_block < RTX_3090_GRIDSIZE; idx_block++) {
    if (idx_curr_relation <
            num_blocks_xdim_for_all_prev_relation_vect.size() - 1 &&
        idx_block >=
            num_blocks_xdim_for_all_prev_relation_vect[idx_curr_relation]) {
      assert(curr_beg_node_entry_idx / COARSE_SGEMM_NODES_PER_BLOCK ==
             num_blocks_xdim_for_same_relation_vect[idx_curr_relation]);
      idx_curr_relation++;
      curr_beg_node_entry_idx = 0;
    }
    blockid_relation_id_vect.push_back(idx_curr_relation);
    beg_node_entry_idxes_vect.push_back(curr_beg_node_entry_idx);
    curr_beg_node_entry_idx += COARSE_SGEMM_NODES_PER_BLOCK;
    num_blocks_xdim_for_same_relation_per_block_vect.push_back(
        num_blocks_xdim_for_same_relation_vect[idx_curr_relation]);
  }

  dim3 block(COARSE_SGEMM_BLOCKSIZE, 1, 1);
  dim3 grid(RTX_3090_GRIDSIZE, 2, 1);
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  // EdgeAttentionConcatenatedCOOKernel<<<grid, block>>>(
  // thrust::raw_pointer_cast(outEdges_per_relation_vect.data()),
  // concatenated_coo_matrix_column_indices.size(),
  // thrust::raw_pointer_cast(concatenated_coo_matrix_column_indices.data()),
  // thrust::raw_pointer_cast(concatenated_coo_matrix_values.data()),
  // node_input_data);
  EdgeAttentionConcatenatedFirstStageWeightMulDestCOOKernel<
      TILE_SZ_A, TILE_SZ_B, OUT_DIM, NUM_HEADS><<<grid, block>>>(
      thrust::raw_pointer_cast(intermediate_node_vect_d.data()),
      concatenated_coo_matrix_column_indices.size(),
      thrust::raw_pointer_cast(concatenated_coo_matrix_column_indices.data()),
      thrust::raw_pointer_cast(concatenated_coo_matrix_values.data()),
      node_input_data, relation_attention_matrices,
      thrust::raw_pointer_cast(dest_node_to_unique_index_per_relation_d.data()),
      thrust::raw_pointer_cast(
          unique_indices_to_column_indices_per_relation_d.data()),
      thrust::raw_pointer_cast(
          num_unique_indices_to_column_indices_per_relation.data()),
      num_relations,
      thrust::raw_pointer_cast(
          num_blocks_xdim_for_same_relation_per_block_vect.data()),
      thrust::raw_pointer_cast(beg_node_entry_idxes_vect.data()),
      thrust::raw_pointer_cast(blockid_relation_id_vect.data()));
  dim3 block2(RTX_3090_BLOCKSIZE, 1, 1);
  dim3 grid2(RTX_3090_GRIDSIZE, 1, 1);
  EdgeAttentionConcatenatedSecondStageSrcInnerProductDestIntemediateCOOKernel<<<
      grid2, block2>>>(
      thrust::raw_pointer_cast(outEdges_vect.data()),
      concatenated_coo_matrix_column_indices.size(),
      thrust::raw_pointer_cast(concatenated_coo_matrix_column_indices.data()),
      thrust::raw_pointer_cast(concatenated_coo_matrix_row_indices.data()),
      thrust::raw_pointer_cast(concatenated_coo_matrix_values.data()), nullptr,
      node_input_data,
      thrust::raw_pointer_cast(intermediate_node_vect_d.data()),
      thrust::raw_pointer_cast(
          dest_node_to_unique_index_per_relation_d.data()));

  /*for (int idx_relation = 0; idx_relation < num_relations; idx_relation++)
  {
      std::cout << "intermediate_node_vect[" << idx_relation << "]" <<
  std::endl; thrust::copy(intermediate_node_vect[idx_relation].begin(),
  intermediate_node_vect[idx_relation].end(),
  std::ostream_iterator<float>(std::cout, ","));
  }*/
  thrust::copy(intermediate_node_vect[0].begin(),
               intermediate_node_vect[0].begin() + 1,
               std::ostream_iterator<float>(std::cout, ","));
  cuda_err_chk(cudaPeekAtLastError());
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::cout
      << "GPU doGPUEdgeAttentionConcatenatedCOO_128_8 Kernel time: "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << " us" << std::endl;
  // cudaFree(outEdges);
  cudaFree(node_input_data);
  cudaFree(relation_attention_matrices);
  return outEdges_vect;
}

thrust::device_vector<float4> doGPUEdgeAttentionConcatenatedCOOKernel_128_8(
    std::vector<cusp::coo_matrix<int, int, cusp::device_memory>> coo_matrices,
    cusp::coo_matrix<int, int, cusp::device_memory> concatenated_coo_matrix,
    int num_relations, bool FlagInitWithRandomValue) {
  std::vector<cusp::coo_matrix<int, int,
                               cusp::device_memory>::column_indices_array_type>
      coo_matrices_column_indices;
  for (int idx_relation = 0; idx_relation < coo_matrices.size();
       idx_relation++) {
    coo_matrices_column_indices.push_back(
        coo_matrices[idx_relation].column_indices);
  }
  return EdgeAttentionConcatenatedSrcWeightMulDestCOOKernel<128, 8, 256, 4>(
      concatenated_coo_matrix.num_rows, concatenated_coo_matrix.row_indices,
      concatenated_coo_matrix.column_indices, coo_matrices_column_indices,
      concatenated_coo_matrix.values, num_relations, FlagInitWithRandomValue);
}

thrust::device_vector<float4> doGPUEdgeAttentionConcatenatedCOOKernel_128_16(
    std::vector<cusp::coo_matrix<int, int, cusp::device_memory>> coo_matrices,
    cusp::coo_matrix<int, int, cusp::device_memory> concatenated_coo_matrix,
    int num_relations, bool FlagInitWithRandomValue) {
  std::vector<cusp::coo_matrix<int, int,
                               cusp::device_memory>::column_indices_array_type>
      coo_matrices_column_indices;
  for (int idx_relation = 0; idx_relation < coo_matrices.size();
       idx_relation++) {
    coo_matrices_column_indices.push_back(
        coo_matrices[idx_relation].column_indices);
  }
  return EdgeAttentionConcatenatedSrcWeightMulDestCOOKernel<128, 16, 256, 4>(
      concatenated_coo_matrix.num_rows, concatenated_coo_matrix.row_indices,
      concatenated_coo_matrix.column_indices, coo_matrices_column_indices,
      concatenated_coo_matrix.values, num_relations, FlagInitWithRandomValue);
}

// FIXME: enable the ignored FlagEqualWorkPartitionForBlocks flag
thrust::device_vector<float4> doGPUEdgeAttentionConcatenatedCOOKernel_256_32(
    std::vector<cusp::coo_matrix<int, int, cusp::device_memory>> coo_matrices,
    cusp::coo_matrix<int, int, cusp::device_memory> concatenated_coo_matrix,
    int num_relations, bool FlagInitWithRandomValue,
    bool FlagEqualWorkPartitionForBlocks) {
  std::cout << "WARNING: FlagEqualWorkPartitionForBlocks IGNORED!!"
            << std::endl;
  std::vector<cusp::coo_matrix<int, int,
                               cusp::device_memory>::column_indices_array_type>
      coo_matrices_column_indices;
  for (int idx_relation = 0; idx_relation < coo_matrices.size();
       idx_relation++) {
    coo_matrices_column_indices.push_back(
        coo_matrices[idx_relation].column_indices);
  }
  return EdgeAttentionConcatenatedSrcWeightMulDestCOOKernel<256, 32, 256, 4>(
      concatenated_coo_matrix.num_rows, concatenated_coo_matrix.row_indices,
      concatenated_coo_matrix.column_indices, coo_matrices_column_indices,
      concatenated_coo_matrix.values, num_relations, FlagInitWithRandomValue);
}

// FIXME: enable the ignored FlagEqualWorkPartitionForBlocks flag
thrust::device_vector<float4> doGPUEdgeAttentionConcatenatedCOOKernel_256_8(
    std::vector<cusp::coo_matrix<int, int, cusp::device_memory>> coo_matrices,
    cusp::coo_matrix<int, int, cusp::device_memory> concatenated_coo_matrix,
    int num_relations, bool FlagInitWithRandomValue,
    bool FlagEqualWorkPartitionForBlocks) {
  std::cout << "WARNING: FlagEqualWorkPartitionForBlocks IGNORED!!"
            << std::endl;
  std::vector<cusp::coo_matrix<int, int,
                               cusp::device_memory>::column_indices_array_type>
      coo_matrices_column_indices;
  for (int idx_relation = 0; idx_relation < coo_matrices.size();
       idx_relation++) {
    coo_matrices_column_indices.push_back(
        coo_matrices[idx_relation].column_indices);
  }
  return EdgeAttentionConcatenatedSrcWeightMulDestCOOKernel<256, 8, 256, 4>(
      concatenated_coo_matrix.num_rows, concatenated_coo_matrix.row_indices,
      concatenated_coo_matrix.column_indices, coo_matrices_column_indices,
      concatenated_coo_matrix.values, num_relations, FlagInitWithRandomValue);
}

template <int OUT_DIM /*256*/, int NUM_HEADS /*4*/>
thrust::device_vector<float4>
EdgeAttentionConcatenatedSrcWeightMulDestCOOKernel_512_32(
    int num_nodes,
    cusp::coo_matrix<int, int, cusp::device_memory>::row_indices_array_type
        concatenated_coo_matrix_row_indices,
    cusp::coo_matrix<int, int, cusp::device_memory>::column_indices_array_type
        concatenated_coo_matrix_column_indices,
    std::vector<cusp::coo_matrix<
        int, int, cusp::device_memory>::column_indices_array_type>
        coo_matrices_column_indices,
    cusp::coo_matrix<int, int, cusp::device_memory>::values_array_type
        concatenated_coo_matrix_values,
    int num_relations, bool FlagInitWithRandomValue) {
  constexpr int TILE_SZ_A = 512;
  constexpr int TILE_SZ_B = 32;
  constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
  constexpr int COARSE_SGEMM_BLOCKSIZE = (TILE_SZ_A);
  constexpr int COARSE_SGEMM_NODES_PER_BLOCK = (TILE_SZ_B);

  std::vector<thrust::device_vector<float>> intermediate_node_vect(
      num_relations);
  thrust::device_vector<float *> intermediate_node_vect_d;
  thrust::device_vector<float4> outEdges_vect(
      concatenated_coo_matrix_column_indices.size(),
      make_float4(0.0f, 0.0f, 0.0f, 0.0f));

  std::vector<cusp::coo_matrix<int, int,
                               cusp::device_memory>::column_indices_array_type>
      coo_matrices_column_indices_unique(num_relations);
  thrust::device_vector<int *> unique_indices_to_column_indices_per_relation_d;
  thrust::device_vector<int> num_unique_indices_to_column_indices_per_relation(
      num_relations, -1);

  std::vector<thrust::device_vector<int>>
      dest_node_to_unique_index_per_relation =
          std::vector<thrust::device_vector<int>>(
              num_relations, thrust::device_vector<int>(num_nodes, -1));
  thrust::device_vector<int *> dest_node_to_unique_index_per_relation_d;

  for (int idx_relation = 0; idx_relation < num_relations; idx_relation++) {
    dest_node_to_unique_index_per_relation_d.push_back(thrust::raw_pointer_cast(
        dest_node_to_unique_index_per_relation[idx_relation].data()));
  }

  for (int idx_relation = 0; idx_relation < num_relations; idx_relation++) {
    coo_matrices_column_indices_unique[idx_relation] =
        coo_matrices_column_indices[idx_relation];
    std::cout << "curr_coo_matrix_column_indices_unique address"
              << thrust::raw_pointer_cast(
                     coo_matrices_column_indices_unique[idx_relation].data())
              << " original address: "
              << thrust::raw_pointer_cast(
                     coo_matrices_column_indices[idx_relation].data())
              << std::endl;
    thrust::sort(thrust::device,
                 coo_matrices_column_indices_unique[idx_relation].begin(),
                 coo_matrices_column_indices_unique[idx_relation].end());
    auto curr_unique_vector_end =
        thrust::unique(thrust::device,
                       coo_matrices_column_indices_unique[idx_relation].begin(),
                       coo_matrices_column_indices_unique[idx_relation].end());
    coo_matrices_column_indices_unique[idx_relation].resize(thrust::distance(
        coo_matrices_column_indices_unique[idx_relation].begin(),
        curr_unique_vector_end));
  }

  /*for (int idx_relation = 0; idx_relation < num_relations; idx_relation++) {
      std::cout << "relation " << idx_relation << "
  coo_matrices_column_indices_unique" <<std::endl;
      thrust::copy(coo_matrices_column_indices_unique[idx_relation].begin(),
  coo_matrices_column_indices_unique[idx_relation].end(),
  std::ostream_iterator<int>(std::cout, ","));
  }*/

  for (int idx_relation = 0; idx_relation < num_relations; idx_relation++) {
    thrust::counting_iterator<int> first_counting_iter(0);
    thrust::counting_iterator<int> last_counting_iter =
        first_counting_iter +
        coo_matrices_column_indices_unique[idx_relation].size();

    int *curr_dest_node_to_unique_index_per_relation_d =
        dest_node_to_unique_index_per_relation_d[idx_relation];
    /*thrust::for_each(thrust::device,
    coo_matrices_column_indices_unique[idx_relation].begin(),
        coo_matrices_column_indices_unique[idx_relation].end(),
        [=]__host__ __device__(int t) {
        curr_dest_node_to_unique_index_per_relation_d[t] = 1;
    });*/
    thrust::for_each(
        thrust::device,
        thrust::make_zip_iterator(thrust::make_tuple(
            coo_matrices_column_indices_unique[idx_relation].begin(),
            first_counting_iter)),
        thrust::make_zip_iterator(thrust::make_tuple(
            coo_matrices_column_indices_unique[idx_relation].end(),
            last_counting_iter)),
        [=] __host__ __device__(thrust::tuple<int, int> t) {
          curr_dest_node_to_unique_index_per_relation_d[thrust::get<0>(t)] =
              thrust::get<1>(t);
        });
    std::cout
        << "relation " << idx_relation
        << " dest_node_to_unique_index_per_relation_d[idx_relation] address: "
        << dest_node_to_unique_index_per_relation_d[idx_relation] << std::endl;
    /*thrust::copy(dest_node_to_unique_index_per_relation[idx_relation].begin(),
     * dest_node_to_unique_index_per_relation[idx_relation].end(),std::ostream_iterator<int>(std::cout,
     * ","));*/
  }

  for (int idx_relation = 0; idx_relation < num_relations; idx_relation++) {
    intermediate_node_vect[idx_relation].resize(
        coo_matrices_column_indices_unique[idx_relation].size() * OUT_DIM);
  }

  for (int idx_relation = 0; idx_relation < num_relations; idx_relation++) {
    intermediate_node_vect_d.push_back(
        thrust::raw_pointer_cast(intermediate_node_vect[idx_relation].data()));
  }

  for (int idx_relation = 0; idx_relation < num_relations; idx_relation++) {
    unique_indices_to_column_indices_per_relation_d.push_back(
        thrust::raw_pointer_cast(
            coo_matrices_column_indices_unique[idx_relation].data()));
    num_unique_indices_to_column_indices_per_relation[idx_relation] =
        coo_matrices_column_indices_unique[idx_relation].size();
  }

  // float *outEdges;
  // float *mus;

  float *node_input_data;
  float *relation_attention_matrices;
  cudaMalloc((void **)&node_input_data,
             sizeof(float) * num_nodes * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD);
  cudaMalloc((void **)&relation_attention_matrices,
             sizeof(float) * num_relations * NUM_HEADS *
                 NODE_INPUT_DIM_PER_HEAD * NODE_INPUT_DIM_PER_HEAD);
  // cudaMalloc((void **)&mus, sizeof(float) * num_relations);
  // cudaMalloc((void **)&outEdges, sizeof(float) *
  // concatenated_coo_matrix_column_indices.size());

  if (FlagInitWithRandomValue) {
    curandGenerator_t m_prng;
    // Create a new generator
    curandCreateGenerator(&m_prng, CURAND_RNG_PSEUDO_DEFAULT);
    // Set the generator options
    curandSetPseudoRandomGeneratorSeed(m_prng, (unsigned long)0);
    // Generate random numbers
    // curandGenerateUniform(m_prng, mus, num_relations);
    curandGenerateUniform(m_prng, node_input_data,
                          num_nodes * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD);
    curandGenerateUniform(m_prng, relation_attention_matrices,
                          num_relations * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD *
                              NODE_INPUT_DIM_PER_HEAD);
  } else {
    // cudaMemset(mus, 1, sizeof(float) * num_relations);
    cudaMemset(node_input_data, 64,
               sizeof(float) * num_nodes * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD);
    cudaMemset(relation_attention_matrices, 64,
               sizeof(float) * num_relations * NUM_HEADS *
                   NODE_INPUT_DIM_PER_HEAD * NODE_INPUT_DIM_PER_HEAD);
  }

  thrust::device_vector<int> num_blocks_xdim_for_same_relation_per_block_vect;
  thrust::device_vector<int> blockid_relation_id_vect;
  thrust::device_vector<int> beg_node_entry_idxes_vect;
  std::vector<int> num_blocks_xdim_for_same_relation_vect;
  std::vector<int> num_blocks_xdim_for_all_prev_relation_vect;

  std::chrono::high_resolution_clock::time_point t1;
  // for ease of programming equally partition the workload to different blocks
  // at this moment.
  // if (FlagEqualWorkPartitionForBlocks)
  //{
  num_blocks_xdim_for_all_prev_relation_vect.push_back(0);
  for (int idx_relationship = 0; idx_relationship < num_relations;
       idx_relationship++) {
    int num_blocks_xdim_for_this_and_prev_relation =
        (idx_relationship + 1 + 0.0) / (num_relations + 0.0) *
        RTX_3090_GRIDSIZE;
    num_blocks_xdim_for_all_prev_relation_vect.push_back(
        num_blocks_xdim_for_this_and_prev_relation);
  }
  for (int idx_relationship = 0; idx_relationship < num_relations;
       idx_relationship++) {
    num_blocks_xdim_for_same_relation_vect.push_back(
        num_blocks_xdim_for_all_prev_relation_vect[idx_relationship + 1] -
        num_blocks_xdim_for_all_prev_relation_vect[idx_relationship]);
  }
  num_blocks_xdim_for_all_prev_relation_vect.erase(
      num_blocks_xdim_for_all_prev_relation_vect.begin());
  int idx_curr_relation = 0;
  int curr_beg_node_entry_idx = 0;

  // grid and thread configuration of the first stage
  //   block (0,0): (head0 (64 element), 16 nodes), (head1 (64 element), 16
  //   nodes); block(1,0): (head0 (64 element), 16 nodes), (head1 (64 element),
  //   16 nodes); ... block(BLOCKDIM_X-1,0): (head0 (64 element), 16 nodes),
  //   (head1 (64 element), 16 nodes); block (0,1): (head2 (64 element), 16
  //   nodes), (head3 (64 element), 16 nodes); block(1,1): (head2 (64 element),
  //   16 nodes), (head3 (64 element), 16 nodes); ... block(BLOCKDIM_X-1,1):
  //   (head2 (64 element), 16 nodes), (head3 (64 element), 16 nodes);

  for (int idx_block = 0; idx_block < RTX_3090_GRIDSIZE; idx_block++) {
    if (idx_curr_relation <
            num_blocks_xdim_for_all_prev_relation_vect.size() - 1 &&
        idx_block >=
            num_blocks_xdim_for_all_prev_relation_vect[idx_curr_relation]) {
      assert(curr_beg_node_entry_idx / COARSE_SGEMM_NODES_PER_BLOCK ==
             num_blocks_xdim_for_same_relation_vect[idx_curr_relation]);
      idx_curr_relation++;
      curr_beg_node_entry_idx = 0;
    }
    blockid_relation_id_vect.push_back(idx_curr_relation);
    beg_node_entry_idxes_vect.push_back(curr_beg_node_entry_idx);
    curr_beg_node_entry_idx += COARSE_SGEMM_NODES_PER_BLOCK;
    num_blocks_xdim_for_same_relation_per_block_vect.push_back(
        num_blocks_xdim_for_same_relation_vect[idx_curr_relation]);
  }

  dim3 block(COARSE_SGEMM_BLOCKSIZE, 1, 1);
  dim3 grid(RTX_3090_GRIDSIZE, 1, 1);
  t1 = std::chrono::high_resolution_clock::now();
  // EdgeAttentionConcatenatedCOOKernel<<<grid, block>>>(
  // thrust::raw_pointer_cast(outEdges_per_relation_vect.data()),
  // concatenated_coo_matrix_column_indices.size(),
  // thrust::raw_pointer_cast(concatenated_coo_matrix_column_indices.data()),
  // thrust::raw_pointer_cast(concatenated_coo_matrix_values.data()),
  // node_input_data);
  EdgeAttentionConcatenatedFirstStageWeightMulDestCOOKernel<512, 32, 256, 4>
      <<<grid, block>>>(
          thrust::raw_pointer_cast(intermediate_node_vect_d.data()),
          concatenated_coo_matrix_column_indices.size(),
          thrust::raw_pointer_cast(
              concatenated_coo_matrix_column_indices.data()),
          thrust::raw_pointer_cast(concatenated_coo_matrix_values.data()),
          node_input_data, relation_attention_matrices,
          thrust::raw_pointer_cast(
              dest_node_to_unique_index_per_relation_d.data()),
          thrust::raw_pointer_cast(
              unique_indices_to_column_indices_per_relation_d.data()),
          thrust::raw_pointer_cast(
              num_unique_indices_to_column_indices_per_relation.data()),
          num_relations,
          thrust::raw_pointer_cast(
              num_blocks_xdim_for_same_relation_per_block_vect.data()),
          thrust::raw_pointer_cast(beg_node_entry_idxes_vect.data()),
          thrust::raw_pointer_cast(blockid_relation_id_vect.data()));
  //}

  dim3 block2(RTX_3090_BLOCKSIZE, 1, 1);
  dim3 grid2(RTX_3090_GRIDSIZE, 1, 1);
  EdgeAttentionConcatenatedSecondStageSrcInnerProductDestIntemediateCOOKernel<<<
      grid2, block2>>>(
      thrust::raw_pointer_cast(outEdges_vect.data()),
      concatenated_coo_matrix_column_indices.size(),
      thrust::raw_pointer_cast(concatenated_coo_matrix_column_indices.data()),
      thrust::raw_pointer_cast(concatenated_coo_matrix_row_indices.data()),
      thrust::raw_pointer_cast(concatenated_coo_matrix_values.data()), nullptr,
      node_input_data,
      thrust::raw_pointer_cast(intermediate_node_vect_d.data()),
      thrust::raw_pointer_cast(
          dest_node_to_unique_index_per_relation_d.data()));

  // for (int idx_relation = 0; idx_relation < num_relations; idx_relation++)
  // {
  //     std::cout << "intermediate_node_vect[" << idx_relation << "]" <<
  //     std::endl; thrust::copy(intermediate_node_vect[idx_relation].begin(),
  //     intermediate_node_vect[idx_relation].end(),
  //     std::ostream_iterator<float>(std::cout, ","));
  // }
  cuda_err_chk(cudaPeekAtLastError());
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::cout
      << "GPU doGPUEdgeAttentionConcatenatedCOO_512_32 Kernel time: "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << " us" << std::endl;
  // cudaFree(outEdges);
  cudaFree(node_input_data);
  cudaFree(relation_attention_matrices);
  return outEdges_vect;
}

// FIXME: enable the ignored FlagEqualWorkPartitionForBlocks flag
thrust::device_vector<float4> doGPUEdgeAttentionConcatenatedCOOKernel_512_32(
    std::vector<cusp::coo_matrix<int, int, cusp::device_memory>> coo_matrices,
    cusp::coo_matrix<int, int, cusp::device_memory> concatenated_coo_matrix,
    int num_relations, bool FlagInitWithRandomValue,
    bool FlagEqualWorkPartitionForBlocks) {
  std::cout << "ignoring FlagEqualWorkPartitionForBlocks" << std::endl;
  std::vector<cusp::coo_matrix<int, int,
                               cusp::device_memory>::column_indices_array_type>
      coo_matrices_column_indices;
  for (int idx_relation = 0; idx_relation < coo_matrices.size();
       idx_relation++) {
    coo_matrices_column_indices.push_back(
        coo_matrices[idx_relation].column_indices);
  }
  return EdgeAttentionConcatenatedSrcWeightMulDestCOOKernel_512_32<256, 4>(
      concatenated_coo_matrix.num_rows, concatenated_coo_matrix.row_indices,
      concatenated_coo_matrix.column_indices, coo_matrices_column_indices,
      concatenated_coo_matrix.values, num_relations, FlagInitWithRandomValue);
}

// #undef COARSE_SGEMM_BLOCKSIZE
// #undef COARSE_SGEMM_NODES_PER_BLOCK
// #undef OUT_DIM
// #undef NUM_HEADS
// #undef NODE_INPUT_DIM_PER_HEAD