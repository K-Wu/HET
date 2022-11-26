#pragma once
#include "DGLHackKernel/DGLHackKernel.h"

template <typename Idx, typename DType>
__global__ void HGTBackwardFusedGradientSmFirstPartGradientAImpl(
    Idx* ranges, Idx* dst_ids, Idx* eids, Idx* types,
    DType* grad_a,               // |E| * N_HEADS
    DType* grad_sm_first_stage,  //|V| * N_REL_TYPES * N_HEADS * DIM_PER_HEAD
    DType* grad_t_neighbour,     //|V| * N_HEADS * DIM_PER_HEAD
    DType* message,              //|E| * N_HEADS * DIM_PER_HEAD
    DType* sigmas,               //|E| * N_HEADS
    Idx num_nodes, Idx num_heads, Idx feat_dim_per_head, Idx n_rel_types) {
  assert(n_rel_types == 2);
  // delta a = delta t_neighbour^(l+1) * sigma^-1 * m^T
  int lane_idx = threadIdx.x % 32;
  if (blockIdx.x < num_nodes) {
    Idx beg = __ldg(ranges + blockIdx.x);
    Idx end = __ldg(ranges + blockIdx.x + 1);
    Idx tx = threadIdx.x;
    for (; tx < feat_dim_per_head * num_heads; tx += blockDim.x) {
      // each block deald with one source node, and each thread deals with one
      // element along the feature dimension

      Idx tidx_head = tx / feat_dim_per_head;
      Idx tidx_ele_in_head = tx % feat_dim_per_head;
      // load delta t neighbor
      DType delta_t_neighbour_ele =
          __ldg(grad_t_neighbour + blockIdx.x * num_heads * feat_dim_per_head +
                tidx_head * feat_dim_per_head + tidx_ele_in_head);
      DType agg0 = 0.;
      DType agg1 = 0.;
      for (; beg < end; beg++) {
        // dealing with each edge
        // broadcast sigma. each thread load one element from m.
        Idx dst_id = __ldg(dst_ids + beg);
        Idx eid = __ldg(eids + beg);
        Idx type_id = __ldg(types + beg);
        DType msg_ele = __ldg(message + eid * num_heads * feat_dim_per_head +
                              tidx_head * feat_dim_per_head + tidx_ele_in_head);
        DType sigma = __ldg(sigmas + eid * num_heads + tidx_head);
        if (type_id < 2) {
          agg0 += sigma * msg_ele * delta_t_neighbour_ele;
        } else {
          agg1 += sigma * msg_ele * delta_t_neighbour_ele;
        }
        DType inner_agg = sigma * (1 - sigma) * msg_ele * delta_t_neighbour_ele;
// agg += g*w*n;
#pragma unroll
        for (int i_reduction = 16; i_reduction > 0;
             i_reduction = i_reduction / 2) {
          inner_agg += __shfl_down_sync(-1, inner_agg, i_reduction);
        }

        if (lane_idx == 0) {
          atomicAdd(grad_a + eid * num_heads + tidx_head, inner_agg);
        }
      }
      atomicAdd(grad_sm_first_stage +
                    blockIdx.x * n_rel_types * num_heads * feat_dim_per_head +
                    0 * num_heads * feat_dim_per_head + tx,
                agg0);
      atomicAdd(grad_sm_first_stage +
                    blockIdx.x * n_rel_types * num_heads * feat_dim_per_head +
                    1 * num_heads * feat_dim_per_head + tx,
                agg1);
    }
  }
}

template <typename Idx, typename DType>
__global__ void HGTBackwardGradientAImpl(
    Idx* ranges, Idx* dst_ids, Idx* eids, Idx* types,
    DType* grad_a,            // |E| * N_HEADS
    DType* grad_t_neighbour,  //|V| * N_HEADS * DIM_PER_HEAD
    DType* message,           //|E| * N_HEADS * DIM_PER_HEAD
    DType* sigmas,            //|E| * N_HEADS
    Idx num_nodes, Idx num_heads, Idx feat_dim_per_head, Idx n_rel_types) {
  assert(n_rel_types == 2);

  // delta a = delta t_neighbour^(l+1) * sigma^-1 * m^T
  int lane_idx = threadIdx.x % 32;
  if (blockIdx.x < num_nodes) {
    Idx beg = __ldg(ranges + blockIdx.x);
    Idx end = __ldg(ranges + blockIdx.x + 1);
    Idx tx = threadIdx.x;
    for (; tx < feat_dim_per_head * num_heads; tx += blockDim.x) {
      // each block deald with one source node, and each thread deals with one
      // element along the feature dimension

      Idx tidx_head = tx / feat_dim_per_head;
      Idx tidx_ele_in_head = tx % feat_dim_per_head;
      // load delta t neighbor
      DType delta_t_neighbour_ele =
          __ldg(grad_t_neighbour + blockIdx.x * num_heads * feat_dim_per_head +
                tidx_head * feat_dim_per_head + tidx_ele_in_head);

      for (; beg < end; beg++) {
        // dealing with each edge
        // broadcast sigma. each thread load one element from m.
        Idx dst_id = __ldg(dst_ids + beg);
        Idx eid = __ldg(eids + beg);
        Idx type_id = __ldg(types + beg);
        DType msg_ele = __ldg(message + eid * num_heads * feat_dim_per_head +
                              tidx_head * feat_dim_per_head + tidx_ele_in_head);
        DType sigma = __ldg(sigmas + eid * num_heads + tidx_head);
        DType inner_agg = sigma * (1 - sigma) * msg_ele * delta_t_neighbour_ele;
// agg += g*w*n;
#pragma unroll
        for (int i_reduction = 16; i_reduction > 0;
             i_reduction = i_reduction / 2) {
          inner_agg += __shfl_down_sync(-1, inner_agg, i_reduction);
        }

        if (lane_idx == 0) {
          atomicAdd(grad_a + eid * num_heads + tidx_head, inner_agg);
        }
      }
    }
  }
}

template <typename Idx, typename DType>
__global__ void HGTBackwardGradientSmFirstPartImpl(
    Idx* ranges, Idx* dst_ids, Idx* eids, Idx* types,
    DType* grad_sm_first_stage,  //|V| * N_REL_TYPES * N_HEADS * DIM_PER_HEAD
    DType* grad_t_neighbour,     //|V| * N_HEADS * DIM_PER_HEAD
    DType* message,              //|E| * N_HEADS * DIM_PER_HEAD
    DType* sigmas,               //|E| * N_HEADS
    Idx num_nodes, Idx num_heads, Idx feat_dim_per_head, Idx n_rel_types) {
  assert(n_rel_types == 2);  // some bit manipulation is used and thus the
                             // kernel is intended for MAG only

  // delta Sm = \Sum_outgoing (m * delta t_neighbour^(l+1) * sigma)
  // We need to store one delta Sm for each relationship type
  if (blockIdx.x < num_nodes) {
    Idx beg = __ldg(ranges + blockIdx.x);
    Idx end = __ldg(ranges + blockIdx.x + 1);
    Idx tx = threadIdx.x;
    for (; tx < feat_dim_per_head * num_heads; tx += blockDim.x) {
      // each block deald with one source node, and each thread deals with one
      // element along the feature dimension

      Idx tidx_head = tx / feat_dim_per_head;
      Idx tidx_ele_in_head = tx % feat_dim_per_head;
      // load delta t neighbor
      DType delta_t_neighbour_ele =
          __ldg(grad_t_neighbour + blockIdx.x * num_heads * feat_dim_per_head +
                tidx_head * feat_dim_per_head + tidx_ele_in_head);

      // simple hack here effective for OGBN MAG: for grad_sm_first_stage, map
      // relaationship 0, 1 to first half and relationship type 2,3 to the
      // second half, thus reducing 50% footprint
      DType agg0 = 0.;
      DType agg1 = 0.;
      for (; beg < end; beg++) {
        // dealing with each edge
        //  broadcast sigma
        Idx dst_id = __ldg(dst_ids + beg);
        Idx eid = __ldg(eids + beg);
        Idx type_id = __ldg(types + beg);
        DType msg_ele = __ldg(message + eid * num_heads * feat_dim_per_head +
                              tidx_head * feat_dim_per_head + tidx_ele_in_head);
        DType sigma = __ldg(sigmas + eid * num_heads + tidx_head);
        if (type_id < 2) {
          agg0 += sigma * msg_ele * delta_t_neighbour_ele;
        } else {
          agg1 += sigma * msg_ele * delta_t_neighbour_ele;
        }
        // atomicAdd();
      }
      atomicAdd(grad_sm_first_stage +
                    blockIdx.x * n_rel_types * num_heads * feat_dim_per_head +
                    0 * num_heads * feat_dim_per_head + tx,
                agg0);
      atomicAdd(grad_sm_first_stage +
                    blockIdx.x * n_rel_types * num_heads * feat_dim_per_head +
                    1 * num_heads * feat_dim_per_head + tx,
                agg1);
    }
  }
}

template </*int XPU, */ typename Idx, typename DType>
void HGTBackPropGradientSMAFusion(
    // GraphRef graph,
    MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>> csr,
    MySimpleNDArray<DType, thrust::device_allocator<DType>>&
        grad_sm_first_stage,
    MySimpleNDArray<DType, thrust::device_allocator<DType>>& grad_a,
    MySimpleNDArray<DType, thrust::device_allocator<DType>>& grad_t_neighbour,
    MySimpleNDArray<DType, thrust::device_allocator<DType>>& message,
    MySimpleNDArray<DType, thrust::device_allocator<DType>>& sigmas) {
  // LOG(INFO) << "Calling implementation of rgn layer 1 forward";
  // assert(csr.IsSortedByEdgeType_CPU());
  // typedef int32_t Idx;
  // typedef float DType;
  // auto csr = graph->GetCsrSortedByEdgeType(false);
  // auto ranges = csr[0];
  // auto ids = csr[1];
  // auto eids = csr[2];
  // auto type_ids = csr[3];
  auto range_data =
      static_cast<Idx*>(thrust::raw_pointer_cast(csr.row_ptr.data()));
  auto ids_data =
      static_cast<Idx*>(thrust::raw_pointer_cast(csr.col_idx.data()));
  // auto eids_data = static_cast<Idx*>(thrust::raw_pointer_cast(eids);
  auto eids_data = static_cast<Idx*>(thrust::raw_pointer_cast(csr.eids.data()));
  auto typeids_data =
      static_cast<Idx*>(thrust::raw_pointer_cast(csr.rel_type.data()));
  auto grad_sm_first_stage_data = grad_sm_first_stage.Ptr();
  auto grad_a_data = grad_a.Ptr();
  auto grad_t_neighbour_data = grad_t_neighbour.Ptr();
  auto message_data = message.Ptr();
  auto sigmas_data = sigmas.Ptr();

  // print_dims(hidden);
  // print_dims(weight);
  // print_dims(norm);
  // print_dims(ret);
  // Idx num_nodes = ranges->shape[0] - 1;
  // Idx num_edges = eids->shape[0];
  Idx num_nodes = csr.num_rows;
  Idx num_edges = csr.col_idx.size();
  Idx num_heads = grad_sm_first_stage.shape[2];
  Idx feat_dim_per_head = grad_sm_first_stage.shape[3];
  Idx n_rel_types = grad_sm_first_stage.shape[1];
  int nblks = num_nodes;
  int nthrs = num_heads * feat_dim_per_head;
  // auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  HGTBackwardGradientSmFirstPartImpl<Idx, DType><<<nblks, nthrs>>>(
      range_data, ids_data, eids_data, typeids_data, grad_sm_first_stage_data,
      grad_t_neighbour_data, message_data, sigmas_data, num_nodes, num_heads,
      feat_dim_per_head, n_rel_types);
  cuda_err_chk(cudaPeekAtLastError());
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::cout
      << "HGTBackwardGradientSmFirstPartImpl time: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << " ms" << std::endl;

  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t1_kernel2 =
      std::chrono::high_resolution_clock::now();
  HGTBackwardGradientAImpl<Idx, DType><<<nblks, nthrs>>>(
      range_data, ids_data, eids_data, typeids_data, grad_a_data,
      grad_t_neighbour_data, message_data, sigmas_data, num_nodes, num_heads,
      feat_dim_per_head, n_rel_types);
  cuda_err_chk(cudaPeekAtLastError());
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t2_kernel2 =
      std::chrono::high_resolution_clock::now();
  std::cout << "HGTBackwardGradientAImpl time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   t2_kernel2 - t1_kernel2)
                   .count()
            << " ms" << std::endl;

  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t1_kernel3 =
      std::chrono::high_resolution_clock::now();

  HGTBackwardFusedGradientSmFirstPartGradientAImpl<Idx, DType>
      <<<nblks, nthrs>>>(range_data, ids_data, eids_data, typeids_data,
                         grad_a_data, grad_sm_first_stage_data,
                         grad_t_neighbour_data, message_data, sigmas_data,
                         num_nodes, num_heads, feat_dim_per_head, n_rel_types);
  cuda_err_chk(cudaPeekAtLastError());
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t2_kernel3 =
      std::chrono::high_resolution_clock::now();
  std::cout << "HGTBackwardFusedGradientSmFirstPartGradientAImpl time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   t2_kernel3 - t1_kernel3)
                   .count()
            << " ms" << std::endl;
}

template <typename Idx, typename DType>
struct BackwardHGTMessageData {
  Idx num_heads{0};
  Idx message_src_xlen{0};
  Idx* eids;
  DType *grad_message_src{nullptr}, *unnormalized_attn_score{nullptr},
      *edge_softmax_sum{nullptr}, *out{nullptr}, *grad_out{nullptr};
};

// based on _fusedGatBackwardGradFeatSrc, as it is to calculate the gradient of
// message
// TODO: add mu into the term
template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool RelationalFlag, bool ETypeRelPtrFlag>
__device__ __forceinline__ void
_hgtMessageAccumBasedOnOriAttnScoreAndEdgeSoftmaxSumBackwardKernel(
    BackwardHGTMessageData<Idx, DType> gdata, const Idx* row_offsets,
    const Idx* column_indices, const Idx* etypes, int64_t num_rows,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  Idx num_heads = gdata.num_heads;  // originally e_xlen
  Idx hidden_xlen = gdata.message_src_xlen / num_heads;
  for (Idx src_vid = blockIdx.y; src_vid < num_rows; src_vid += gridDim.y) {
    Idx start_off = row_offsets[src_vid];
    Idx end_off = row_offsets[src_vid + 1];
    for (Idx head_idx = blockIdx.x * blockDim.x + threadIdx.x;
         head_idx < num_heads; head_idx += blockDim.x * gridDim.x) {
      for (Idx feat_idx = threadIdx.y; feat_idx < hidden_xlen;
           feat_idx += blockDim.y) {
        DType s = 0.;
        Idx message_src_offset = -1;
        if constexpr (CompactAsOfNodeFlag && !RelationalFlag) {
          // in this case, message_src_offset is the same regardless of which
          // outgoing edge we deal with
          message_src_offset = src_vid * gdata.message_src_xlen +
                               head_idx * hidden_xlen + feat_idx;
        }
        for (Idx e = start_off; e < end_off; ++e) {
          Idx eid = gdata.eids[e];
          Idx dst_vid = column_indices[e];
          Idx dst_vid_relational = -1;
          if constexpr (!CompactAsOfNodeFlag) {
            // in this case, message_src_offset, er_idx and el_idx are related
            // to edge id, regardless of the type of the edge
            message_src_offset = eid * gdata.message_src_xlen +
                                 head_idx * hidden_xlen + feat_idx;
          } else {  // CompactAsOfNodeFlag
            if constexpr (RelationalFlag) {
              // Idx etype = etypes[e];
              Idx etype = -1;
              if constexpr (ETypeRelPtrFlag) {
                etype = binary_search(num_relations, etypes, e);
              } else {  // !ETypeRelPtrFlag
                etype = etypes[e];
              }
              Idx src_vid_relational = find_relational_compact_as_of_node_index(
                  etype, src_vid, unique_srcs_and_dests_rel_ptr,
                  unique_srcs_and_dests_node_indices);
              message_src_offset = src_vid_relational * gdata.message_src_xlen +
                                   head_idx * hidden_xlen + feat_idx;
              dst_vid_relational = find_relational_compact_as_of_node_index(
                  etype, dst_vid, unique_srcs_and_dests_rel_ptr,
                  unique_srcs_and_dests_node_indices);
            }
          }
          // TODO: maybe it's better to cache exp/sum to reduce mem traffic as
          // well as redundant computation?
          Idx sum_vid = dst_vid;
          if constexpr (RelationalFlag && CompactAsOfNodeFlag) {
            sum_vid = dst_vid_relational;
          }
          if constexpr (!CompactAsOfNodeFlag || RelationalFlag) {
            atomicAdd(
                gdata.grad_message_src + message_src_offset,
                gdata.unnormalized_attn_score[eid * num_heads + head_idx] /
                    gdata.edge_softmax_sum[sum_vid * num_heads + head_idx] *
                    gdata.grad_out[dst_vid * gdata.message_src_xlen +
                                   head_idx * hidden_xlen + feat_idx]);
          } else {  // CompactAsOfNodeFlag && !RelationalFlag
            // exp scheme (both eid and head_idx) could be used for attn_score
            // message_src's could be used for message_src
            s += gdata.unnormalized_attn_score[eid * num_heads + head_idx] /
                 gdata.edge_softmax_sum[sum_vid * num_heads + head_idx] *
                 gdata.grad_out[dst_vid * gdata.message_src_xlen +
                                head_idx * hidden_xlen + feat_idx];
          }
        }
        if constexpr (CompactAsOfNodeFlag && !RelationalFlag) {
          gdata.grad_message_src[message_src_offset] = s;
        }
      }
    }
  }
}

template <typename Idx, typename DType>
struct BackwardHGTAttnScoreData {
  Idx num_heads{0};
  Idx message_src_xlen{0};
  Idx* eids;
  DType *grad_message_src{nullptr}, message_src{nullptr},
      *unnormalized_attn_score{nullptr}, *edge_softmax_sum{nullptr},
      *out{nullptr}, *grad_out{nullptr};
};

// S_j = expf(z_j) / sum_k expf(z_k)
// deltaz_edge=S_edge*deltaout_dst^T*message_edge - S_edge * deltaout_dst^T *
// out_dst
// TODO: add mu into the term
// based on _fusedGatBackwardGradElEr, as it is to calculate gradient of
// attention
template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool RelationalFlag, bool ETypeRelPtrFlag>
__device__ __forceinline__ void _hgtEdgeSoftmaxAccumStageOnlyBackwardKernel(
    BackwardHGTAttnScoreData<Idx, DType> gdata, const Idx* row_offsets,
    const Idx* column_indices, const Idx* etypes, int64_t num_rows,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  if constexpr (!CompactAsOfNodeFlag) {
    CONSTEXPR_FALSE_CLAUSE_UNREACHABLE(CompactAsOfNodeFlag,
                                       "not implemented yet");
  }
  Idx num_heads = gdata.num_heads;  // originally e_xlen
  Idx hidden_xlen = gdata.message_src_xlen / num_heads;
  for (Idx src_vid = blockIdx.y; src_vid < num_rows; src_vid += gridDim.y) {
    Idx start_off = row_offsets[src_vid];
    Idx end_off = row_offsets[src_vid + 1];
    for (Idx head_idx = blockIdx.x * blockDim.x + threadIdx.x;
         head_idx < num_heads; head_idx += blockDim.x * gridDim.x) {
      for (Idx feat_idx = threadIdx.y; feat_idx < hidden_xlen;
           feat_idx += blockDim.y) {
        DType s = 0.;
        Idx message_src_offset = -1;
        Idx message_src_idx = -1;
        if constexpr (CompactAsOfNodeFlag && !RelationalFlag) {
          // in this case, message_src_offset is the same regardless of which
          // outgoing edge we deal with
          message_src_offset = src_vid * gdata.message_src_xlen +
                               head_idx * hidden_xlen + feat_idx;
          message_src_idx =
              (src_vid * num_heads + head_idx) * hidden_xlen + feat_idx;
        }
        for (Idx e = start_off; e < end_off; ++e) {
          Idx edge_offset = gdata.eids[e] * num_heads + head_idx;
          Idx eid = gdata.eids[e];
          Idx dst_vid = column_indices[e];
          Idx edge_softmax_sum_idx = -1;
          Idx dst_vid_relational = -1;
          if constexpr (!CompactAsOfNodeFlag) {
            // in this case, message_src_offset, edge_softmax_sum_idx and
            // message_src_idx are related to edge id, regardless of the type of
            // the edge
            message_src_offset = eid * gdata.message_src_xlen +
                                 head_idx * hidden_xlen + feat_idx;
            edge_softmax_sum_idx = eid * num_heads + head_idx;
            message_src_idx =
                (eid * num_heads + head_idx) * hidden_xlen + feat_idx;
          } else {  // CompactAsOfNodeFlag
            if constexpr (!RelationalFlag) {
              edge_softmax_sum_idx = dst_vid * num_heads + head_idx;
            } else {
              // in this case, edge_softmax_sum_idx (sum's index) is related to
              // (relation, unique node index) message_src_idx is related to
              // (relation, unique node index) message_src_offset is related to
              // (relation, unique node index) Idx etype = etypes[e];
              Idx etype = -1;
              if constexpr (ETypeRelPtrFlag) {
                etype = binary_search(num_relations, etypes, e);
              } else {
                etype = etypes[e];
              }
              dst_vid_relational = find_relational_compact_as_of_node_index(
                  etype, dst_vid, unique_srcs_and_dests_rel_ptr,
                  unique_srcs_and_dests_node_indices);
              edge_softmax_sum_idx = dst_vid_relational * num_heads + head_idx;
              Idx src_vid_relational = find_relational_compact_as_of_node_index(
                  etype, src_vid, unique_srcs_and_dests_rel_ptr,
                  unique_srcs_and_dests_node_indices);
              message_src_idx =
                  (src_vid_relational * num_heads + head_idx) * hidden_xlen +
                  feat_idx;
              message_src_offset = src_vid_relational * gdata.message_src_xlen +
                                   head_idx * hidden_xlen + feat_idx;
            }
          }
          Idx dst_out_offset = dst_vid * gdata.message_src_xlen +
                               head_idx * hidden_xlen + feat_idx;
          DType grad_exp = gdata.grad_out[dst_out_offset] *
                           (gdata.message_src[message_src_offset] -
                            gdata.out[dst_out_offset]) /
                           gdata.edge_softmax_sum[edge_softmax_sum_idx];

          DType grad_for_this_feat_idx =
              gdata.grad_out[dst_out_offset] *
              (gdata.message_src[message_src_offset] -
               gdata.out[dst_out_offset]) *
              gdata.attn_score[edge_offset] /
              gdata.edge_softmax_sum[edge_softmax_sum_idx];
          // el idx scheme could be used for message (only item idx, not the
          // feature idx scheme) exp idx scheme could be used for attn_score

          s += grad_for_this_feat_idx;
          if constexpr (!CompactAsOfNodeFlag || RelationalFlag) {
            atomicAdd(gdata.grad_message_src + message_src_idx,
                      grad_for_this_feat_idx);
          }
        }
        if constexpr (CompactAsOfNodeFlag && !RelationalFlag) {
          atomicAdd(gdata.grad_message_src + message_src_idx, s);
        }
      }
    }
  }
}