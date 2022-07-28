#include "DGLHackKernel.h"

template<typename Idx, typename DType>
__global__ void HGTBackwardFusedGradientSmFirstPartGradientAImpl(Idx* ranges, 
  Idx* dst_ids, 
  Idx* eids, 
  Idx* types, 
  DType* grad_a, // |E| * N_HEADS
  DType* grad_sm_first_stage,//|V| * N_REL_TYPES * N_HEADS * DIM_PER_HEAD
  DType* grad_t_neighbour,//|V| * N_HEADS * DIM_PER_HEAD
  DType* message,//|E| * N_HEADS * DIM_PER_HEAD
  DType* sigmas,//|E| * N_HEADS
  Idx num_nodes, 
  Idx num_heads, 
  Idx feat_dim_per_head, 
  Idx n_rel_types) {
    // delta a = delta t_neighbour^(l+1) * sigma^-1 * m^T
    int lane_idx = threadIdx.x%32;
    if (blockIdx.x < num_nodes) {
        Idx beg = __ldg(ranges + blockIdx.x);
        Idx end = __ldg(ranges + blockIdx.x + 1);
        Idx tx = threadIdx.x;
        for (;tx<feat_dim_per_head * num_heads; tx += blockDim.x) {
            // each block deald with one source node, and each thread deals with one element along the feature dimension
            
            Idx tidx_head = tx / feat_dim_per_head;
            Idx tidx_ele_in_head = tx % feat_dim_per_head;
            // load delta t neighbor 
            DType delta_t_neighbour_ele = __ldg(grad_t_neighbour + blockIdx.x*num_heads*feat_dim_per_head + tidx_head*feat_dim_per_head+tidx_ele_in_head);
            DType agg = 0.;
            for(;beg<end;beg++) {
                // dealing with each edge
                // broadcast sigma. each thread load one element from m.
                Idx dst_id = __ldg(dst_ids + beg);
                Idx eid = __ldg(eids + beg);
                Idx type_id = __ldg(types + beg);
                DType msg_ele = __ldg(message + eid * num_heads * feat_dim_per_head + tidx_head*feat_dim_per_head+tidx_ele_in_head);
                DType sigma = __ldg(sigmas + eid * num_heads + tidx_head);
                agg += sigma*msg_ele*delta_t_neighbour_ele;
                DType inner_agg = sigma*(1-sigma) * msg_ele * delta_t_neighbour_ele;
                //agg += g*w*n;
                #pragma unroll
                for (int i_reduction = 16; i_reduction > 0; i_reduction = i_reduction / 2)
                {
                    inner_agg += __shfl_down_sync(-1, inner_agg, i_reduction);
                }
                
                if(lane_idx==0){
                    atomicAdd(grad_a + eid*num_heads+tidx_head, inner_agg);
                }
            }
            atomicAdd(grad_sm_first_stage + num_nodes * n_rel_types * num_heads* feat_dim_per_head + blockIdx.x * num_heads* feat_dim_per_head + tx, agg);
        }
    }
}

template<typename Idx, typename DType>
__global__ void HGTBackwardGradientAImpl(Idx* ranges, 
  Idx* dst_ids, 
  Idx* eids, 
  Idx* types, 
  DType* grad_a, // |E| * N_HEADS
  DType* grad_t_neighbour,//|V| * N_HEADS * DIM_PER_HEAD
  DType* message,//|E| * N_HEADS * DIM_PER_HEAD
  DType* sigmas,//|E| * N_HEADS
  Idx num_nodes, 
  Idx num_heads, 
  Idx feat_dim_per_head, 
  Idx n_rel_types) {
    // delta a = delta t_neighbour^(l+1) * sigma^-1 * m^T
    int lane_idx = threadIdx.x%32;
    if (blockIdx.x < num_nodes) {
        Idx beg = __ldg(ranges + blockIdx.x);
        Idx end = __ldg(ranges + blockIdx.x + 1);
        Idx tx = threadIdx.x;
        for (;tx<feat_dim_per_head * num_heads; tx += blockDim.x) {
            // each block deald with one source node, and each thread deals with one element along the feature dimension
            
            Idx tidx_head = tx / feat_dim_per_head;
            Idx tidx_ele_in_head = tx % feat_dim_per_head;
            // load delta t neighbor 
            DType delta_t_neighbour_ele = __ldg(grad_t_neighbour + blockIdx.x*num_heads*feat_dim_per_head + tidx_head*feat_dim_per_head+tidx_ele_in_head);
            
            for(;beg<end;beg++) {
                // dealing with each edge
                // broadcast sigma. each thread load one element from m.
                Idx dst_id = __ldg(dst_ids + beg);
                Idx eid = __ldg(eids + beg);
                Idx type_id = __ldg(types + beg);
                DType msg_ele = __ldg(message + eid * num_heads * feat_dim_per_head + tidx_head*feat_dim_per_head+tidx_ele_in_head);
                DType sigma = __ldg(sigmas + eid * num_heads + tidx_head);
                DType inner_agg = sigma*(1-sigma) * msg_ele * delta_t_neighbour_ele;
                //agg += g*w*n;
                #pragma unroll
                for (int i_reduction = 16; i_reduction > 0; i_reduction = i_reduction / 2)
                {
                    inner_agg += __shfl_down_sync(-1, inner_agg, i_reduction);
                }
                
                if(lane_idx==0){
                    atomicAdd(grad_a + eid*num_heads+tidx_head, inner_agg);
                }
            }
            
        }
    }
}

template<typename Idx, typename DType>
__global__ void HGTBackwardGradientSmFirstPartImpl(Idx* ranges, 
  Idx* dst_ids, 
  Idx* eids, 
  Idx* types, 
  DType* grad_sm_first_stage,//|V| * N_REL_TYPES * N_HEADS * DIM_PER_HEAD
  DType* grad_t_neighbour, //|V| * N_HEADS * DIM_PER_HEAD
  DType* message, //|E| * N_HEADS * DIM_PER_HEAD
  DType* sigmas,//|E| * N_HEADS
  Idx num_nodes, 
  Idx num_heads, 
  Idx feat_dim_per_head, 
  Idx n_rel_types) {
    // delta Sm = \Sum_outgoing (m * delta t_neighbour^(l+1) * sigma) 
    // We need to store one delta Sm for each relationship type
    if (blockIdx.x < num_nodes) {
        Idx beg = __ldg(ranges + blockIdx.x);
        Idx end = __ldg(ranges + blockIdx.x + 1);
        Idx tx = threadIdx.x;
        for (;tx<feat_dim_per_head * num_heads; tx += blockDim.x) {
            // each block deald with one source node, and each thread deals with one element along the feature dimension
            
            Idx tidx_head = tx / feat_dim_per_head;
            Idx tidx_ele_in_head = tx % feat_dim_per_head;
            // load delta t neighbor 
            DType delta_t_neighbour_ele = __ldg(grad_t_neighbour + blockIdx.x*num_heads*feat_dim_per_head + tidx_head*feat_dim_per_head+tidx_ele_in_head);
            DType agg = 0.;
            for(;beg<end;beg++) {
                // dealing with each edge
                //  broadcast sigma
                Idx dst_id = __ldg(dst_ids + beg);
                Idx eid = __ldg(eids + beg);
                Idx type_id = __ldg(types + beg);
                DType msg_ele = __ldg(message + eid * num_heads * feat_dim_per_head + tidx_head*feat_dim_per_head+tidx_ele_in_head);
                DType sigma = __ldg(sigmas + eid * num_heads + tidx_head);
                agg += sigma*msg_ele*delta_t_neighbour_ele;
                //atomicAdd();
            }
            atomicAdd(grad_sm_first_stage + num_nodes * n_rel_types * num_heads* feat_dim_per_head + blockIdx.x * num_heads* feat_dim_per_head + tx, agg);
        }
    }
}



template </*int XPU, */typename Idx, typename DType>
void HGTBackPropGradientSMAFusion(
    //GraphRef graph,
    MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>> csr,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_sm_first_stage,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_a, 
    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_t_neighbour,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> message,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> sigmas){
        //LOG(INFO) << "Calling implementation of rgn layer 1 forward";
        //assert(csr.IsSortedByEdgeType_CPU());
        //typedef int32_t Idx;
        //typedef float DType;
        //auto csr = graph->GetCsrSortedByEdgeType(false); 
        //auto ranges = csr[0];
        //auto ids = csr[1];
        //auto eids = csr[2];
        //auto type_ids = csr[3];
        auto range_data = static_cast<Idx*>(thrust::raw_pointer_cast(csr.row_ptr.data()));
        auto ids_data = static_cast<Idx*>(thrust::raw_pointer_cast(csr.col_idx.data()));
        //auto eids_data = static_cast<Idx*>(thrust::raw_pointer_cast(eids);
        auto eids_data = static_cast<Idx*>(thrust::raw_pointer_cast(csr.eids.data()));
        auto typeids_data = static_cast<Idx*>(thrust::raw_pointer_cast(csr.rel_type.data()));
        auto grad_sm_first_stage_data = grad_sm_first_stage.Ptr();
        auto grad_a_data = grad_a.Ptr();
        auto grad_t_neighbour_data = grad_t_neighbour.Ptr();
        auto message_data = message.Ptr();
        auto sigmas_data = sigmas.Ptr();

        //print_dims(hidden);
        //print_dims(weight);
        //print_dims(norm);
        //print_dims(ret);
        // Idx num_nodes = ranges->shape[0] - 1;
        // Idx num_edges = eids->shape[0];
        Idx num_nodes = csr.num_rows;
        Idx num_edges = csr.col_idx.size();
        Idx num_heads = grad_sm_first_stage.shape[2];
        Idx feat_dim_per_head = grad_sm_first_stage.shape[3];
        Idx n_rel_types = grad_sm_first_stage.shape[1];
        int nblks = num_nodes;
        int nthrs = num_heads * feat_dim_per_head;
        //auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
        cuda_err_chk(cudaDeviceSynchronize());
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        HGTBackwardGradientSmFirstPartImpl<Idx, DType><<<nblks, nthrs/*, 0, thr_entry->stream*/>>>
            (range_data, ids_data, eids_data, typeids_data, grad_sm_first_stage_data, grad_t_neighbour_data, message_data, sigmas_data,num_nodes,  num_heads,   feat_dim_per_head, n_rel_types);
        cuda_err_chk(cudaPeekAtLastError());
        cuda_err_chk(cudaDeviceSynchronize());
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::cout << "HGTBackwardGradientSmFirstPartImpl time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms"<<std::endl;


        cuda_err_chk(cudaDeviceSynchronize());
        std::chrono::high_resolution_clock::time_point t1_kernel2 = std::chrono::high_resolution_clock::now();
        HGTBackwardGradientAImpl<Idx, DType><<<nblks, nthrs/*, 0, thr_entry->stream*/>>>
            (range_data, ids_data, eids_data, typeids_data,grad_a_data, grad_t_neighbour_data, message_data, sigmas_data,num_nodes,  num_heads,   feat_dim_per_head, n_rel_types);
        cuda_err_chk(cudaPeekAtLastError());
        cuda_err_chk(cudaDeviceSynchronize());
        std::chrono::high_resolution_clock::time_point t2_kernel2 = std::chrono::high_resolution_clock::now();
        std::cout << "HGTBackwardGradientAImpl time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2_kernel2 - t1_kernel2).count() << " ms"<<std::endl;
    }