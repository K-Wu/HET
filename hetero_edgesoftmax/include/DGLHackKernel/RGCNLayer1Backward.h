#pragma once
#include "DGLHackKernel.h"


template<typename Idx, typename DType>
__global__ void RgcnLayer1BackwardKernelImpl(Idx* ranges, 
  Idx* dst_ids, 
  Idx* eids, 
  Idx* types, 
  DType* hidden, 
  DType* weight, 
  DType* norm, 
  DType* grad_out, 
  DType* grad_hidden, 
  DType* grad_weight, 
  Idx num_nodes, 
  Idx feat_len_y, 
  Idx feat_len_x, 
  Idx ntypes) {
    if (blockIdx.x < num_nodes) {
        Idx beg = __ldg(ranges + blockIdx.x);
        Idx end = __ldg(ranges + blockIdx.x + 1);
        Idx tx = threadIdx.x;
        for (;tx<feat_len_x * feat_len_y; tx += blockDim.x) {
            Idx ty = tx / feat_len_x;
            Idx th = tx % feat_len_x;
            DType h = __ldg(hidden + blockIdx.x*feat_len_y + ty);
            DType agg = 0.;
            for(;beg<end;beg++) {
                Idx dst_id = __ldg(dst_ids + beg);
                Idx eid = __ldg(eids + beg);
                Idx type_id = __ldg(types + beg);
                DType g = __ldg(grad_out + dst_id * feat_len_x + th);
                DType w = __ldg(weight + type_id*feat_len_y*feat_len_x + tx);
                DType n = __ldg(norm + eid);
                agg += g*w*n;
                atomicAdd(grad_weight + type_id*feat_len_y*feat_len_x + tx, g*h*n);
            }
            atomicAdd(grad_hidden + blockIdx.x*feat_len_y + ty, agg);
        }
    }
}

template </*int XPU, */typename Idx, typename DType>
void RgcnLayer1BackwardImpl(
    //GraphRef graph,
    MyHeteroIntegratedCSR<int32_t, thrust::device_allocator<int32_t>> transposed_csr,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> hidden,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> weight,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> norm,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_out,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_hidden,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_weight,
    MySimpleNDArray<Idx, thrust::device_allocator<Idx>> eids){
        //assert(csr.IsSortedByEdgeType_CPU());
        //cudaDeviceSynchronize();
        //auto t1 = std::chrono::steady_clock::now();
        //typedef int32_t Idx;
        //typedef float DType;
        //auto csr = graph->GetCsrSortedByEdgeType(true);
        //auto ranges = csr[0];
        //auto ids = csr[1];
        //auto eids = csr[2];
        //auto type_ids = csr[3];
        auto range_data = static_cast<Idx*>(thrust::raw_pointer_cast(transposed_csr.row_ptr.data()));
        auto ids_data = static_cast<Idx*>(thrust::raw_pointer_cast(transposed_csr.col_idx.data()));
        auto eids_data = eids.Ptr<Idx>();
        auto typeids_data = static_cast<Idx*>(thrust::raw_pointer_cast(transposed_csr.rel_type.data()));
        auto hidden_data = hidden.Ptr<DType>();
        auto weight_data = weight.Ptr<DType>();
        auto norm_data = norm.Ptr<DType>();
        auto grad_out_data = grad_out.Ptr<DType>();
        auto grad_hidden_data = grad_hidden.Ptr<DType>();
        auto grad_weight_data = grad_weight.Ptr<DType>();
        //print_dims(hidden);
        //print_dims(weight);
        //print_dims(norm);
        //print_dims(grad_out);
        //print_dims(grad_hidden);
        //print_dims(grad_weight);
        //Idx num_nodes = ranges->shape[0] - 1;
        //Idx num_edges = eids->shape[0];
        //Idx ntypes = weight->shape[0];
        //Idx feat_len_y = weight->shape[1];
        //Idx feat_len_x = weight->shape[2];
        Idx num_nodes = transposed_csr.num_rows;
        Idx num_edges = transposed_csr.col_idx.size();
        Idx ntypes = weight.shape[0];
        Idx feat_len_y = weight.shape[1];
        Idx feat_len_x = weight.shape[2];
        int nblks = num_nodes;
        int nthrs = feat_len_y * feat_len_x;
        //auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
        
        RgcnLayer1BackwardKernelImpl<Idx, DType>
            <<<nblks, nthrs/*, 0, thr_entry->stream*/>>>
            (range_data, ids_data, eids_data, typeids_data,
             hidden_data, weight_data, norm_data, grad_out_data, grad_hidden_data, grad_weight_data,
             num_nodes, feat_len_y, feat_len_x, ntypes);
        //cudaDeviceSynchronize();
        //auto t2 = std::chrono::steady_clock::now();
        //LOG(INFO) << "layer 1 backward kernel takes:" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 -t1).count()/1000.0 << " s";
    }