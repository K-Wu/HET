#pragma once
#include "DGLHackKernel.h"



// bgs:
template<typename Idx, typename DType>
__global__ void RgcnLayer1KernelImpl(const Idx* ranges, 
  const Idx* src_ids, 
  const Idx* eids, 
  const Idx* types, 
  const DType* hidden, 
  const DType* weight, 
  const DType* norm, 
  DType* ret, 
  Idx num_nodes, 
  Idx feat_len_y, 
  Idx feat_len_x, 
  Idx ntypes) {
    if (blockIdx.x < num_nodes) {
        Idx beg = __ldg(ranges + blockIdx.x);
        Idx end = __ldg(ranges + blockIdx.x + 1);
        Idx tx = threadIdx.x;
        Idx ty = threadIdx.x / feat_len_x;
        Idx th = threadIdx.x % feat_len_x;
        DType agg_val = 0.; 
        DType w = 0.;
        Idx cur_type_id = -1;
        for(;beg<end;beg++) {
            Idx src_id = __ldg(src_ids + beg);
            Idx eid = __ldg(eids + beg);
            Idx type_id = __ldg(types + beg);
            if (type_id != cur_type_id) {
                w = __ldg(weight + type_id*feat_len_y*feat_len_x + tx);
            }
            DType h = __ldg(hidden + src_id*feat_len_y + ty);
            DType n = __ldg(norm + eid);
            agg_val += h * w * n;
        }
        atomicAdd(ret + blockIdx.x*feat_len_x + th, agg_val);
    }
}

template </*int XPU, */typename Idx, typename DType>
void RgcnLayer1Impl(
    //GraphRef graph,
    MyHeteroIntegratedCSR<int32_t, thrust::device_allocator<int32_t>> csr,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> hidden,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> weight,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> norm,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> ret,
    MySimpleNDArray<Idx, thrust::device_allocator<Idx>> eids){
        //LOG(INFO) << "Calling implementation of rgn layer 1 forward";
        //assert(csr.IsSortedByEdgeType_CPU());
        //typedef int32_t Idx;
        //typedef float DType;
        //auto csr = graph->GetCsrSortedByEdgeType(false); //TODO: implement this API
        //auto ranges = csr[0];
        //auto ids = csr[1];
        //auto eids = csr[2];
        //auto type_ids = csr[3];
        auto range_data = static_cast<Idx*>(thrust::raw_pointer_cast(csr.row_ptr.data()));
        auto ids_data = static_cast<Idx*>(thrust::raw_pointer_cast(csr.col_idx.data()));
        auto eids_data = eids.Ptr<Idx>();
        auto typeids_data = static_cast<Idx*>(thrust::raw_pointer_cast(csr.rel_type.data()));
        auto hidden_data = hidden.Ptr<DType>();
        auto weight_data = weight.Ptr<DType>();
        auto norm_data = norm.Ptr<DType>();
        auto ret_data = ret.Ptr<DType>();
        //print_dims(hidden);
        //print_dims(weight);
        //print_dims(norm);
        //print_dims(ret);
        // Idx num_nodes = ranges->shape[0] - 1;
        // Idx num_edges = eids->shape[0];
        Idx num_nodes = csr.num_rows;
        Idx num_edges = csr.col_idx.size();
        Idx ntypes = weight.shape[0];
        Idx feat_len_y = weight.shape[1];
        Idx feat_len_x = weight.shape[2];
        int nblks = num_nodes;
        int nthrs = feat_len_y * feat_len_x;
        //auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
        RgcnLayer1KernelImpl<Idx, DType><<<nblks, nthrs/*, 0, thr_entry->stream*/>>>
            (range_data, ids_data, eids_data, typeids_data, hidden_data, weight_data, norm_data, ret_data, num_nodes, feat_len_y, feat_len_x, ntypes);
    }