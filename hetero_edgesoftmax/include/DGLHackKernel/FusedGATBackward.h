#pragma once
#include "DGLHackKernel.h"


template <typename Idx, typename DType>
struct BackwardGatFusedData {
  // feat_size size along feature dimension
  Idx feat_src_xlen{0};
  Idx feat_src_hidden{0};
  Idx e_xlen{0};
  Idx ret_xlen{0};
  // num nodes
  Idx n{0};
  Idx *eids;
  DType leaky_relu_slope;
  // Inputs
  DType *feat_src{nullptr}, *el{nullptr}, *er{nullptr};
  DType *sum{nullptr}, *exp{nullptr}, *ret{nullptr};
  // Output
  DType *grad_out{nullptr}, *grad_feat_src{nullptr}, *grad_el{nullptr}, *grad_er{nullptr};
};

template <typename DType>
__device__ DType gradLeaky(DType val, DType slope) {
    return val > 0 ? 1 : slope;
}

template <typename Idx, typename DType>
__global__ void fusedGatBackwardGradFeatSrc(BackwardGatFusedData<Idx, DType> gdata, const Idx* row_offsets, const Idx* column_indices, int64_t num_rows) {
    Idx src_vid = blockIdx.y;
    Idx stride_vid = gridDim.y;
    Idx e_xlen = gdata.e_xlen;
    Idx stride_head = blockDim.x * gridDim.x;
    Idx hidden_xlen = gdata.feat_src_xlen/e_xlen;
    while (src_vid < num_rows) {
        Idx start_off = row_offsets[src_vid];
        Idx end_off = row_offsets[src_vid+1];
        Idx head_idx = blockIdx.x * blockDim.x  + threadIdx.x;
        while (head_idx < e_xlen) {
            Idx feat_idx = threadIdx.y;
            while (feat_idx < hidden_xlen) {
                DType s = 0.;
                for (Idx e=start_off; e<end_off; ++e) {
                    Idx eid = gdata.eids[e];
                    Idx dst_id = column_indices[e];
                    // TODO: maybe it's better to cache exp/sum to reduce mem traffic as well as redundant computation?
                    s += gdata.exp[eid*e_xlen + head_idx] / gdata.sum[dst_id*e_xlen + head_idx]
                        * gdata.grad_out[dst_id*gdata.feat_src_xlen + head_idx*hidden_xlen + feat_idx];
                }
                gdata.grad_feat_src[src_vid*gdata.feat_src_xlen + head_idx*hidden_xlen + feat_idx] = s;
                feat_idx += blockDim.y;
            }
            head_idx += stride_head;
        }
        src_vid += stride_vid;
    }
}

template <typename Idx, typename DType>
__global__ void fusedGatBackwardGradElEr(BackwardGatFusedData<Idx, DType> gdata, const Idx* row_offsets, const Idx* column_indices, int64_t num_rows) {
    Idx src_vid = blockIdx.y;
    Idx stride_vid = gridDim.y;
    Idx e_xlen = gdata.e_xlen;
    Idx stride_head = blockDim.x * gridDim.x;
    Idx hidden_xlen = gdata.feat_src_xlen/e_xlen;
    while (src_vid < num_rows) {
        Idx start_off = row_offsets[src_vid];
        Idx end_off = row_offsets[src_vid+1];
        Idx head_idx = blockIdx.x * blockDim.x  + threadIdx.x;
        while (head_idx < e_xlen) {
            Idx feat_idx = threadIdx.y;
            while (feat_idx < hidden_xlen) {
                DType s = 0.;
                Idx feat_src_offset = src_vid*gdata.feat_src_xlen + head_idx*hidden_xlen + feat_idx;
                Idx src_node_feat_offset = src_vid*e_xlen + head_idx;
                for (Idx e=start_off; e<end_off; ++e) {
                    Idx edge_offset = gdata.eids[e] * e_xlen + head_idx;
                    Idx dst_vid = column_indices[e];
                    Idx dst_node_feat_offset = dst_vid*e_xlen + head_idx;
                    Idx dst_out_offset = dst_vid*gdata.feat_src_xlen + head_idx*hidden_xlen + feat_idx;
                    DType grad_exp = gdata.grad_out[dst_out_offset]* (gdata.feat_src[feat_src_offset]- gdata.ret[dst_out_offset])/gdata.sum[dst_node_feat_offset] ;
                    DType tmp_sum = gdata.el[src_node_feat_offset] + gdata.er[dst_node_feat_offset];
                    DType tmp2 = grad_exp * gdata.exp[edge_offset] * gradLeaky(tmp_sum, gdata.leaky_relu_slope);
                    s += tmp2;
                    atomicAdd(gdata.grad_er + dst_node_feat_offset, tmp2);
                }
                atomicAdd(gdata.grad_el + src_node_feat_offset , s);
                feat_idx += blockDim.y;
            }
            head_idx += stride_head;
        }
        src_vid += stride_vid;
    }
}

template </*int XPU, */typename Idx, typename DType>
void BackwardFusedGatKernelImpl(
    //const CSRWrapper& graph, //TODO: remove CSRWrapper
    // create CSR in driver code
    MyHeteroSeparateCSR<Idx, thrust::device_allocator<Idx>> outcsr,
    MySimpleNDArray<DType,thrust::device_allocator<DType>> feat_src,
    MySimpleNDArray<DType,thrust::device_allocator<DType>> el,
    MySimpleNDArray<DType,thrust::device_allocator<DType>> er,
    MySimpleNDArray<DType,thrust::device_allocator<DType>> sum,
    MySimpleNDArray<DType,thrust::device_allocator<DType>> exp,
    MySimpleNDArray<DType,thrust::device_allocator<DType>> ret,
    MySimpleNDArray<DType,thrust::device_allocator<DType>> grad_out,
    MySimpleNDArray<DType,thrust::device_allocator<DType>> grad_feat_src,
    MySimpleNDArray<DType,thrust::device_allocator<DType>> grad_el,
    MySimpleNDArray<DType,thrust::device_allocator<DType>> grad_er,
    MySimpleNDArray<Idx, thrust::device_allocator<Idx>> eids, //thrust::sequence<Idx>(eids.data.begin(),eids.data.end(), 0); TODO: check if it needs a different eid
    float slope) {
        // As GAT only has 1 type of relationship, we use a specialcase of separateCSR where num releationship is asserted as 1 
        assert(outcsr.num_rels==1);
        //typedef int32_t Idx;
        //typedef float DType;
        const Idx MAX_NBLKS = 65535;
        const Idx MAX_NTHRS = 1024;
        // zero out ret, and packing feat_src, el, er, ret, graph together into one struct using raw float pointers
        // get csr matrix
        BackwardGatFusedData<Idx, DType> gdata;
        int64_t el_xlen =  el.ComputeXLength();
        int64_t feat_src_xlen =  feat_src.ComputeXLength();
        gdata.feat_src = feat_src.Ptr<DType>();
        gdata.el = el.Ptr<DType>();
        gdata.er = er.Ptr<DType>();
        gdata.sum = sum.Ptr<DType>();
        gdata.exp = exp.Ptr<DType>();
        gdata.ret = ret.Ptr<DType>();
        gdata.grad_out= grad_out.Ptr<DType>();
        gdata.grad_feat_src = grad_feat_src.Ptr<DType>();
        gdata.grad_el = grad_el.Ptr<DType>();
        gdata.grad_er = grad_er.Ptr<DType>();
        gdata.leaky_relu_slope = slope;
        //gdata.n = el.GetSize()/sizeof(DType)/el_xlen; 
        gdata.n = el.data.size()/el_xlen;
        gdata.e_xlen = el_xlen;
        gdata.feat_src_xlen =  feat_src_xlen;
        gdata.feat_src_hidden = feat_src_xlen/el_xlen;
        //auto outcsr = graph.GetOutCSRMatrix();
        //minigun::Csr<Idx> ocsr = utils::CreateCsr<Idx>(outcsr.indptr, outcsr.indices);
        gdata.eids = eids.Ptr<Idx>();//static_cast<Idx*>(outcsr.data->data);
        // write a device function and call it from here
        //LOG(INFO) << "Within Fused Gat Kernel Impl." << "feat_src_dim:" << feat_src.GetSize()/sizeof(DType)/feat_src_xlen << "*" << feat_src_xlen 
        //    <<" el_dim:" << el.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen  << " ret_dim:" << ret.GetSize()/sizeof(DType)/ret_len <<"*" << ret_len
        //    <<" sum_dim:" << sum.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen
        //    <<" exp_dim:" << exp.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen
        //    << " graph csr row_offset length:" <<csr.row_offsets.length << " graph csr column indices length:" << csr.column_indices.length;
        //print_gdata<Idx, DType>(feat_src,el,er,sum,exp,grad_out,ocsr,el_xlen, feat_src_xlen);
        // Configure kernel launch parameters.
        //auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
        int nthrs_x = FindNumThreads(el_xlen, 64);
        int nthrs_y = FindNumThreads(gdata.feat_src_hidden, MAX_NTHRS/nthrs_x);
        int nblks_x = 1;
        int nblks_y = std::min(gdata.n, MAX_NBLKS);
        const dim3 nthrs(nthrs_x, nthrs_y);
        const dim3 nblks(nblks_x, nblks_y);
        //LOG(INFO) << "GradFeatSrc kernel blk dim:" << nblks_x << "*" <<nblks_y << " thr dim:" <<nthrs_x << "*" << nthrs_y;
        fusedGatBackwardGradFeatSrc<<<nblks, nthrs/*, 0, thr_entry->stream*/>>>(gdata, static_cast<Idx*>(thrust::raw_pointer_cast(outcsr.row_ptr.data())), static_cast<Idx*>(thrust::raw_pointer_cast(outcsr.col_idx.data())), outcsr.num_rows);
        //const dim3 nthrs3(nthrs_y, nthrs_x);
        //fusedGatBackwardGradElEr4<<<nblks, nthrs3, 0, thr_entry->stream>>>(gdata, ocsr);
        fusedGatBackwardGradElEr<Idx, DType><<<nblks, nthrs/*, 0, thr_entry->stream*/>>>(gdata, static_cast<Idx*>(thrust::raw_pointer_cast(outcsr.row_ptr.data())), static_cast<Idx*>(thrust::raw_pointer_cast(outcsr.col_idx.data())), outcsr.num_rows);
}