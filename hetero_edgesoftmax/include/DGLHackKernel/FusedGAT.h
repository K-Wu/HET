#pragma once
#include "DGLHackKernel.h"


template <typename Idx, typename DType>
struct GatFusedData {
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
  // Intermediates
  DType  *sum{nullptr}, *exp{nullptr};
  // Output
  DType *ret{nullptr};
};

template <typename DType>
__device__ DType gatLeakyReluExp(DType val, DType slope) {
    return val > 0 ? exp(val) : exp(slope * val);
}

template <typename Idx, typename DType>
__global__ void gatSumProdZipDivKernel(GatFusedData<Idx, DType> gdata, 
const Idx* row_offsets, const Idx* column_indices, int64_t num_rows) {
    Idx dst_vid = blockIdx.y;
    Idx stride_vid =  gridDim.y;
    Idx stride_head = blockDim.x * gridDim.x;
    Idx e_xlen = gdata.e_xlen;
    Idx hidden_xlen = gdata.feat_src_xlen/e_xlen;
    while (dst_vid < num_rows) {
        Idx start_off = *(row_offsets + dst_vid);
        Idx end_off = *(row_offsets + dst_vid + 1);
        Idx head_idx = blockIdx.x * blockDim.x + threadIdx.x;
        while (head_idx < e_xlen) {
            Idx feat_idx = threadIdx.y;
            while (feat_idx < hidden_xlen) {
                DType s = 0.;
                for (Idx eid=start_off; eid<end_off; eid++) {
                    Idx src_vid = column_indices[eid];
                    //s +=  gdata.exp[gdata.eids[eid] * e_xlen + head_idx] / gdata.sum[dst_vid*e_xlen + head_idx] 
                    s +=  gdata.exp[eid * e_xlen + head_idx] / gdata.sum[dst_vid*e_xlen + head_idx] 
                                        * gdata.feat_src[src_vid*gdata.feat_src_xlen + head_idx*hidden_xlen + feat_idx];
                }
                gdata.ret[dst_vid*gdata.feat_src_xlen + head_idx*hidden_xlen + feat_idx] = s;
                feat_idx += blockDim.y;
            }
            head_idx += stride_head;
        }
        dst_vid += stride_vid;
    }
}

template <typename Idx, typename DType>
__global__ void gatExpLeakyReluSumKernel(GatFusedData<Idx, DType> gdata, 
const Idx* row_offsets, const Idx* column_indices, int64_t num_rows) {
    //extern __shared__ DType er[];
    Idx tx = blockIdx.x * blockDim.x + threadIdx.x;
    Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
    Idx stride_x = blockDim.x * gridDim.x;
    Idx stride_y = blockDim.y * gridDim.y;
    Idx dst_vid = ty;
    Idx e_xlen = gdata.e_xlen;
    while (dst_vid < num_rows) {
        Idx start_off = *(row_offsets + dst_vid);
        Idx end_off = *(row_offsets + dst_vid + 1);
        Idx feat_idx = tx;
        while (feat_idx < e_xlen) {
            // 1. Load dstnation vertex into shared memory
            Idx feat_off_dst = dst_vid * e_xlen + feat_idx;
            //er[threadIdx.x] = gdata.er[feat_off_dst];
            //__syncthreads();
            // 2. Do the computation
            DType sum = 0.;
            for (Idx eid=start_off; eid<end_off; ++eid) {
                Idx src_id = *(column_indices + eid);
                Idx feat_off_src = src_id * e_xlen + feat_idx;
                //DType tmp = gatLeakyReluExp(gdata.el[feat_off_src] + er[threadIdx.x], gdata.leaky_relu_slope);
                DType tmp = gatLeakyReluExp(gdata.el[feat_off_src] + gdata.er[feat_off_dst], gdata.leaky_relu_slope);
                //gdata.exp[Idx(gdata.eids[eid] * e_xlen) + feat_idx] = tmp;
                gdata.exp[Idx(eid * e_xlen) + feat_idx] = tmp;
                
                sum += tmp;
            }
            gdata.sum[Idx(dst_vid*e_xlen) + feat_idx] = sum;
            feat_idx += stride_x;
        }
        dst_vid += stride_y;
    }
}


template </*int XPU, */typename Idx, typename DType>
void FusedGatKernelImpl(
    MyHeteroSeparateCSR<Idx, thrust::device_allocator<Idx>> incsr, // create incsr in the driver logic
    MySimpleNDArray<DType, thrust::device_allocator<DType>> feat_src,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> el,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> er,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> sum,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> exp,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> ret,
    //MySimpleNDArray<Idx, thrust::device_allocator<Idx>> eids,//thrust::sequence<Idx>(eids.data.begin(),eids.data.end(), 0);
    float slope) {
        // As GAT only has 1 type of relationship, we use a specialcase of separateCSR where num releationship is asserted as 1 
        assert(incsr.num_rels==1);
        //static_assert(XPU==kDLGPU);
        const Idx MAX_NBLKS = 65535;
        const Idx MAX_NTHRS = 1024;
        // zero out ret, and packing feat_src, el, er, ret, graph together into one struct using raw float pointers
        // get csr matrix
        GatFusedData<Idx, DType> gdata;
        //int64_t el_xlen =  SeastarComputeXLength(el);
        //int64_t feat_src_xlen =  SeastarComputeXLength(feat_src);
        //int64_t ret_len =  SeastarComputeXLength(ret);

        int64_t el_xlen = el.ComputeXLength();
        int64_t feat_src_xlen = feat_src.ComputeXLength();
        int64_t ret_len =  ret.ComputeXLength();

        gdata.feat_src = feat_src.Ptr();
        gdata.el = el.Ptr();
        gdata.er = er.Ptr();
        gdata.sum = sum.Ptr();
        gdata.exp = exp.Ptr();
        gdata.ret = ret.Ptr();
        gdata.leaky_relu_slope = slope;
        gdata.n = el.data.size()/el_xlen; 
        gdata.e_xlen = el_xlen;
        gdata.feat_src_xlen =  feat_src_xlen;
        gdata.feat_src_hidden = feat_src_xlen/el_xlen;
        gdata.ret_xlen = ret_len;
        //std::vector<IdArray> incsr_elements = graph->GetAdj(0,true, "csr");
        //printf("!!!!!%d, %d, %d\n",graph->NumVertices(0), graph->NumVertexTypes(), graph->NumEdgeTypes());
        //aten::CSRMatrix incsr(graph->NumVertices(0), graph->NumVertices(0), incsr_elements[0], incsr_elements[1], incsr_elements[2]);

        //gdata.eids = incsr.data.Ptr<Idx>(); 
        //gdata.eids = eids.Ptr();
        gdata.eids = static_cast<Idx*>(thrust::raw_pointer_cast(incsr.eids.data()));

        // write a device function and call it from here
        //LOG(INFO) << "Within Fused Gat Kernel Impl." << "feat_src_dim:" << feat_src.GetSize()/sizeof(DType)/feat_src_xlen << "*" << feat_src_xlen 
        //    <<" el_dim:" << el.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen  << " ret_dim:" << ret.GetSize()/sizeof(DType)/ret_len <<"*" << ret_len
        //    <<" sum_dim:" << sum.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen
        //    <<" exp_dim:" << exp.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen
        //    << " graph csr row_offset length:" <<csr.row_offsets.length << " graph csr column indices length:" << csr.column_indices.length;

        // Configure kernel launch parameters.
        //auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
        int nthrs_x = 32;
        int nthrs_y = 1;
        int nblks_x = (el_xlen + nthrs_x-1)/(nthrs_x);
        int nblks_y = std::min(gdata.n, MAX_NBLKS);
        const dim3 nblks(nblks_x, nblks_y);
        const dim3 nthrs(nthrs_x, nthrs_y);
        //LOG(INFO) << "kernel1 blk dim:" << nblks_x << "*" <<nblks_y << " thr dim:" <<nthrs_x << "*" << nthrs_y;
        //aten::CSRMatrix incsr = static_pointer_cast<ImmutableGraph*>(graph)->GetInCSR()->ToCSRMatrix();
        //std::vector<IdArray> incsr_elements = graph->GetAdj();
        //aten::CSRMatrix incsr(graph->NumVertices(), graph->NumVertices(), incsr_elements[0], incsr_elements[1], incsr_elements[2]);
        //print_gdata<Idx, DType>(feat_src,el,er,sum,exp,ret,el_xlen, feat_src_xlen, graph->NumVertices(0),incsr_elements[1].NumElements(), incsr_elements[0], incsr_elements[1], incsr_elements[2]);
        //gatExpLeakyReluSumKernel<<<nblks, nthrs, el_xlen*sizeof(DType), thr_entry->stream>>>(gdata, csr);
        cuda_err_chk(cudaDeviceSynchronize());
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        gatExpLeakyReluSumKernel<Idx, DType><<<nblks, nthrs/*, 0, thr_entry->stream*/>>>(gdata, static_cast<Idx*>(thrust::raw_pointer_cast(incsr.row_ptr.data())), static_cast<Idx*>(thrust::raw_pointer_cast(incsr.col_idx.data())), incsr.num_rows );
        //CUDA_KERNEL_CALL(gatExpLeakyReluSumKernel, nblks, nthrs, 0, thr_entry->stream, gdata, incsr.indptr.Ptr<Idx>(), incsr.indices.Ptr<Idx>(), incsr.num_rows );
        //print_gdata<Idx, DType>(feat_src,el,er,sum,exp,ret,el_xlen, feat_src_xlen, graph->NumVertices(0), incsr_elements[1].NumElements(), incsr_elements[0], incsr_elements[1], incsr_elements[2]);
        nthrs_x = FindNumThreads(el_xlen, 64);
        nthrs_y = FindNumThreads(gdata.feat_src_hidden, MAX_NTHRS/nthrs_x);
        nblks_x = 1;
        nblks_y = std::min(gdata.n, MAX_NBLKS);
        const dim3 nthrs2(nthrs_x, nthrs_y);
        const dim3 nblks2(nblks_x, nblks_y);
        //LOG(INFO) << "kernel2 blk dim:" << nblks_x << "*" <<nblks_y << " thr dim:" <<nthrs_x << "*" << nthrs_y;
        gatSumProdZipDivKernel<Idx, DType><<<nblks2, nthrs2/*, 0, thr_entry->stream*/>>>(gdata, static_cast<Idx*>(thrust::raw_pointer_cast(incsr.row_ptr.data())), static_cast<Idx*>(thrust::raw_pointer_cast(incsr.col_idx.data())), incsr.num_rows);
        cuda_err_chk(cudaPeekAtLastError());
        cuda_err_chk(cudaDeviceSynchronize());
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::cout << "FusedGatKernelImpl fused<"<<0<<"> time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms"<<std::endl;

        //LOG(INFO) << "kernel2 blk dim:" << nblks_x << "*" <<nblks_y << " thr dim:" <<nthrs_x << "*" << nthrs_y;
        //    printf("n_rows: %d\n", incsr.num_rows);
        //    printf("e_xlen: %d\n", gdata.e_xlen);
        //    printf("hidden_xlen: %d\n", gdata.feat_src_xlen/gdata.e_xlen);
        //    printf("stride_head: %d\n", nblks_x * nthrs_x);
        //    printf("stride_vid: %d\n", nblks_y);
        //    printf("dst_vid: %d\n", nthrs_y);
        
        //CUDA_KERNEL_CALL(gatSumProdZipDivKernel,nblks2, nthrs2, 0, thr_entry->stream,gdata, incsr.indptr.Ptr<Idx>(), incsr.indices.Ptr<Idx>(), incsr.num_rows);

}