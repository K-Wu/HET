#pragma once
#include "DGLHackKernel.h"


int FusetGATProfiling_main(cusp::csr_matrix<int, int, cusp::host_memory> graph, int64_t num_heads, int64_t num_hidden){
    typedef int32_t Idx;
    typedef float DType;
  

    MyHeteroSeparateCSR<Idx, std::allocator<Idx>> incsr_h(std::vector<cusp::csr_matrix<int, int, cusp::host_memory>>{graph});
    MyHeteroSeparateCSR<Idx, std::allocator<Idx>> outcsr_h(incsr_h);
    MySimpleNDArray<Idx, std::allocator<Idx>> eids_h(std::vector<int64_t>{incsr_h.total_num_nnzs});
    thrust::sequence<>(eids_h.data.begin(),eids_h.data.end(), 0);
    MySimpleNDArray<Idx, std::allocator<Idx>> transposed_eids_h(eids_h);
    outcsr_h.Transpose<>(std::optional<std::reference_wrapper<typename thrust::detail::vector_base<Idx, std::allocator<Idx>>>>{transposed_eids_h.data});

    // copy CSR+eid data to device

    MyHeteroSeparateCSR<Idx, thrust::device_allocator<Idx>> incsr(incsr_h);
    MyHeteroSeparateCSR<Idx, thrust::device_allocator<Idx>> outcsr(outcsr_h);
    MySimpleNDArray<Idx, thrust::device_allocator<Idx>> eids(eids_h);
    MySimpleNDArray<Idx, thrust::device_allocator<Idx>> transposed_eids(transposed_eids_h);
    

    MySimpleNDArray<DType, thrust::device_allocator<DType>> feat_src=GenerateRandomNDArray<DType>({incsr.num_rows, num_heads, num_hidden});
    MySimpleNDArray<DType, thrust::device_allocator<DType>> el=GenerateRandomNDArray<DType>({incsr.num_rows, num_heads, 1});
    MySimpleNDArray<DType, thrust::device_allocator<DType>> er=GenerateRandomNDArray<DType>({incsr.num_rows, num_heads, 1});
    MySimpleNDArray<DType, thrust::device_allocator<DType>> sum=GenerateRandomNDArray<DType>({incsr.num_rows, num_heads, 1});
    MySimpleNDArray<DType, thrust::device_allocator<DType>> exp=GenerateRandomNDArray<DType>({incsr.total_num_nnzs, num_heads, 1});
    MySimpleNDArray<DType, thrust::device_allocator<DType>> ret=GenerateRandomNDArray<DType>({incsr.num_rows, num_heads, num_hidden});
    
    

    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_out=GenerateRandomNDArray<DType>({incsr.num_rows, num_heads, num_hidden}); // TODO: verify if the assumption that the shape is the same as ret is correct
    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_feat_src=GenerateRandomNDArray<DType>({incsr.num_rows, num_heads, num_hidden});
    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_el=GenerateRandomNDArray<DType>({incsr.num_rows, num_heads, 1});
    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_er=GenerateRandomNDArray<DType>({incsr.num_rows, num_heads, 1});

    float slope;

    FusedGatKernelImpl<Idx, DType>(incsr, feat_src, el, er, sum, exp, ret, eids, slope);
    // TODO: check if transpsoed eid is needed here
    BackwardFusedGatKernelImpl<Idx, DType>(outcsr, feat_src, el, er, sum, exp, ret, grad_out, grad_feat_src, grad_el, grad_er, transposed_eids, slope);
    return 0;
}
// TODO: implement thrust::vector Transpose(CSRType, eids) with optional eid as output

cusp::csr_matrix<int, int, cusp::host_memory> LoadFB15k237Data(){
    std::vector<unsigned long> srcs_shape;
    std::vector<unsigned long> dsts_shape;
    std::vector<unsigned long> etypes_shape;

    //bool fortran_order;
    std::vector<int> srcs_data;
    std::vector<int> dsts_data;
    std::vector<int> etypes_data;

    int num_nodes = 14541;
    int num_edges = 620232;

    npy::LoadArrayFromNumpy("data/MyHybData/fb15k237.coo.srcs.npy", srcs_shape, srcs_data);
    npy::LoadArrayFromNumpy("data/MyHybData/fb15k237.coo.dsts.npy", dsts_shape, dsts_data);
    npy::LoadArrayFromNumpy("data/MyHybData/fb15k237.coo.etypes.npy", etypes_shape, etypes_data);
    cusp::coo_matrix<int, int, cusp::host_memory> coo_matrix_h(num_nodes, num_nodes, srcs_data.size());
    for (int64_t i = 0; i < srcs_data.size(); i++) {
        coo_matrix_h.row_indices[i] = srcs_data[i];
        coo_matrix_h.column_indices[i] = dsts_data[i];
        coo_matrix_h.values[i] = etypes_data[i];
    }
    return coo_matrix_h;
}

cusp::csr_matrix<int, int, cusp::host_memory> LoadOGBNWikiKG2Data(){
    std::vector<unsigned long> srcs_shape;
    std::vector<unsigned long> dsts_shape;
    std::vector<unsigned long> etypes_shape;

    //bool fortran_order;
    std::vector<int> srcs_data;
    std::vector<int> dsts_data;
    std::vector<int> etypes_data;

    int num_nodes = 2500604;
    int num_edges = 16109182;

    npy::LoadArrayFromNumpy("data/MyHybData/ogbn-wikikg2.coo.srcs.npy", srcs_shape, srcs_data);
    npy::LoadArrayFromNumpy("data/MyHybData/ogbn-wikikg2.coo.dsts.npy", dsts_shape, dsts_data);
    npy::LoadArrayFromNumpy("data/MyHybData/ogbn-wikikg2.coo.etypes.npy", etypes_shape, etypes_data);
    cusp::coo_matrix<int, int, cusp::host_memory> coo_matrix_h(num_nodes, num_nodes, srcs_data.size());
    for (int64_t i = 0; i < srcs_data.size(); i++) {
        coo_matrix_h.row_indices[i] = srcs_data[i];
        coo_matrix_h.column_indices[i] = dsts_data[i];
        coo_matrix_h.values[i] = etypes_data[i];
    }
    return coo_matrix_h;
}

int RGCNLayer1Profiling_main(cusp::csr_matrix<int, int, cusp::host_memory> graph, int64_t in_feat, int64_t out_feat){
    typedef int32_t Idx;
    typedef float DType;

    // load data

    

    //MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>> csr;
    //MySimpleNDArray<Idx, thrust::device_allocator<Idx>> eids({csr.total_num_nnzs});
    //thrust::sequence<>(eids.data.begin(),eids.data.end(), 0);
    //MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>> transposed_csr(csr);
    //MySimpleNDArray<Idx, thrust::device_allocator<Idx>> transposed_eids(eids);
    //transposed_csr.Transpose(transposed_eids);

    MyHeteroIntegratedCSR<Idx, std::allocator<Idx>> csr_h(graph.row_offsets, graph.column_indices, graph.values);
    MyHeteroIntegratedCSR<Idx, std::allocator<Idx>> transposed_csr_h(csr_h);
    MySimpleNDArray<Idx, std::allocator<Idx>> eids_h(std::vector<int64_t>{csr_h.total_num_nnzs});
    thrust::sequence<>(eids_h.data.begin(),eids_h.data.end(), 0);
    MySimpleNDArray<Idx, std::allocator<Idx>> transposed_eids_h(eids_h);
    transposed_csr_h.Transpose<>(std::optional<std::reference_wrapper<typename thrust::detail::vector_base<Idx, std::allocator<Idx>>>>{transposed_eids_h.data});

    // copy CSR+eid data to device

    MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>> csr(csr_h);
    MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>> transposed_csr(transposed_csr_h);
    MySimpleNDArray<Idx, thrust::device_allocator<Idx>> eids(std::vector<int64_t>{csr.total_num_nnzs},csr.rel_type);
    MySimpleNDArray<Idx, thrust::device_allocator<Idx>> transposed_eids(std::vector<int64_t>{transposed_csr.total_num_nnzs},transposed_csr.rel_type);
    

    MySimpleNDArray<DType, thrust::device_allocator<DType>> hidden=GenerateRandomNDArray<DType>({csr.total_num_nnzs,in_feat}); // TODO: assuming hidden is x. need to verify if that is correct

    MySimpleNDArray<DType, thrust::device_allocator<DType>> weight=GenerateRandomNDArray<DType>({csr.num_rels, in_feat, out_feat});
    // asuming num_bases == num_rels
    MySimpleNDArray<DType, thrust::device_allocator<DType>> norm=GenerateRandomNDArray<DType>({csr.total_num_nnzs,1});
    MySimpleNDArray<DType, thrust::device_allocator<DType>> ret=GenerateRandomNDArray<DType>({in_feat, out_feat});
    
    
    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_out=GenerateRandomNDArray<DType>({in_feat, out_feat});// TODO: verify if the assumption that the shape is the same as ret is correct
    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_hidden=GenerateRandomNDArray<DType>({csr.total_num_nnzs,in_feat});
    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_weight=GenerateRandomNDArray<DType>({csr.num_rels, in_feat, out_feat});

    // sort edges according to relationship type

    RgcnLayer1Impl<Idx, DType>(csr, hidden, weight, norm, ret, eids);
    RgcnLayer1BackwardImpl<Idx, DType>(transposed_csr, hidden, weight, norm, grad_out, grad_hidden, grad_weight, transposed_eids);
    return 0;
}
