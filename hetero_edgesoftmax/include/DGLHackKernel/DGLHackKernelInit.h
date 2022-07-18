#pragma once
#include "DGLHackKernel.h"


int FusetGATProfiling_main(cusp::csr_matrix<int, int, cusp::host_memory> graph, int64_t num_heads, int64_t num_hidden){
    typedef int32_t Idx;
    typedef float DType;

    MySimpleNDArray<Idx, std::allocator<Idx>> eids_h(std::vector<int64_t>{(int64_t) graph.values.size()});
    thrust::sequence<>(eids_h.data.begin(),eids_h.data.end(), 0);
    //MySimpleNDArray<Idx, std::allocator<Idx>> transposed_eids_h(eids_h);

    MyHeteroSeparateCSR<Idx, std::allocator<Idx>> incsr_h(std::vector<cusp::csr_matrix<int, int, cusp::host_memory>>{graph}, eids_h.data);
    MyHeteroSeparateCSR<Idx, std::allocator<Idx>> outcsr_h(incsr_h);
    
    outcsr_h.Transpose();//std::optional<std::reference_wrapper<typename thrust::detail::vector_base<Idx, std::allocator<Idx>>>>{transposed_eids_h.data});

    // copy CSR+eid data to device

    MyHeteroSeparateCSR<Idx, thrust::device_allocator<Idx>> incsr(incsr_h);
    MyHeteroSeparateCSR<Idx, thrust::device_allocator<Idx>> outcsr(outcsr_h);
    //MySimpleNDArray<Idx, thrust::device_allocator<Idx>> eids(eids_h);
    //MySimpleNDArray<Idx, thrust::device_allocator<Idx>> transposed_eids(transposed_eids_h);
    

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

    float slope=0.2;

    FusedGatKernelImpl<Idx, DType>(incsr, feat_src, el, er, sum, exp, ret, slope);
    // TODO: check if transpsoed eid is needed here
    BackwardFusedGatKernelImpl<Idx, DType>(outcsr, feat_src, el, er, sum, exp, ret, grad_out, grad_feat_src, grad_el, grad_er, slope);
    return 0;
}
// TODO: implement thrust::vector Transpose(CSRType, eids) with optional eid as output

cusp::csr_matrix<int, int, cusp::host_memory> LoadFB15k237Data(){
    std::vector<unsigned long> srcs_shape;
    std::vector<unsigned long> dsts_shape;
    std::vector<unsigned long> etypes_shape;

    bool fortran_order = false;
    std::vector<int64_t> srcs_data;
    std::vector<int64_t> dsts_data;
    std::vector<int64_t> etypes_data;

    int num_nodes = 14541;
    int num_edges = 620232;

    npy::LoadArrayFromNumpy("data/MyHybData/fb15k237.coo.srcs.npy", srcs_shape, fortran_order, srcs_data);
    npy::LoadArrayFromNumpy("data/MyHybData/fb15k237.coo.dsts.npy", dsts_shape, fortran_order, dsts_data);
    npy::LoadArrayFromNumpy("data/MyHybData/fb15k237.coo.etypes.npy", etypes_shape, fortran_order, etypes_data);
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

    bool fortran_order = false;
    std::vector<int64_t> srcs_data;
    std::vector<int64_t> dsts_data;
    std::vector<int64_t> etypes_data;

    int num_nodes = 2500604;
    int num_edges = 16109182;

    npy::LoadArrayFromNumpy("data/MyHybData/ogbn-wikikg2.coo.srcs.npy", srcs_shape, fortran_order, srcs_data);
    npy::LoadArrayFromNumpy("data/MyHybData/ogbn-wikikg2.coo.dsts.npy", dsts_shape, fortran_order, dsts_data);
    npy::LoadArrayFromNumpy("data/MyHybData/ogbn-wikikg2.coo.etypes.npy", etypes_shape, fortran_order, etypes_data);
    cusp::coo_matrix<int, int, cusp::host_memory> coo_matrix_h(num_nodes, num_nodes, srcs_data.size());
    for (int64_t i = 0; i < srcs_data.size(); i++) {
        coo_matrix_h.row_indices[i] = srcs_data[i];
        coo_matrix_h.column_indices[i] = dsts_data[i];
        coo_matrix_h.values[i] = etypes_data[i];
    }
    return coo_matrix_h;
}


int _RGCNLayer1Profiling_main(cusp::csr_matrix<int, int, cusp::host_memory> graph, int64_t in_feat, int64_t out_feat, bool flagUseMyHyb, bool flagCheckCorrect){
    typedef int32_t Idx;
    typedef float DType;
    // load data
    //MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>> csr;
    //MySimpleNDArray<Idx, thrust::device_allocator<Idx>> eids({csr.total_num_nnzs});
    //thrust::sequence<>(eids.data.begin(),eids.data.end(), 0);
    //MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>> transposed_csr(csr);
    //MySimpleNDArray<Idx, thrust::device_allocator<Idx>> transposed_eids(eids);
    //transposed_csr.Transpose(transposed_eids);
    MySimpleNDArray<Idx, std::allocator<Idx>> eids_h(std::vector<int64_t>{(int64_t) graph.column_indices.size()});
    thrust::sequence<>(eids_h.data.begin(),eids_h.data.end(), 0);
    MySimpleNDArray<Idx, std::allocator<Idx>> transposed_eids_h(eids_h);

    MyHeteroIntegratedCSR<Idx, std::allocator<Idx>> csr_h(graph.row_offsets, graph.column_indices, graph.values, eids_h.data);
    MyHeteroIntegratedCSR<Idx, std::allocator<Idx>> transposed_csr_h(csr_h);
    
    //transposed_csr_h.Transpose<>(std::optional<std::reference_wrapper<typename thrust::detail::vector_base<Idx, std::allocator<Idx>>>>{transposed_eids_h.data});
    transposed_csr_h.Transpose();

    
    //MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>> csr(csr_h);
    //MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>> transposed_csr(transposed_csr_h);
    //MySimpleNDArray<Idx, thrust::device_allocator<Idx>> eids(eids_h);
    //MySimpleNDArray<Idx, thrust::device_allocator<Idx>> transposed_eids(transposed_eids_h);

    MyHyb<Idx, std::allocator<Idx>, MyHeteroIntegratedCSR<Idx, std::allocator<Idx>>> myhyb_h;// = IntegratedCSRToHyb_ADHOC_CPU(csr_h, 4, 4, csr_h.num_rows);
    MyHyb<Idx, std::allocator<Idx>, MyHeteroIntegratedCSR<Idx, std::allocator<Idx>>> transposed_myhyb_h;// = IntegratedCSRToHyb_ADHOC_CPU(transposed_csr_h, 4, 4, transposed_csr_h.num_rows);
    if (flagUseMyHyb || flagCheckCorrect) {
        myhyb_h = IntegratedCSRToHyb_ADHOC_CPU(csr_h, 4, 4, csr_h.num_rows);
        transposed_myhyb_h=IntegratedCSRToHyb_ADHOC_CPU(transposed_csr_h, 4, 4, transposed_csr_h.num_rows);
    }
    // copy MyHyb data to device and/or copy CSR+eid data to device
    MyHyb<Idx, thrust::device_allocator<Idx>, MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>>>  myhyb(myhyb_h);
    MyHyb<Idx, thrust::device_allocator<Idx>, MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>>> transposed_myhyb(transposed_myhyb_h);
    
    MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>> csr;//(csr_h);
    MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>> transposed_csr;//(transposed_csr_h);

    if ((!flagUseMyHyb) || flagCheckCorrect) {
        csr = csr_h;
        transposed_csr = transposed_csr_h;
    }

    MySimpleNDArray<DType, thrust::device_allocator<DType>> hidden=GenerateRandomNDArray<DType>({myhyb.num_rows,in_feat}); // TODO: assuming hidden is x. need to verify if that is correct
    MySimpleNDArray<DType, thrust::device_allocator<DType>> weight=GenerateRandomNDArray<DType>({myhyb.num_rels, in_feat, out_feat});
    // asuming num_bases == num_rels
    MySimpleNDArray<DType, thrust::device_allocator<DType>> norm=GenerateRandomNDArray<DType>({myhyb.total_num_nnzs,1});
    MySimpleNDArray<DType, thrust::device_allocator<DType>> ret;//=GenerateRandomNDArray<DType>({myhyb.num_rows, out_feat});
    MySimpleNDArray<DType, thrust::device_allocator<DType>> ret2;//=GenerateRandomNDArray<DType>({csr.num_rows, out_feat});
    
    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_out=GenerateRandomNDArray<DType>({myhyb.num_rows, out_feat});// TODO: verify if the assumption that the shape is the same as ret is correct
    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_hidden;//=GenerateRandomNDArray<DType>({myhyb.total_num_nnzs,in_feat});
    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_weight;//=GenerateRandomNDArray<DType>({myhyb.num_rels, in_feat, out_feat});
    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_hidden2;//=GenerateRandomNDArray<DType>({myhyb.total_num_nnzs,in_feat});
    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_weight2;//=GenerateRandomNDArray<DType>({myhyb.num_rels, in_feat, out_feat});

    if ((!flagUseMyHyb)|| flagCheckCorrect) {
        ret=GenerateRandomNDArray<DType>({myhyb.num_rows, out_feat});
        grad_hidden=GenerateRandomNDArray<DType>({myhyb.total_num_nnzs,in_feat});
        grad_weight=GenerateRandomNDArray<DType>({myhyb.num_rels, in_feat, out_feat});
    }
    if (flagUseMyHyb || flagCheckCorrect) {
        ret2 = GenerateRandomNDArray<DType>({csr.num_rows, out_feat});
        grad_hidden2=GenerateRandomNDArray<DType>({myhyb.total_num_nnzs,in_feat});
        grad_weight2=GenerateRandomNDArray<DType>({myhyb.num_rels, in_feat, out_feat});
    }
    
    // sort edges according to relationship type

    RgcnLayer1MyHYBImpl<Idx, DType, 4, 4>(myhyb, hidden, weight, norm, ret);
    RgcnLayer1BackwardMyHYBImpl<Idx, DType, 4, 4>(transposed_myhyb, hidden, weight, norm, grad_out, grad_hidden, grad_weight);
    RgcnLayer1Impl<Idx, DType>(csr, hidden, weight, norm, ret2);
    RgcnLayer1BackwardImpl<Idx, DType>(transposed_csr, hidden, weight, norm, grad_out, grad_hidden2, grad_weight2);

    if (flagCheckCorrect){
        std::cout<<"check correctness in _RGCNLayer1Profiling_main"<<std::endl;
        std::cout<<"ret: "<< ret.IsEqual<>(ret2)<< std::endl;
        std::cout<<"grad_hidden: "<< grad_hidden.IsEqual<>(grad_hidden2)<< std::endl;
        std::cout<<"grad_weight: "<< grad_weight.IsEqual<>(grad_weight2)<< std::endl;
    }

    return 0;
}

int RGCNLayer1Profiling_MyHYB_main(cusp::csr_matrix<int, int, cusp::host_memory> graph, int64_t in_feat, int64_t out_feat){
    return _RGCNLayer1Profiling_main(graph, in_feat, out_feat, true, false);
}


int RGCNLayer1Profiling_main(cusp::csr_matrix<int, int, cusp::host_memory> graph, int64_t in_feat, int64_t out_feat){
    return _RGCNLayer1Profiling_main(graph, in_feat, out_feat, false, false);
}

int RGCNLayer1Profiling_main_check_correctness(cusp::csr_matrix<int, int, cusp::host_memory> graph, int64_t in_feat, int64_t out_feat){
    return _RGCNLayer1Profiling_main(graph, in_feat, out_feat, true, true);
}