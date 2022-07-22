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
    BackwardFusedGatKernelImpl<Idx, DType, true>(outcsr, feat_src, el, er, sum, exp, ret, grad_out, grad_feat_src, grad_el, grad_er, slope);
    BackwardFusedGatKernelImpl<Idx, DType, false>(outcsr, feat_src, el, er, sum, exp, ret, grad_out, grad_feat_src, grad_el, grad_er, slope);
    return 0;
}

cusp::csr_matrix<int, int, cusp::host_memory> LoadFB15k237Data(){
    typedef int Idx;
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
    cusp::coo_matrix<Idx, Idx, cusp::host_memory> coo_matrix_h(num_nodes, num_nodes, srcs_data.size());
    for (int64_t i = 0; i < srcs_data.size(); i++) {
        coo_matrix_h.row_indices[i] = srcs_data[i];
        coo_matrix_h.column_indices[i] = dsts_data[i];
        coo_matrix_h.values[i] = etypes_data[i];
    }
    return coo_matrix_h;
}

cusp::csr_matrix<int, int, cusp::host_memory> LoadOGBNWikiKG2Data(){
    typedef int Idx;
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
    cusp::coo_matrix<Idx, Idx, cusp::host_memory> coo_matrix_h(num_nodes, num_nodes, srcs_data.size());
    for (int64_t i = 0; i < srcs_data.size(); i++) {
        coo_matrix_h.row_indices[i] = srcs_data[i];
        coo_matrix_h.column_indices[i] = dsts_data[i];
        coo_matrix_h.values[i] = etypes_data[i];
    }
    return coo_matrix_h;
}

MySegmentCSR<int, std::allocator<int>, MyHeteroSeparateCSR<int, std::allocator<int>>> LoadSegmentCSR_OGBN_MAG(){
    typedef int Idx;

    bool fortran_order = false;
    // problem definition
    int num_nodes=1939743;// 1134649 authors + 59965 field of studies + 8740 institution + 736389 papers
    int num_cutoff_nodes=400000;
    //maximal non-cut-off index  734649
    //edge type num  4
    int maximal_non_cutoff_node_idx = num_nodes-num_cutoff_nodes+1;
    int num_rels = 4;

    // TODO: use load array from numpy to MySimpleNDArray API to streamline the following code

    // TODO: we also needs to load eid to unify eids
    //std::vector<int64_t> maximal_edge_num_per_src_node, maximal_edge_type_per_src_node;
    //std::vector<unsigned long> maximal_edge_num_per_src_node_shape, maximal_edge_type_per_src_node_shape;
    //npy::LoadArrayFromNumpy("ogbn_mag.segment_csr.part_0.edge_nums.npy", maximal_edge_num_per_src_node_shape, fortran_order, maximal_edge_num_per_src_node);
    //npy::LoadArrayFromNumpy("ogbn_mag.segment_csr.part_0.edge_types.npy", maximal_edge_type_per_src_node_shape, fortran_order, maximal_edge_type_per_src_node);
    auto maximal_edge_num_per_src_node = LoadMySimpleNDArrayFromNumpy<Idx, std::allocator<Idx>,int64_t>("data/MyHybData/SegmentCSR/ogbn_mag.segment_csr.part_0.edge_nums.npy");
    auto maximal_edge_types_per_src_node = LoadMySimpleNDArrayFromNumpy<Idx, std::allocator<Idx>,int64_t>("data/MyHybData/SegmentCSR/ogbn_mag.segment_csr.part_0.edge_types.npy");

    //std::vector<int64_t> src_node_per_edge_type, num_src_nodes_per_edge_type;
    //std::vector<unsigned long> src_node_per_edge_type_shape, num_src_nodes_per_edge_type_shape;
    //npy::LoadArrayFromNumpy("ogbn_mag.segment_csr.part_1.edge_type_num.npy", src_node_per_edge_type_shape, fortran_order, src_node_per_edge_type);
    //npy::LoadArrayFromNumpy("ogbn_mag.segment_csr.part_1.src_node_per_edge_type.npy", num_src_nodes_per_edge_type_shape, fortran_order, num_src_nodes_per_edge_type);
    auto src_node_per_edge_type = LoadMySimpleNDArrayFromNumpy<Idx, std::allocator<Idx>,int64_t>("data/MyHybData/SegmentCSR/ogbn_mag.segment_csr.part_1.edge_type_num.npy");
    auto num_src_nodes_per_edge_type = LoadMySimpleNDArrayFromNumpy<Idx, std::allocator<Idx>,int64_t>("data/MyHybData/SegmentCSR/ogbn_mag.segment_csr.part_1.src_node_per_edge_type.npy");

    //std::vector<int64_t> dense_edges;
    //std::vector<unsigned long> dense_edges_shape;
    //npy::LoadArrayFromNumpy("ogbn_mag.segment_csr.part_2.maximal_edges.npy", dense_edges_shape, fortran_order, dense_edges);
    // TODO: needs padding
    // TODO: when padding, both dense_edges and offset_num_src_nodes_per_edge_type (or num_src_nodes_per_edge_type) needs to be padded.
    auto dense_edges = LoadMySimpleNDArrayFromNumpy<Idx, std::allocator<Idx>,int64_t>("data/MyHybData/SegmentCSR/ogbn_mag.segment_csr.part_2.edges.npy");
    
    //std::vector<int64_t> residue_srcs_data, residue_dsts_data, residue_etypes_data;
    //std::vector<unsigned long> residue_srcs_shape, residue_dsts_shape, residue_etypes_shape;
    //npy::LoadArrayFromNumpy("data/MyHybData/SegmentCSR/ogbn_mag.segment_csr.part_3.srcs.npy", residue_srcs_shape, fortran_order, residue_srcs_data);
    //npy::LoadArrayFromNumpy("data/MyHybData/SegmentCSR/ogbn_mag.segment_csr.part_3.dsts.npy", residue_dsts_shape, fortran_order, residue_dsts_data);
    //npy::LoadArrayFromNumpy("data/MyHybData/SegmentCSR/ogbn_mag.segment_csr.part_3.types.npy", residue_etypes_shape, fortran_order, residue_etypes_data);
    auto residue_srcs_data = LoadMySimpleNDArrayFromNumpy<Idx, std::allocator<Idx>,int64_t>("data/MyHybData/SegmentCSR/ogbn_mag.segment_csr.part_3.srcs.npy");
    auto residue_dsts_data = LoadMySimpleNDArrayFromNumpy<Idx, std::allocator<Idx>,int64_t>("data/MyHybData/SegmentCSR/ogbn_mag.segment_csr.part_3.dsts.npy");
    auto residue_etypes_data = LoadMySimpleNDArrayFromNumpy<Idx, std::allocator<Idx>,int64_t>("data/MyHybData/SegmentCSR/ogbn_mag.segment_csr.part_3.types.npy");

    cusp::coo_matrix<Idx, Idx, cusp::host_memory> residue_coo_matrix_h(num_nodes, num_nodes, residue_srcs_data.data.size());
    std::vector<int64_t> residue_num_nnzs(num_rels,0);
    for (int64_t i = 0; i < residue_srcs_data.data.size(); i++) {
        residue_coo_matrix_h.row_indices[i] = residue_srcs_data.data[i];
        residue_coo_matrix_h.column_indices[i] = residue_dsts_data.data[i];
        residue_coo_matrix_h.values[i] = residue_etypes_data.data[i];
        residue_num_nnzs[residue_etypes_data.data[i]]++;
    }
    cusp::csr_matrix<Idx, Idx, cusp::host_memory> residue_csr_matrix_h(residue_coo_matrix_h);
    thrust::host_vector<Idx> residue_coo_eids(residue_srcs_data.data.size());
    thrust::sequence<>(residue_coo_eids.begin(), residue_coo_eids.end(), 0);
    thrust::host_vector<Idx> dense_edges_eids(dense_edges.data.size());
    thrust::sequence<>(dense_edges_eids.begin(), dense_edges_eids.end(), residue_srcs_data.data.size());

    MyHeteroIntegratedCSR<Idx, std::allocator<Idx>> residude_csr_integrated(residue_csr_matrix_h.num_rows, residue_csr_matrix_h.num_cols, num_rels,
        residue_num_nnzs,
        residue_csr_matrix_h.row_offsets,
        residue_csr_matrix_h.column_indices,
        residue_csr_matrix_h.values,
        residue_coo_eids);

    MyHeteroSeparateCSR<Idx, std::allocator<Idx>> residue_csr = ToSeparateCSR_CPU<Idx>(residude_csr_integrated);

    auto pad_result = MySegmentCSRPadDenseEdges(dense_edges.data, maximal_edge_num_per_src_node.data, 8);
    auto pad_result2 = MySegmentCSRPadDenseEdges(dense_edges_eids, maximal_edge_num_per_src_node.data, 8);
    auto padded_dense_edges = pad_result.first;
    auto padded_exclusive_scan_maximal_edge_num_per_src_node = pad_result.second;
    auto padded_dense_edges_eids = pad_result2.first;

    MySegmentCSR<Idx, std::allocator<Idx>, MyHeteroSeparateCSR<Idx, std::allocator<Idx>>> mysegmentcsr(num_nodes, num_nodes, maximal_non_cutoff_node_idx, maximal_edge_num_per_src_node.data, maximal_edge_types_per_src_node.data, src_node_per_edge_type.data, num_src_nodes_per_edge_type.data, padded_dense_edges, padded_exclusive_scan_maximal_edge_num_per_src_node, residue_csr, padded_dense_edges_eids);
    return mysegmentcsr;
}

int _HGTExperimental_main(MySegmentCSR<int, std::allocator<int>, MyHeteroSeparateCSR<int, std::allocator<int>>>& graph, int num_heads, int in_feat, int out_feat){
    assert(num_heads == 4);
    typedef int32_t Idx;
    typedef float DType;
    MySegmentCSR<int, thrust::device_allocator<int>, MyHeteroSeparateCSR<int, thrust::device_allocator<int>>> deivce_graph=graph;
    MySimpleNDArray<DType, thrust::device_allocator<DType>> node_features = GenerateRandomNDArray<DType>({graph.num_rows, num_heads, in_feat});
    MySimpleNDArray<DType, thrust::device_allocator<DType>> weight=GenerateRandomNDArray<DType>({graph.num_rels, num_heads, in_feat, out_feat});
    //MySimpleNDArray<DType, thrust::device_allocator<DType>> intermediate_vectors=GenerateRandomNDArray<DType>({graph.num_rels, graph.num_rows, num_heads, out_feat});
    MySimpleNDArray<DType, thrust::device_allocator<DType>> attention = GenerateRandomNDArray<DType>({graph.total_num_nnzs, 4});
    HGTForwardImpl(deivce_graph, num_heads, in_feat, out_feat, node_features, weight, attention);
    return 0;
}

int _RGCNLayer1Profiling_main(cusp::csr_matrix<int, int, cusp::host_memory> graph, int64_t in_feat, int64_t out_feat, bool flagUseMyHyb, bool flagCheckCorrect){
    typedef int32_t Idx;
    typedef float DType;
    if (flagCheckCorrect){
        std::cout<<"Warning: flagCheckCorrect is true in _RGCNLayer1Profiling_main, ignoring flagUseMyHyb and both myhyb and the original kernels will be run."<<std::endl;
    }

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

    MySimpleNDArray<DType, thrust::device_allocator<DType>> hidden=GenerateRandomNDArray<DType>({csr_h.num_rows,in_feat}); // TODO: assuming hidden is x. need to verify if that is correct
    MySimpleNDArray<DType, thrust::device_allocator<DType>> weight=GenerateRandomNDArray<DType>({csr_h.num_rels, in_feat, out_feat});
    // asuming num_bases == num_rels
    MySimpleNDArray<DType, thrust::device_allocator<DType>> norm=GenerateRandomNDArray<DType>({csr_h.total_num_nnzs,1});
    MySimpleNDArray<DType, thrust::device_allocator<DType>> ret;//=GenerateRandomNDArray<DType>({myhyb.num_rows, out_feat});
    MySimpleNDArray<DType, thrust::device_allocator<DType>> ret2;//=GenerateRandomNDArray<DType>({csr.num_rows, out_feat});
    
    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_out=GenerateRandomNDArray<DType>({csr_h.num_rows, out_feat});// TODO: verify if the assumption that the shape is the same as ret is correct
    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_hidden;//=GenerateRandomNDArray<DType>({myhyb.total_num_nnzs,in_feat});
    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_weight;//=GenerateRandomNDArray<DType>({myhyb.num_rels, in_feat, out_feat});
    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_hidden2;//=GenerateRandomNDArray<DType>({myhyb.total_num_nnzs,in_feat});
    MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_weight2;//=GenerateRandomNDArray<DType>({myhyb.num_rels, in_feat, out_feat});

    if ((!flagUseMyHyb)|| flagCheckCorrect) {
        ret=GenerateRandomNDArray<DType>({csr_h.num_rows, out_feat});
        grad_hidden=GenerateRandomNDArray<DType>({csr_h.total_num_nnzs,in_feat});
        grad_weight=GenerateRandomNDArray<DType>({csr_h.num_rels, in_feat, out_feat});
        RgcnLayer1Impl<Idx, DType>(csr, hidden, weight, norm, ret);
        RgcnLayer1BackwardImpl<Idx, DType>(transposed_csr, hidden, weight, norm, grad_out, grad_hidden, grad_weight);
    }
    if (flagUseMyHyb || flagCheckCorrect) {
        ret2 = GenerateRandomNDArray<DType>({csr_h.num_rows, out_feat});
        grad_hidden2=GenerateRandomNDArray<DType>({csr_h.total_num_nnzs,in_feat});
        grad_weight2=GenerateRandomNDArray<DType>({csr_h.num_rels, in_feat, out_feat});
        
        RgcnLayer1MyHYBImpl<Idx, DType, 4, 4>(myhyb, hidden, weight, norm, ret2);
        RgcnLayer1BackwardMyHYBImpl<Idx, DType, 4, 4>(transposed_myhyb, hidden, weight, norm, grad_out, grad_hidden2, grad_weight2);
    }
    

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