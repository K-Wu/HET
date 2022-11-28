#pragma once
#include "DGLHackKernel/DGLHackKernel.h"
#include "DGLHackKernel/OpPrototyping/ModelsProfiling.h"

cusp::csr_matrix<int, int, cusp::host_memory> LoadFB15k237Data(
    bool sorted = false, bool sorted_by_src = false,
    std::string data_path_prefix = "data/MyHybData/") {
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

  if (sorted) {
    printf("WARNING: infidel loading FB15k237. Check readme.md for details.\n");
    if (sorted_by_src) {
      npy::LoadArrayFromNumpy(
          (data_path_prefix +
           "fb15k237.coo.infidel_sorted.by_srcs_outgoing_freq.srcs.npy")
              .c_str(),
          srcs_shape, fortran_order, srcs_data);
      npy::LoadArrayFromNumpy(
          (data_path_prefix +
           "fb15k237.coo.infidel_sorted.by_srcs_outgoing_freq.dsts.npy")
              .c_str(),
          dsts_shape, fortran_order, dsts_data);
      npy::LoadArrayFromNumpy(
          (data_path_prefix +
           "fb15k237.coo.infidel_sorted.by_srcs_outgoing_freq.etypes.npy")
              .c_str(),
          etypes_shape, fortran_order, etypes_data);
    } else {
      npy::LoadArrayFromNumpy(
          (data_path_prefix +
           "fb15k237.coo.infidel_sorted.by_etype_freq.srcs.npy")
              .c_str(),
          srcs_shape, fortran_order, srcs_data);
      npy::LoadArrayFromNumpy(
          (data_path_prefix +
           "fb15k237.coo.infidel_sorted.by_etype_freq.dsts.npy")
              .c_str(),
          dsts_shape, fortran_order, dsts_data);
      npy::LoadArrayFromNumpy(
          (data_path_prefix +
           "fb15k237.coo.infidel_sorted.by_etype_freq.etypes.npy")
              .c_str(),
          etypes_shape, fortran_order, etypes_data);
    }
  } else {
    npy::LoadArrayFromNumpy(
        (data_path_prefix + "fb15k237.coo.srcs.npy").c_str(), srcs_shape,
        fortran_order, srcs_data);
    npy::LoadArrayFromNumpy(
        (data_path_prefix + "fb15k237.coo.dsts.npy").c_str(), dsts_shape,
        fortran_order, dsts_data);
    npy::LoadArrayFromNumpy(
        (data_path_prefix + "fb15k237.coo.etypes.npy").c_str(), etypes_shape,
        fortran_order, etypes_data);
  }
  cusp::coo_matrix<Idx, Idx, cusp::host_memory> coo_matrix_h(
      num_nodes, num_nodes, srcs_data.size());
  // in this step, the raw data in int64_t is implicitly converted to the target
  // int32_t Idx type
  for (int64_t i = 0; i < srcs_data.size(); i++) {
    coo_matrix_h.row_indices[i] = srcs_data[i];
    coo_matrix_h.column_indices[i] = dsts_data[i];
    coo_matrix_h.values[i] = etypes_data[i];
  }
  return coo_matrix_h;
}

cusp::csr_matrix<int, int, cusp::host_memory> LoadOGBNWikiKG2Data(
    bool sorted = false, std::string data_path_prefix = "data/MyHybData/") {
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
  // num_relationship: 535
  if (sorted) {
    printf(
        "WARNING: infidel loading ogbn-wikikg2. Check readme.md for "
        "details.\n");
    npy::LoadArrayFromNumpy(
        (data_path_prefix + "ogbn-wikikg2.infidel_sorted.coo.srcs.npy").c_str(),
        srcs_shape, fortran_order, srcs_data);
    npy::LoadArrayFromNumpy(
        (data_path_prefix + "ogbn-wikikg2.infidel_sorted.coo.dsts.npy").c_str(),
        dsts_shape, fortran_order, dsts_data);
    npy::LoadArrayFromNumpy(
        (data_path_prefix + "ogbn-wikikg2.infidel_sorted.coo.etypes.npy")
            .c_str(),
        etypes_shape, fortran_order, etypes_data);
  } else {
    npy::LoadArrayFromNumpy(
        (data_path_prefix + "ogbn-wikikg2.coo.srcs.npy").c_str(), srcs_shape,
        fortran_order, srcs_data);
    npy::LoadArrayFromNumpy(
        (data_path_prefix + "ogbn-wikikg2.coo.dsts.npy").c_str(), dsts_shape,
        fortran_order, dsts_data);
    npy::LoadArrayFromNumpy(
        (data_path_prefix + "ogbn-wikikg2.coo.etypes.npy").c_str(),
        etypes_shape, fortran_order, etypes_data);
  }
  cusp::coo_matrix<Idx, Idx, cusp::host_memory> coo_matrix_h(
      num_nodes, num_nodes, srcs_data.size());
  for (int64_t i = 0; i < srcs_data.size(); i++) {
    coo_matrix_h.row_indices[i] = srcs_data[i];
    coo_matrix_h.column_indices[i] = dsts_data[i];
    coo_matrix_h.values[i] = etypes_data[i];
  }
  return coo_matrix_h;
}

MyHeteroIntegratedCSR<int, std::allocator<int>> LoadOGBN_MAG(
    std::string data_path_prefix = "data/ogbn_mag/") {
  std::vector<unsigned long> is_about_shape;
  std::vector<unsigned long> affliated_with_shape;
  std::vector<unsigned long> citing_shape;
  std::vector<unsigned long> writing_shape;

  bool fortran_order = false;
  std::vector<int> is_about_data;
  std::vector<int> affliated_with_data;
  std::vector<int> citing_data;
  std::vector<int> writing_data;

  npy::LoadArrayFromNumpy((data_path_prefix + "is-about_coo_1.npy").c_str(),
                          is_about_shape, fortran_order, is_about_data);
  npy::LoadArrayFromNumpy((data_path_prefix + "citing_coo_1.npy").c_str(),
                          citing_shape, fortran_order, citing_data);
  npy::LoadArrayFromNumpy((data_path_prefix + "writing_coo_1.npy").c_str(),
                          writing_shape, fortran_order, writing_data);
  npy::LoadArrayFromNumpy((data_path_prefix + "affliated_with_1.npy").c_str(),
                          affliated_with_shape, fortran_order,
                          affliated_with_data);

  // npy::LoadArrayFromNumpy("data/ogbn_mag_0.1/written-by_coo_2.npy",
  // written_by_shape, fortran_order, written_by_data);
  // npy::LoadArrayFromNumpy("data/ogbn_mag_0.1/has_coo_2.npy", has_shape,
  // fortran_order, has_data);
  // npy::LoadArrayFromNumpy("data/ogbn_mag_0.1/is-about_coo_2.npy",
  // is_about_shape, fortran_order, is_about_data);
  // npy::LoadArrayFromNumpy("data/ogbn_mag_0.1/cited_coo_2.npy", cited_shape,
  // fortran_order, cited_data);
  // npy::LoadArrayFromNumpy("data/ogbn_mag_0.1/citing_coo_2.npy", citing_shape,
  // fortran_order, citing_data);
  // npy::LoadArrayFromNumpy("data/ogbn_mag_0.1/writing_coo_2.npy",
  // writing_shape, fortran_order, writing_data);

  std::vector<int> max_idxes;
  max_idxes.push_back(
      *std::max_element(is_about_data.begin(), is_about_data.end()));
  max_idxes.push_back(*std::max_element(affliated_with_data.begin(),
                                        affliated_with_data.end()));
  max_idxes.push_back(
      *std::max_element(citing_data.begin(), citing_data.end()));
  max_idxes.push_back(
      *std::max_element(writing_data.begin(), writing_data.end()));
  int max_idx = *std::max_element(max_idxes.begin(), max_idxes.end());

  // cusp::csr_matrix<int, int, cusp::host_memory> csr_host(5, 8, 12);

  cusp::coo_matrix<int, int, cusp::host_memory> is_about_coo_h(
      max_idx + 1, max_idx + 1, is_about_data.size() / 2);
  for (int idx = 0; idx < is_about_data.size() / 2; idx++) {
    is_about_coo_h.row_indices[idx] = is_about_data[idx];
    is_about_coo_h.column_indices[idx] =
        is_about_data[idx + is_about_data.size() / 2];
  }

  cusp::coo_matrix<int, int, cusp::host_memory> affliated_with_coo_h(
      max_idx + 1, max_idx + 1, affliated_with_data.size() / 2);
  for (int idx = 0; idx < affliated_with_data.size() / 2; idx++) {
    affliated_with_coo_h.row_indices[idx] = affliated_with_data[idx];
    affliated_with_coo_h.column_indices[idx] =
        affliated_with_data[idx + affliated_with_data.size() / 2];
  }

  cusp::coo_matrix<int, int, cusp::host_memory> citing_coo_h(
      max_idx + 1, max_idx + 1, citing_data.size() / 2);
  for (int idx = 0; idx < citing_data.size() / 2; idx++) {
    citing_coo_h.row_indices[idx] = citing_data[idx];
    citing_coo_h.column_indices[idx] =
        citing_data[idx + citing_data.size() / 2];
  }

  cusp::coo_matrix<int, int, cusp::host_memory> writing_coo_h(
      max_idx + 1, max_idx + 1, writing_data.size() / 2);
  for (int idx = 0; idx < writing_data.size() / 2; idx++) {
    writing_coo_h.row_indices[idx] = writing_data[idx];
    writing_coo_h.column_indices[idx] =
        writing_data[idx + writing_data.size() / 2];
  }

  is_about_coo_h.sort_by_row_and_column();
  affliated_with_coo_h.sort_by_row_and_column();
  citing_coo_h.sort_by_row_and_column();
  writing_coo_h.sort_by_row_and_column();

  cusp::csr_matrix<int, int, cusp::host_memory> is_about_csr_h(is_about_coo_h);
  cusp::csr_matrix<int, int, cusp::host_memory> affliated_with_csr_h(
      affliated_with_coo_h);
  cusp::csr_matrix<int, int, cusp::host_memory> citing_csr_h(citing_coo_h);
  cusp::csr_matrix<int, int, cusp::host_memory> writing_csr_h(writing_coo_h);

  HET::OpPrototyping::MySimpleNDArray<int, std::allocator<int>> eids_h(
      std::vector<int64_t>{
          (int64_t)(is_about_data.size() / 2 + affliated_with_data.size() / 2 +
                    citing_data.size() / 2 + writing_data.size() / 2)});
  thrust::sequence<>(eids_h.data.begin(), eids_h.data.end(), 0);

  MyHeteroSeparateCSR<int, std::allocator<int>> my_hetero_separate_csr_h(
      std::vector<cusp::csr_matrix<int, int, cusp::host_memory>>{
          is_about_csr_h, affliated_with_csr_h, citing_csr_h, writing_csr_h},
      eids_h.data);

  MyHeteroIntegratedCSR<int, std::allocator<int>> my_hetero_integrated_csr_h =
      ToIntegratedCSR_CPU<int>(my_hetero_separate_csr_h);
  return my_hetero_integrated_csr_h;
}

MySegmentCSR<int, std::allocator<int>,
             MyHeteroSeparateCSR<int, std::allocator<int>>>
LoadSegmentCSR_OGBN_MAG() {
  typedef int Idx;

  // problem definition
  int num_nodes = 1939743;  // 1134649 authors + 59965 field of studies + 8740
                            // institution + 736389 papers
  int num_cutoff_nodes = 400000;
  // maximal non-cut-off index  734649
  // edge type num  4
  int maximal_non_cutoff_node_idx = num_nodes - num_cutoff_nodes + 1;
  int num_rels = 4;

  // TODO: use load array from numpy to MySimpleNDArray API to streamline the
  // following code

  // TODO: we also needs to load eid to unify eids
  // std::vector<int64_t> maximal_edge_num_per_src_node,
  // maximal_edge_type_per_src_node; std::vector<unsigned long>
  // maximal_edge_num_per_src_node_shape, maximal_edge_type_per_src_node_shape;
  // npy::LoadArrayFromNumpy("ogbn_mag.segment_csr.part_0.edge_nums.npy",
  // maximal_edge_num_per_src_node_shape, fortran_order,
  // maximal_edge_num_per_src_node);
  // npy::LoadArrayFromNumpy("ogbn_mag.segment_csr.part_0.edge_types.npy",
  // maximal_edge_type_per_src_node_shape, fortran_order,
  // maximal_edge_type_per_src_node);
  auto maximal_edge_num_per_src_node =
      HET::OpPrototyping::LoadMySimpleNDArrayFromNumpy<Idx, std::allocator<Idx>,
                                                       int64_t>(
          "data/MyHybData/SegmentCSR/"
          "ogbn_mag.segment_csr.part_0.edge_nums.npy");
  auto maximal_edge_types_per_src_node =
      HET::OpPrototyping::LoadMySimpleNDArrayFromNumpy<Idx, std::allocator<Idx>,
                                                       int64_t>(
          "data/MyHybData/SegmentCSR/"
          "ogbn_mag.segment_csr.part_0.edge_types.npy");

  // std::vector<int64_t> src_node_per_edge_type, num_src_nodes_per_edge_type;
  // std::vector<unsigned long> src_node_per_edge_type_shape,
  // num_src_nodes_per_edge_type_shape;
  // npy::LoadArrayFromNumpy("ogbn_mag.segment_csr.part_1.edge_type_num.npy",
  // src_node_per_edge_type_shape, fortran_order, src_node_per_edge_type);
  // npy::LoadArrayFromNumpy("ogbn_mag.segment_csr.part_1.src_node_per_edge_type.npy",
  // num_src_nodes_per_edge_type_shape, fortran_order,
  // num_src_nodes_per_edge_type);
  auto num_src_nodes_per_edge_type =
      HET::OpPrototyping::LoadMySimpleNDArrayFromNumpy<Idx, std::allocator<Idx>,
                                                       int64_t>(
          "data/MyHybData/SegmentCSR/"
          "ogbn_mag.segment_csr.part_1.edge_type_num.npy");
  auto src_node_per_edge_type =
      HET::OpPrototyping::LoadMySimpleNDArrayFromNumpy<Idx, std::allocator<Idx>,
                                                       int64_t>(
          "data/MyHybData/SegmentCSR/"
          "ogbn_mag.segment_csr.part_1.src_node_per_edge_type.npy");

  // std::vector<int64_t> dense_edges;
  // std::vector<unsigned long> dense_edges_shape;
  // npy::LoadArrayFromNumpy("ogbn_mag.segment_csr.part_2.maximal_edges.npy",
  // dense_edges_shape, fortran_order, dense_edges);
  // TODO: needs padding
  // TODO: when padding, both dense_edges and offset_num_src_nodes_per_edge_type
  // (or num_src_nodes_per_edge_type) needs to be padded.
  auto dense_edges =
      HET::OpPrototyping::LoadMySimpleNDArrayFromNumpy<Idx, std::allocator<Idx>,
                                                       int64_t>(
          "data/MyHybData/SegmentCSR/ogbn_mag.segment_csr.part_2.edges.npy");

  // std::vector<int64_t> residue_srcs_data, residue_dsts_data,
  // residue_etypes_data; std::vector<unsigned long> residue_srcs_shape,
  // residue_dsts_shape, residue_etypes_shape;
  // npy::LoadArrayFromNumpy("data/MyHybData/SegmentCSR/ogbn_mag.segment_csr.part_3.srcs.npy",
  // residue_srcs_shape, fortran_order, residue_srcs_data);
  // npy::LoadArrayFromNumpy("data/MyHybData/SegmentCSR/ogbn_mag.segment_csr.part_3.dsts.npy",
  // residue_dsts_shape, fortran_order, residue_dsts_data);
  // npy::LoadArrayFromNumpy("data/MyHybData/SegmentCSR/ogbn_mag.segment_csr.part_3.types.npy",
  // residue_etypes_shape, fortran_order, residue_etypes_data);
  auto residue_srcs_data =
      HET::OpPrototyping::LoadMySimpleNDArrayFromNumpy<Idx, std::allocator<Idx>,
                                                       int64_t>(
          "data/MyHybData/SegmentCSR/ogbn_mag.segment_csr.part_3.srcs.npy");
  auto residue_dsts_data =
      HET::OpPrototyping::LoadMySimpleNDArrayFromNumpy<Idx, std::allocator<Idx>,
                                                       int64_t>(
          "data/MyHybData/SegmentCSR/ogbn_mag.segment_csr.part_3.dsts.npy");
  auto residue_etypes_data =
      HET::OpPrototyping::LoadMySimpleNDArrayFromNumpy<Idx, std::allocator<Idx>,
                                                       int64_t>(
          "data/MyHybData/SegmentCSR/ogbn_mag.segment_csr.part_3.types.npy");

  cusp::coo_matrix<Idx, Idx, cusp::host_memory> residue_coo_matrix_h(
      num_nodes, num_nodes, residue_srcs_data.data.size());
  std::vector<int64_t> residue_num_nnzs(num_rels, 0);
  for (int64_t i = 0; i < residue_srcs_data.data.size(); i++) {
    residue_coo_matrix_h.row_indices[i] = residue_srcs_data.data[i];
    residue_coo_matrix_h.column_indices[i] = residue_dsts_data.data[i];
    residue_coo_matrix_h.values[i] = residue_etypes_data.data[i];
    residue_num_nnzs[residue_etypes_data.data[i]]++;
  }
  cusp::csr_matrix<Idx, Idx, cusp::host_memory> residue_csr_matrix_h(
      residue_coo_matrix_h);
  thrust::host_vector<Idx> residue_coo_eids(residue_srcs_data.data.size());
  thrust::sequence<>(residue_coo_eids.begin(), residue_coo_eids.end(), 0);
  thrust::host_vector<Idx> dense_edges_eids(dense_edges.data.size());
  thrust::sequence<>(dense_edges_eids.begin(), dense_edges_eids.end(),
                     residue_srcs_data.data.size());

  MyHeteroIntegratedCSR<Idx, std::allocator<Idx>> residude_csr_integrated(
      residue_csr_matrix_h.num_rows, residue_csr_matrix_h.num_cols, num_rels,
      residue_num_nnzs, residue_csr_matrix_h.row_offsets,
      residue_csr_matrix_h.column_indices, residue_csr_matrix_h.values,
      residue_coo_eids);

  MyHeteroSeparateCSR<Idx, std::allocator<Idx>> residue_csr =
      ToSeparateCSR_CPU<Idx>(residude_csr_integrated);

  auto pad_result = MySegmentCSRPadDenseEdges(
      dense_edges.data, maximal_edge_num_per_src_node.data, 8);
  auto pad_result2 = MySegmentCSRPadDenseEdges(
      dense_edges_eids, maximal_edge_num_per_src_node.data, 8);
  auto padded_dense_edges = pad_result.second;
  auto padded_exclusive_scan_maximal_edge_num_per_src_node = pad_result.first;
  auto padded_dense_edges_eids = pad_result2.second;

  MySegmentCSR<Idx, std::allocator<Idx>,
               MyHeteroSeparateCSR<Idx, std::allocator<Idx>>>
      mysegmentcsr(num_nodes, num_nodes, maximal_non_cutoff_node_idx,
                   maximal_edge_num_per_src_node.data,
                   maximal_edge_types_per_src_node.data,
                   src_node_per_edge_type.data,
                   num_src_nodes_per_edge_type.data, padded_dense_edges,
                   padded_exclusive_scan_maximal_edge_num_per_src_node,
                   residue_csr, padded_dense_edges_eids);
  return mysegmentcsr;
}

int RGCNLayer1Profiling_MyHYB_main(
    cusp::csr_matrix<int, int, cusp::host_memory> graph, int64_t in_feat,
    int64_t out_feat) {
  return HET::OpPrototyping::_RGCNLayer1Profiling_main(graph, in_feat, out_feat,
                                                       true, false);
}

int RGCNLayer1Profiling_main(
    cusp::csr_matrix<int, int, cusp::host_memory> graph, int64_t in_feat,
    int64_t out_feat) {
  return HET::OpPrototyping::_RGCNLayer1Profiling_main(graph, in_feat, out_feat,
                                                       false, false);
}

int RGCNLayer1Profiling_main_check_correctness(
    cusp::csr_matrix<int, int, cusp::host_memory> graph, int64_t in_feat,
    int64_t out_feat) {
  return HET::OpPrototyping::_RGCNLayer1Profiling_main(graph, in_feat, out_feat,
                                                       true, true);
}