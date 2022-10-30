#pragma once
// std::vector<at::Tensor> load_fb15k237(bool sorted, bool sorted_by_src,
// std::string data_path_prefix){
//   cusp::csr_matrix<int, int, cusp::host_memory> fb15k237_graph =
//   LoadFB15k237Data(sorted, sorted_by_src, data_path_prefix);
// }

// std::vector<at::Tensor> load_ogbn_wikikg2(bool sorted, std::string
// data_path_prefix){
//   cusp::csr_matrix<int, int, cusp::host_memory> ogbn_wikikg2_graph =
//   LoadOGBNWikiKG2Data(sorted, data_path_prefix);
// }

// std::vector<at::Tensor> load_mag(std::string data_path_prefix){
//   MyHeteroIntegratedCSR<int, std::allocator<int>>  mag_graph =
//   LoadOGBN_MAG(data_path_prefix);

// }

// Add the following lines to TORCH_LIBRARY declaration once complete
// m.def("load_fb15k237", load_fb15k237);
// m.def("load_ogbn_wikikg2", load_ogbn_wikikg2);
// m.def("load_mag", load_mag);
