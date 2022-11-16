// from
// https://stackoverflow.com/questions/68401650/how-can-i-make-a-pytorch-extension-with-cmake
// kernels defined in this file are simply wrapped up in
// hetero_edgesoftmax/python/kernels to provide python API, then used to
// define autograd functions and layers in hetero_edgesoftmax/python/kernels,
// which is finally referred to by end2end cases in
// hetero_edgesoftmax/python/<model name>/.*.py
// NB: This contains wrapper versions for python api export originally
// implemented at
// [[hetero_edgesoftmax/include/DGLHackKernel/RGCN/RGCNLayersKernels.cu.h]].
// Please update accordingly whenever there is update.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <iostream>
#include <map>

#include "DGLHackKernel/DGLHackKernel.h"
#include "hetero_edgesoftmax.h"
// TODO: assume int32_t and float32 for now. but we may need to support other
// types
// TODO: check if torch builtin has the same encoding as int32_t and float32
#include "DGLHackKernel/OpExport/DataConverters.inc.h"
#include "DGLHackKernel/OpExport/GATOps.inc.h"
#include "DGLHackKernel/OpExport/HGTOps.inc.h"
#include "DGLHackKernel/OpExport/HGTPrepToAndFromTensors.h"
#include "DGLHackKernel/OpExport/RGATOps.inc.h"
#include "DGLHackKernel/OpExport/RGCNCOOOps.inc.h"
#include "DGLHackKernel/OpExport/RGCNOps.inc.h"

#include "DGLHackKernel/mysgemm/mysgemm_KernelsBlockConfigurations.h"

void try_get_schedule_by_relations(int64_t num_relations, int64_t num_blocks) {
  std::vector<int64_t> mock_job_entries_per_relation_vec(num_relations, 1000);
  auto [num_blocks_along_dimx_for_same_relation_vect,
        num_blocks_along_dimx_for_all_prev_relation_vect] =
      get_schedule_by_relation_kernel_launch_metadata<
          false, false, std::vector<int64_t>::iterator>(
          num_relations, num_blocks, -1,
          mock_job_entries_per_relation_vec.begin(),
          mock_job_entries_per_relation_vec.end());

  // std::vector<int> mock_vect(100, 100);
  // std::vector<int> mock_vect2(100, 100);
  // thrust::device_vector<int>
  // dev_num_blocks_along_dimx_for_same_relation_vect(100);
  // thrust::device_vector<int>
  // dev_num_blocks_along_dimx_for_all_prev_relation_vect(100); for (int idx =
  // 0; idx < 100; idx++){
  //     dev_num_blocks_along_dimx_for_same_relation_vect[idx] = mock_vect[idx];
  //     dev_num_blocks_along_dimx_for_all_prev_relation_vect[idx] =
  //     mock_vect2[idx];
  // }
  // thrust::device_vector<int>
  // dev_num_blocks_along_dimx_for_same_relation_vect(mock_vect.begin(),
  // mock_vect.end()); thrust::device_vector<int>
  // dev_num_blocks_along_dimx_for_all_prev_relation_vect(mock_vect2.begin(),
  // mock_vect2.end());
  thrust::device_vector<int> dev_num_blocks_along_dimx_for_same_relation_vect(
      num_blocks_along_dimx_for_same_relation_vect.begin(),
      num_blocks_along_dimx_for_same_relation_vect.end());
  thrust::device_vector<int>
      dev_num_blocks_along_dimx_for_all_prev_relation_vect(
          num_blocks_along_dimx_for_all_prev_relation_vect.begin(),
          num_blocks_along_dimx_for_all_prev_relation_vect.end());
  return;
}

std::vector<std::vector<at::Tensor>> biops_tensor_info(
    at::Tensor& one_tensor, at::Tensor& other_tensor) {
  std::cout << "one_tensor device: " << one_tensor.device() << std::endl;
  std::cout << "other_tensor device: " << other_tensor.device() << std::endl;

  std::vector<std::vector<at::Tensor>> result = {
      {one_tensor.clone()}, {one_tensor.clone(), other_tensor.clone()}};
  return result;
}

at::Tensor tensor_info(at::Tensor& one_tensor) {
  // NB: storage_offset does play a role in tensor metadata, see in
  // github/pytorch/pytorch repo, pytorch/pytorch/c10/core/TensorImpl.h
  // implements `inline T* data_ptr_impl() const` as `return
  // storage_.unsafe_data<T>() + storage_offset_;`. Notice that storage_offset
  // count in number of elements, not bytes.
  std::cout << "library compiled by gcc " << __GNUC__ << "." << __GNUC_MINOR__
            << "." << __GNUC_PATCHLEVEL__ << ", nvcc " << __CUDACC_VER_MAJOR__
            << "." << __CUDACC_VER_MINOR__ << "." << __CUDACC_VER_BUILD__
            << std::endl;
  std::cout << "one_tensor device: " << one_tensor.device() << std::endl;
  std::cout << "one_tensor dtype: " << one_tensor.dtype() << std::endl;
  std::cout << "one_tensor ndim: " << one_tensor.dim() << std::endl;
  std::cout << "one_tensor shape: " << one_tensor.sizes() << std::endl;
  std::cout << "one_tensor numel: " << one_tensor.numel() << std::endl;
  std::cout << "one_tensor nbytes: " << one_tensor.nbytes() << std::endl;
  std::cout << "one_tensor storage_offset: " << one_tensor.storage_offset()
            << std::endl;
  std::cout << "one_tensor itemsize: " << one_tensor.element_size()
            << std::endl;
  return one_tensor.clone();
}

torch::Dict<std::string, int64_t> test_argument_takein(
    std::string str, bool flag, torch::Dict<std::string, int64_t> dictionary) {
  std::cout << "test_string_takein: " << str << std::endl;
  std::cout << "test_bool_takein: " << flag << std::endl;
  std::cout << "test_dict_takein: " << dictionary.at("key1") << std::endl;
  torch::Dict<std::string, int64_t> result;
  result.insert("key1", dictionary.at("key1") + 1);
  result.insert("flag", flag);
  return result;
}

TORCH_LIBRARY(torch_hetero_edgesoftmax, m) {
  // Utility and debugging functions
  m.def("biops_tensor_info", biops_tensor_info);
  m.def("tensor_info", tensor_info);
  m.def("try_get_schedule_by_relations", try_get_schedule_by_relations);
  // Data Converters
  m.def("transpose_csr", transpose_csr);
  m.def("convert_integrated_csr_to_separate_csr",
        convert_integrated_csr_to_separate_csr);
  m.def("convert_integrated_csr_to_separate_coo",
        convert_integrated_csr_to_separate_coo);
  m.def("convert_integrated_coo_to_separate_csr",
        convert_integrated_coo_to_separate_csr);
  m.def("convert_integrated_coo_to_separate_coo",
        convert_integrated_coo_to_separate_coo);
  m.def("test_argument_takein", test_argument_takein);
  // RGCN CSR Declaration
  m.def("rgcn_layer0_csr", RgcnLayer0Impl_wrapper_integratedcsr);
  m.def("rgcn_layer0_backward_csr",
        RgcnLayer0BackwardImpl_wrapper_integratedcsr);
  m.def("rgcn_layer1_csr", RgcnLayer1Impl_wrapper_integratedcsr);
  m.def("rgcn_layer1_backward_csr",
        RgcnLayer1BackwardImpl_wrapper_integratedcsr);
  // RGCN COO Declaration
  m.def("rgcn_layer1_coo", RgcnLayer1Impl_wrapper_integratedcoo);
  m.def("rgcn_layer1_backward_coo",
        RgcnLayer1BackwardImpl_wrapper_integratedcoo);
  // HGT CSR Declaration
  m.def("hgt_full_graph_message_mean_aggregation_backward_csr",
        hgt_full_graph_message_mean_aggregation_backward_wrapper_integratedcsr);
  m.def("hgt_full_graph_message_mean_aggregation_csr",
        hgt_full_graph_message_mean_aggregation_wrapper_integratedcsr);
  m.def("hgt_full_graph_hetero_attention_ops_backward_csr",
        hgt_full_graph_hetero_attention_ops_backward_wrapper_integratedcsr);
  m.def("hgt_full_graph_hetero_attention_ops_csr",
        hgt_full_graph_hetero_attention_ops_wrapper_integratedcsr);
  m.def("hgt_full_graph_hetero_message_ops_backward_csr",
        hgt_full_graph_hetero_message_ops_backward_wrapper_integratedcsr);
  m.def("hgt_full_graph_hetero_message_ops_csr",
        hgt_full_graph_hetero_message_ops_wrapper_integratedcsr);
  m.def("hgt_full_graph_edge_softmax_ops_csr",
        hgt_full_graph_edge_softmax_ops_wrapper_integratedcsr);
  m.def("hgt_full_graph_edge_softmax_ops_backward_csr",
        hgt_full_graph_edge_softmax_ops_backward_wrapper_integratedcsr);
  // Fused GAT CSR Declaration
  m.def("fused_gat_kernel_csr", FusedGatKernelImpl_wrapper_integratedcsr);
  m.def("backward_fused_gat_csr",
        BackwardFusedGatKernelImpl_wrapper_integratedcsr);
  // RGAT Declaration
  // RGAT Relational GEMM
  m.def("rgat_relational_matmul", RGATRelationalMatMul_wrapper_separatecoo);
  m.def("rgat_relational_matmul_backward",
        BackwardRGATRelationalMatMul_wrapper_separatecoo);
  m.def(
      "backward_rgat_relational_matmul_compact_as_of_node",
      BackwardRGATRelationalMatMulCompactAsOfNode_wrapper_unique_rel_node_indices);
  m.def("rgat_relational_matmul_compact_as_of_node",
        RGATRelationalMatMulCompactAsOfNode_wrapper_unique_rel_node_indices);

  // RGAT Relational SpMM
  m.def(
      "backward_rgat_relational_fused_gat_compact_as_of_node_edge_parallel_"
      "separate_coo",
      BackwardRGATRelationalFusedGATKernelCompactAsOfNode_wrapper_edge_parallel_separatecoo);
  m.def(
      "rgat_relational_fused_gat_compact_as_of_node_edge_parallel_separate_coo",
      RGATRelationalFusedGATKernelCompactAsOfNode_wrapper_edge_parallel_separatecoo);
  m.def("relational_fused_gat_kernel_edge_parallel_separate_coo",
        RelationalFusedGATKernel_wrapper_edge_parallel_separatecoo);
  m.def("backward_relational_fused_gat_edge_parallel_separate_coo",
        BackwardRelationalFusedGATKernel_wrapper_edge_parallel_separatecoo);
  m.def(
      "backward_rgat_relational_fused_gat_compact_as_of_node_csr",
      BackwardRGATRelationalFusedGATKernelCompactAsOfNode_wrapper_integratedcsr);
  m.def("rgat_relational_fused_gat_compact_as_of_node_csr",
        RGATRelationalFusedGATKernelCompactAsOfNode_wrapper_integratedcsr);
  m.def("relational_fused_gat_kernel_csr",
        RelationalFusedGATKernel_wrapper_integratedcsr);
  m.def("backward_relational_fused_gat_csr",
        BackwardRelationalFusedGATKernel_wrapper_integratedcsr);
}
