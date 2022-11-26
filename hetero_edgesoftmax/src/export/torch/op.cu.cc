// from
// https://stackoverflow.com/questions/68401650/how-can-i-make-a-pytorch-extension-with-cmake
// kernels defined in this file are simply wrapped up in
// hetero_edgesoftmax/python/kernels to provide python API, then used to
// define autograd functions and layers in hetero_edgesoftmax/python/kernels,
// which is finally referred to by end2end cases in
// hetero_edgesoftmax/python/<model name>/.*.py
// NB: This contains wrapper versions for python api export originally
// implemented at
// [[hetero_edgesoftmax/include/DGLHackKernel/RGCN/RGCNKernels.cu.h]].
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
#include "DGLHackKernel/OpExport/RGNNOps.inc.h"

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

#ifndef GIT_COMMIT_HASH
#define GIT_COMMIT_HASH "?"
#endif

void build_debug_info() {
  std::cout << "GIT_COMMIT_HASH: " << GIT_COMMIT_HASH << std::endl;
#ifdef ENABLE_DEBUG_MACRO
  std::cout << "WARNING: library built in debug mode without -O3" << std::endl;
#else
  std::cout << "library built in release mode with -O3" << std::endl;
#endif
  std::cout << "library compiled by gcc " << __GNUC__ << "." << __GNUC_MINOR__
            << "." << __GNUC_PATCHLEVEL__ << ", nvcc " << __CUDACC_VER_MAJOR__
            << "." << __CUDACC_VER_MINOR__ << "." << __CUDACC_VER_BUILD__
            << std::endl;
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

using namespace HET::TorchExport;
// NB: let's establish a convention of namespace hierarchy, i.e.,
// HET::TorchExport::ModelName::FwOrBwProp::FullGraphOrMinibatch::FormatSpecificationsEGIntegratedCSR::ComputeScheduleSpecificationsEGEdgeParallel::KernelName_VariantName
// In specific case where there is one level/field of namespace missing, e.g.,
// FullGraphOrMinibatch, we can just skip and move to the next inner level.
TORCH_LIBRARY(torch_hetero_edgesoftmax, m) {
  // Utility and debugging functions
  m.def("build_debug_info", build_debug_info);
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
  m.def("rgcn_layer0_csr", RGCN::FwProp::IntegratedCSR::Layer0Impl);
  m.def("rgcn_layer0_backward_csr", RGCN::BckProp::IntegratedCSR::Layer0Impl);
  m.def("rgcn_layer1_csr", RGCN::FwProp::IntegratedCSR::Layer1Impl);
  m.def("rgcn_layer1_backward_csr", RGCN::BckProp::IntegratedCSR::Layer1Impl);
  m.def("rgcn_layer0_csr_hybrid_assign",
        RGCN::FwProp::IntegratedCSR::Layer0HybridAssignmentImpl);
  m.def("rgcn_layer0_backward_csr_hybrid_assign",
        RGCN::BckProp::IntegratedCSR::Layer0HybridAssignmentImpl);
  m.def("rgcn_layer1_csr_hybrid_assign",
        RGCN::FwProp::IntegratedCSR::Layer1HybridAssignmentImpl);
  m.def("rgcn_layer1_backward_csr_hybrid_assign",
        RGCN::BckProp::IntegratedCSR::Layer1HybridAssignmentImpl);
  // RGCN COO Declaration
  m.def("rgcn_layer1_coo", RgcnLayer1Impl_wrapper_integratedcoo);
  m.def("rgcn_layer1_backward_coo",
        RgcnLayer1BackwardImpl_wrapper_integratedcoo);
  // HGT CSR Declaration
  m.def("hgt_full_graph_message_mean_aggregation_backward_csr",
        HGT::BckProp::IntegratedCSR::full_graph_message_mean_aggregation);
  m.def("hgt_full_graph_message_mean_aggregation_csr",
        HGT::FwProp::IntegratedCSR::full_graph_message_mean_aggregation);
  m.def("hgt_full_graph_hetero_attention_ops_backward_csr",
        HGT::BckProp::IntegratedCSR::full_graph_hetero_attention_ops);
  m.def("hgt_full_graph_hetero_attention_ops_csr",
        HGT::FwProp::IntegratedCSR::full_graph_hetero_attention_ops);
  m.def("hgt_full_graph_edge_softmax_ops_csr",
        HGT::FwProp::IntegratedCSR::full_graph_edge_softmax_ops);
  m.def("hgt_full_graph_edge_softmax_ops_backward_csr",
        HGT::BckProp::IntegratedCSR::full_graph_edge_softmax_ops);
  // Fused GAT CSR Declaration
  m.def("fused_gat_kernel_csr", FusedGatKernelImpl_wrapper_integratedcsr);
  m.def("backward_fused_gat_csr",
        BackwardFusedGatKernelImpl_wrapper_integratedcsr);
  // kernels for generic hetero-gnn use declaration
  // RGNN Relational GEMM
  m.def("rgnn_relational_matmul", RGNN::FwProp::RelationalMatMul_separatecoo);
  m.def("rgnn_relational_matmul_backward",
        RGNN::BckProp::RelationalMatMul_separatecoo);
  m.def(
      "rgnn_relational_matmul_ac_gather_scatter_list_identical",
      RGNN::FwProp::RelationalMatMul_ACGatherScatterListIdentical_separatecoo);
  m.def(
      "rgnn_relational_matmul_backward_ac_gather_scatter_list_identical",
      RGNN::BckProp::RelationalMatMul_ACGatherScatterListIdentical_separatecoo);
  m.def("backward_rgnn_relational_matmul_compact_as_of_node",
        RGNN::FwProp::RelationalMatMulCompactAsOfNode_unique_rel_node_indices);
  m.def("rgnn_relational_matmul_compact_as_of_node",
        RGNN::BckProp::RelationalMatMulCompactAsOfNode_unique_rel_node_indices);
  m.def(
      "rgnn_relational_matmul_compact_as_of_node_single_ended",
      RGNN::FwProp::
          RelationalMatMulCompactAsOfNodeSingleEnded_unique_rel_node_indices);  // args: unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_idx, separate_coo_rel_ptr, separate_coo_node_indices, weight, node_feat, ret,
  m.def(
      "backward_rgnn_relational_matmul_compact_as_of_node_single_ended",
      RGNN::BckProp::
          RelationalMatMulCompactAsOfNodeSingleEnded_unique_rel_node_indices);  // args: unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_idx, separate_coo_rel_ptr, separate_coo_node_indices, weight_transposed, node_feat, ret, gradout, grad_weight, grad_node_feat

  // RGNN innerproduct
  m.def(
      "rgnn_inner_product_node_compact_and_node",
      RGNN::FwProp::
          inner_product_node_compact_and_node_separatecoo);  // args:
                                                             // unique_srcs_and_dests_rel_ptr,
                                                             // unique_srcs_and_dests_node_idx,
                                                             // separate_coo_rel_ptr,
                                                             // separate_coo_eids,
                                                             // separate_coo_node_indices,
                                                             // left_node_compact_data,
                                                             // right_node_vectors,
                                                             // ret,
  m.def(
      "backward_rgnn_inner_product_node_compact_and_node",
      RGNN::BckProp::
          inner_product_node_compact_and_node_separatecoo);  // args:
                                                             // unique_srcs_and_dests_rel_ptr,
                                                             // unique_srcs_and_dests_node_idx,
                                                             // separate_coo_rel_ptr,
                                                             // separate_coo_eids,
                                                             // separate_coo_node_indices,
                                                             // left_node_compact_data,
                                                             // right_node_vectors,
                                                             // ret,
                                                             // gradout,
                                                             // grad_left_node_compact_data,
                                                             // grad_right_node_vectors
  m.def(
      "rgnn_inner_product_edge_and_node",
      RGNN::FwProp::
          inner_product_edge_and_node_separatecoo);  // args: separate_coo_eids,
                                                     // separate_coo_node_indices,
                                                     // left_edge_data,
                                                     // right_node_vectors, ret,
  m.def(
      "backward_rgnn_inner_product_edge_and_node",
      RGNN::BckProp::
          inner_product_edge_and_node_separatecoo);  // args:
                                                     // separate_coo_eids,separate_coo_node_indices,
                                                     // left_edge_data,
                                                     // right_node_vectors,
                                                     // ret, gradout,
                                                     // grad_left_edge_data,
                                                     // grad_right_node_vectors,

  // RGAT Declaration
  // RGAT Relational SpMM
  m.def(
      "backward_rgat_relational_fused_gat_compact_as_of_node_edge_parallel_"
      "separate_coo",
      RGAT::BckProp::
          RelationalFusedGATKernelCompactAsOfNode_edge_parallel_separatecoo);
  m.def(
      "rgat_relational_fused_gat_compact_as_of_node_edge_parallel_separate_coo",
      RGAT::FwProp::
          RelationalFusedGATKernelCompactAsOfNode_edge_parallel_separatecoo);
  m.def("relational_fused_gat_kernel_edge_parallel_separate_coo",
        RGAT::FwProp::RelationalFusedGATKernel_edge_parallel_separatecoo);
  m.def("backward_relational_fused_gat_edge_parallel_separate_coo",
        RGAT::BckProp::RelationalFusedGATKernel_edge_parallel_separatecoo);
  m.def("backward_rgat_relational_fused_gat_compact_as_of_node_csr",
        RGAT::BckProp::RelationalFusedGATKernelCompactAsOfNode_integratedcsr);
  m.def("rgat_relational_fused_gat_compact_as_of_node_csr",
        RGAT::FwProp::RelationalFusedGATKernelCompactAsOfNode_integratedcsr);
  m.def("relational_fused_gat_kernel_csr",
        RGAT::FwProp::RelationalFusedGATKernel_integratedcsr);
  m.def("backward_relational_fused_gat_csr",
        RGAT::BckProp::RelationalFusedGATKernel_integratedcsr);
}
