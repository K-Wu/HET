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

#include "DGLHackKernel/OpExport/HGTPrepToAndFromTensors.h"

// TODO: assume int32_t and float32 for now. but we may need to support other
// types
// TODO: check if torch builtin has the same encoding as int32_t and float32
#include "DGLHackKernel/OpExport/DataConverters.inc.h"
#include "DGLHackKernel/OpExport/GATOps.inc.h"
#include "DGLHackKernel/OpExport/HGTOps.inc.h"
#include "DGLHackKernel/OpExport/RGATOps.inc.h"
#include "DGLHackKernel/OpExport/RGCNCOOOps.inc.h"
#include "DGLHackKernel/OpExport/RGCNOps.inc.h"
#include "DGLHackKernel/OpExport/RGNNOps.inc.h"
#include "DGLHackKernel/OpExport/UtilityAndPlayground.inc.h"

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
  m.def("rgcn_layer1_coo", RGCN::FwProp::IntegratedCOO::Layer1Impl);
  m.def("rgcn_layer1_backward_coo",
        RGCN::BckProp::IntegratedCOO::Layer1BackwardImpl);
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
  m.def("fused_gat_kernel_csr", RGCN::FwProp::IntegratedCSR::FusedKernelImpl);
  m.def("backward_fused_gat_csr",
        RGCN::BckProp::IntegratedCSR::FusedKernelImpl);
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
