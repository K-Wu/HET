#pragma once
#include "DGLHackKernel/DGLHackKernel.h"
#include "DGLHackKernel/HGT/HGTPreprocessing.h"
#include "DGLHackKernel/NodeLinear.h"

// this structure also involves input and output data
struct HGTLayerIntermediateData {
  MySimpleNDArray<float, thrust::device_allocator<float>>
      KLinearOutput;  // [num_nodes, num_heads, klinear_out_dim]
  MySimpleNDArray<float, thrust::device_allocator<float>>
      QLinearOutput;  // [num_nodes, num_heads, qlinear_out_dim]
  MySimpleNDArray<float, thrust::device_allocator<float>>
      VLinearOutput;  // [num_nodes, num_heads, vlinear_out_dim]
  MySimpleNDArray<float, thrust::device_allocator<float>>
      ALinearOutput;  // [num_nodes, num_heads, alinear_out_dim]
  MySimpleNDArray<float, thrust::device_allocator<float>>
      ltsgemm_workspace;  // sweep or constant
  MySimpleNDArray<float, thrust::device_allocator<float>>
      NodeOutputFeatures;  // [num_nodes, num_heads, alinear_out_dim]
  MySimpleNDArray<float, thrust::device_allocator<float>>
      NodeInputFeatures;  // [num_nodes, num_heads, input_dim]
  MySimpleNDArray<float4, thrust::device_allocator<float4>>
      EdgeAttention;  // [num_edges, num_heads]
                      // TODO: now EdgeAttention only works for num_heads = 4
  size_t ltsgemm_workspaceSize;
  std::vector<thrust::device_vector<float>> intermediate_node_vect;
  HGTLayerIntermediateData(
      MySimpleNDArray<float, thrust::device_allocator<float>> &KLinearOutput,
      MySimpleNDArray<float, thrust::device_allocator<float>> &QLinearOutput,
      MySimpleNDArray<float, thrust::device_allocator<float>> &VLinearOutput,
      MySimpleNDArray<float, thrust::device_allocator<float>> &ALinearOutput,
      MySimpleNDArray<float, thrust::device_allocator<float>>
          &ltsgemm_workspace,
      MySimpleNDArray<float, thrust::device_allocator<float>>
          &NodeOutputFeatures,
      MySimpleNDArray<float, thrust::device_allocator<float>>
          &NodeInputFeatures,
      size_t ltsgemm_workspaceSize,
      std::vector<thrust::device_vector<float>> &intermediate_node_vect,
      MySimpleNDArray<float4, thrust::device_allocator<float4>> &EdgeAttention)
      : KLinearOutput(KLinearOutput),
        QLinearOutput(QLinearOutput),
        VLinearOutput(VLinearOutput),
        ALinearOutput(ALinearOutput),
        ltsgemm_workspace(ltsgemm_workspace),
        NodeOutputFeatures(NodeOutputFeatures),
        NodeInputFeatures(NodeInputFeatures),
        ltsgemm_workspaceSize(ltsgemm_workspaceSize),
        intermediate_node_vect(intermediate_node_vect),
        EdgeAttention(EdgeAttention) {
    intermediate_node_vect_d = GetDeviceVectorOfPointersToArrays<
        float, std::vector<thrust::device_vector<float>>>(
        intermediate_node_vect);
  }

  HGTLayerIntermediateData(HGTLayerIntermediateData &other)
      : KLinearOutput(other.KLinearOutput),
        QLinearOutput(other.QLinearOutput),
        VLinearOutput(other.VLinearOutput),
        ALinearOutput(other.ALinearOutput),
        ltsgemm_workspace(other.ltsgemm_workspace),
        NodeOutputFeatures(other.NodeOutputFeatures),
        NodeInputFeatures(other.NodeInputFeatures),
        ltsgemm_workspaceSize(other.ltsgemm_workspaceSize),
        intermediate_node_vect(other.intermediate_node_vect),
        EdgeAttention(other.EdgeAttention) {
    intermediate_node_vect_d = GetDeviceVectorOfPointersToArrays<
        float, std::vector<thrust::device_vector<float>>>(
        intermediate_node_vect);
  }
  HGTLayerIntermediateData(HGTLayerIntermediateData &&other)
      : KLinearOutput(std::move(other.KLinearOutput)),
        QLinearOutput(std::move(other.QLinearOutput)),
        VLinearOutput(std::move(other.VLinearOutput)),
        ALinearOutput(std::move(other.ALinearOutput)),
        ltsgemm_workspace(std::move(other.ltsgemm_workspace)),
        NodeOutputFeatures(std::move(other.NodeOutputFeatures)),
        NodeInputFeatures(std::move(other.NodeInputFeatures)),
        ltsgemm_workspaceSize(other.ltsgemm_workspaceSize),
        intermediate_node_vect(std::move(other.intermediate_node_vect)),
        EdgeAttention(std::move(other.EdgeAttention)) {
    intermediate_node_vect_d = GetDeviceVectorOfPointersToArrays<
        float, std::vector<thrust::device_vector<float>>>(
        intermediate_node_vect);
    other.intermediate_node_vect_d.clear();
  }

  void CheckPointerArrayValidity() const {
    assert(intermediate_node_vect_d.size() != 0);
  }
  thrust::device_vector<float *> &get_intermediate_node_vect_d() {
    CheckPointerArrayValidity();
    return intermediate_node_vect_d;
  }
  const thrust::device_vector<float *> &get_intermediate_node_vect_d() const {
    CheckPointerArrayValidity();
    return intermediate_node_vect_d;
  }

 private:
  thrust::device_vector<float *> intermediate_node_vect_d;
};

std::shared_ptr<HGTLayerIntermediateData>
CreateHGTLayerInputIntermediateOutputData(
    HGTLayerHyperParams hyper_params,
    std::shared_ptr<HGTLayerExecPreprocessedData> preprocessed_data) {
  // node_input_data element num: num_nodes * NUM_HEADS *
  // NODE_INPUT_DIM_PER_HEAD
  MySimpleNDArray<float, thrust::device_allocator<float>> KLinearOutput(
      {hyper_params.num_nodes, hyper_params.num_heads,
       hyper_params
           .klinear_out_dim});  // [num_nodes, num_heads, klinear_out_dim]
  MySimpleNDArray<float, thrust::device_allocator<float>> QLinearOutput(
      {hyper_params.num_nodes, hyper_params.num_heads,
       hyper_params
           .qlinear_out_dim});  // [num_nodes, num_heads, qlinear_out_dim]
  MySimpleNDArray<float, thrust::device_allocator<float>> VLinearOutput(
      {hyper_params.num_nodes, hyper_params.num_heads,
       hyper_params
           .vlinear_out_dim});  // [num_nodes, num_heads, vlinear_out_dim]
  MySimpleNDArray<float, thrust::device_allocator<float>> ALinearOutput(
      {hyper_params.num_nodes, hyper_params.num_heads,
       hyper_params
           .alinear_out_dim});  // [num_nodes, num_heads, alinear_out_dim]
  MySimpleNDArray<float, thrust::device_allocator<float>> ltsgemm_workspace(
      {hyper_params.PERF_ltsgemm_workspaceSize});
  MySimpleNDArray<float, thrust::device_allocator<float>> NodeOutputFeatures(
      {hyper_params.num_nodes, hyper_params.num_heads,
       hyper_params.alinear_out_dim});
  MySimpleNDArray<float, thrust::device_allocator<float>> NodeInputFeatures(
      {hyper_params.num_nodes, hyper_params.num_heads, hyper_params.input_dim});
  MySimpleNDArray<float4, thrust::device_allocator<float4>> EdgeAttention(
      {hyper_params.num_edges, hyper_params.num_heads});

  // fill in random data
  // TODO: we may fill in magic number data for debug purpose in future
  KLinearOutput.FillInRandomData();
  QLinearOutput.FillInRandomData();
  VLinearOutput.FillInRandomData();
  ALinearOutput.FillInRandomData();
  // we keep ltsgemm_workspace as is since it is reserved for cublaslt use.
  NodeOutputFeatures.FillInRandomData();
  NodeInputFeatures.FillInRandomData();

  // TODO: there might be alternatives i.e. storing (sW) or (Wt) in the
  // intermediate_node_vect
  // TODO: figuring out OUTDIM and what op this intermediate_node_vect is for
  // and add that to HGTLayerIntermediateData preparing intermediate data:
  // intermediate_node_vect_d
  std::vector<thrust::device_vector<float>> intermediate_node_vect(
      hyper_params.num_relations);  // need to persist (pointed by elements in
                                    // intermediate_node_vect_d)
  for (int idx_relation = 0; idx_relation < hyper_params.num_relations;
       idx_relation++) {
    intermediate_node_vect[idx_relation].resize(
        (preprocessed_data->coo_matrices_column_indices_unique)[idx_relation]
            .size() *
        hyper_params.qlinear_out_dim);
  }
  // TODO: set OUT_DIM to be one of the intermediate data dim for now, but
  // should correct it in future

  return std::make_shared<HGTLayerIntermediateData>(
      KLinearOutput, QLinearOutput, VLinearOutput, ALinearOutput,
      ltsgemm_workspace, NodeOutputFeatures, NodeInputFeatures,
      hyper_params.PERF_ltsgemm_workspaceSize, intermediate_node_vect,
      EdgeAttention);
}
