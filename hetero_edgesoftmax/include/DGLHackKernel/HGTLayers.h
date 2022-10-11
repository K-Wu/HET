#pragma once
#include "DGLHackKernel/DGLHackKernel.h"
#include "DGLHackKernel/NodeLinear.h"

template <class T, class OtherContainerType>
thrust::device_vector<T *> GetDeviceVectorOfPointersToArrays(
    OtherContainerType &other_container) {
  thrust::device_vector<T *> result;
  for (int idx_element = 0; idx_element < other_container.size();
       idx_element++) {
    result.push_back(
        thrust::raw_pointer_cast(other_container[idx_element].data()));
  }
  return result;
}

// TODO: we can reuse the first stage of sWt to calculate heterogensous message.
// The only difference is that heteroegneous message is associated with source
// node rather than destination node. The unique trick to reduce computation may
// still apply but the unique op is applied to the source node rather than
// destination node.
// TODO: For sWt, we may do the unique op on the fly: each warp is assigned a
// large chunk of consecutive source node index, and each warp work
// coorperatively to figure out the non-zero-outgoing-edge source node before
// calculating sW.

// For now, we put MySimpleNDArray as members of each of the structure. Whenever
// we want to reduce redundant deep copy, we may use std::shared_ptr as
// examplified in the answer here https://stackoverflow.com/a/395158 An example
// of initializing shared_ptr: https://godbolt.org/z/Yj86q3fEP backup:
// https://gist.github.com/K-Wu/141d949fd467ec7ff32e003ad0a5c5ce
struct HGTLayerExecPreprocessedData {
  std::vector<cusp::coo_matrix<int, int,
                               cusp::device_memory>::column_indices_array_type>
      coo_matrices_column_indices_unique;  // pointed to by elements of
                                           // unique_indices_to_column_indices_per_relation_d
  std::vector<thrust::device_vector<int>>
      dest_node_to_unique_index_per_relation;  // pointed to by eleemnts of
                                               // dest_node_to_unique_index_per_relation_d
  thrust::device_vector<int> num_unique_indices_to_column_indices_per_relation;

  HGTLayerExecPreprocessedData(
      std::vector<cusp::coo_matrix<
          int, int, cusp::device_memory>::column_indices_array_type>
          &coo_matrices_column_indices_unique,
      thrust::device_vector<int>
          &num_unique_indices_to_column_indices_per_relation,
      std::vector<thrust::device_vector<int>>
          &dest_node_to_unique_index_per_relation)
      : coo_matrices_column_indices_unique(coo_matrices_column_indices_unique),
        num_unique_indices_to_column_indices_per_relation(
            num_unique_indices_to_column_indices_per_relation),
        dest_node_to_unique_index_per_relation(
            dest_node_to_unique_index_per_relation) {
    unique_indices_to_column_indices_per_relation_d =
        GetDeviceVectorOfPointersToArrays<>(coo_matrices_column_indices_unique);
    dest_node_to_unique_index_per_relation_d =
        GetDeviceVectorOfPointersToArrays<>(
            dest_node_to_unique_index_per_relation);
  }

  HGTLayerExecPreprocessedData(HGTLayerExecPreprocessedData &other)
      : coo_matrices_column_indices_unique(
            other.coo_matrices_column_indices_unique),
        num_unique_indices_to_column_indices_per_relation(
            other.num_unique_indices_to_column_indices_per_relation),
        dest_node_to_unique_index_per_relation(
            other.dest_node_to_unique_index_per_relation) {
    unique_indices_to_column_indices_per_relation_d =
        GetDeviceVectorOfPointersToArrays<>(coo_matrices_column_indices_unique);
    dest_node_to_unique_index_per_relation_d =
        GetDeviceVectorOfPointersToArrays<>(
            dest_node_to_unique_index_per_relation);
  }

  HGTLayerExecPreprocessedData(HGTLayerExecPreprocessedData &&other)
      : coo_matrices_column_indices_unique(
            std::move(other.coo_matrices_column_indices_unique)),
        num_unique_indices_to_column_indices_per_relation(
            std::move(other.num_unique_indices_to_column_indices_per_relation)),
        dest_node_to_unique_index_per_relation(
            std::move(other.dest_node_to_unique_index_per_relation)) {
    unique_indices_to_column_indices_per_relation_d =
        GetDeviceVectorOfPointersToArrays<>(coo_matrices_column_indices_unique);
    dest_node_to_unique_index_per_relation_d =
        GetDeviceVectorOfPointersToArrays<>(
            dest_node_to_unique_index_per_relation);

    other.unique_indices_to_column_indices_per_relation_d.clear();
    other.dest_node_to_unique_index_per_relation_d.clear();
  }

  void CheckPointerArrayValidity() const {
    assert(unique_indices_to_column_indices_per_relation_d.size() != 0);
    assert(dest_node_to_unique_index_per_relation_d.size() != 0);
  }

  thrust::device_vector<int *>
      &get_unique_indices_to_column_indices_per_relation_d {
    CheckPointerArrayValidity();
    return unique_indices_to_column_indices_per_relation_d;
  }
  const thrust::device_vector<int *>
      &get_unique_indices_to_column_indices_per_relation_d const {
    CheckPointerArrayValidity();
    return unique_indices_to_column_indices_per_relation_d;
  }
  thrust::device_vector<int *> &get_dest_node_to_unique_index_per_relation_d {
    CheckPointerArrayValidity();
    return dest_node_to_unique_index_per_relation_d;
  }
  const thrust::device_vector<int *>
      &get_dest_node_to_unique_index_per_relation_d const {
    CheckPointerArrayValidity();
    return dest_node_to_unique_index_per_relation_d;
  }

 private:
  thrust::device_vector<int *> unique_indices_to_column_indices_per_relation_d;
  thrust::device_vector<int *> dest_node_to_unique_index_per_relation_d;
};

struct HGTLayerHyperParams {
  int num_relations;
  int num_node_types;
  int num_nodes;
  int num_edges;
  int input_dim;
  int klinear_out_dim;
  int qlinear_out_dim;
  int vlinear_out_dim;
  int message_dim;
  int alinear_out_dim;
  int num_heads;
  // the following are performance-specific parameters
  size_t PERF_ltsgemm_workspaceSize;
  // TODO: implement it once it is finished
  HGTLayerHyperParams(int num_relations, int num_node_types, int num_nodes,
                      int num_edges, int input_dim, int klinear_out_dim,
                      int qlinear_out_dim, int vlinear_out_dim, int message_dim,
                      int alinear_out_dim, int num_heads)
      : num_relations(num_relations),
        num_node_types(num_node_types),
        num_nodes(num_nodes),
        num_edges(num_edges),
        input_dim(input_dim),
        klinear_out_dim(klinear_out_dim),
        qlinear_out_dim(qlinear_out_dim),
        vlinear_out_dim(vlinear_out_dim),
        message_dim(message_dim),
        alinear_out_dim(alinear_out_dim),
        num_heads(num_heads),
        PERF_ltsgemm_workspaceSize(PERF_ltsgemm_workspaceSize) {}

} std::string getStrRepr() const {
  return "HGTLayerHyperParams num_relations: " + std::to_string(num_relations) +
         " num_node_types: " + std::to_string(num_node_types) +
         " num_nodes: " + std::to_string(num_nodes) +
         " num_edges: " + std::to_string(num_edges) +
         " input_dim: " + std::to_string(input_dim) +
         " klinear_out_dim: " + std::to_string(klinear_out_dim) +
         " qlinear_out_dim: " + std::to_string(qlinear_out_dim) +
         " vlinear_out_dim: " + std::to_string(vlinear_out_dim) +
         " message_dim: " + std::to_string(message_dim) +
         " alinear_out_dim: " + std::to_string(alinear_out_dim) +
         " num_heads: " + std::to_string(num_heads) +
         " PERF_ltsgemm_workspaceSize: " +
         std::to_string(PERF_ltsgemm_workspaceSize);
}
}
;

// this structure also involves input and output data
struct HGTLayerIntermediateData {
  MySimpleNDArray<DType, thrust::device_allocator<DType>>
      KLinearOutput;  // [num_nodes, num_heads, klinear_out_dim]
  MySimpleNDArray<DType, thrust::device_allocator<DType>>
      QLinearOutput;  // [num_nodes, num_heads, qlinear_out_dim]
  MySimpleNDArray<DType, thrust::device_allocator<DType>>
      VLinearOutput;  // [num_nodes, num_heads, vlinear_out_dim]
  MySimpleNDArray<DType, thrust::device_allocator<DType>>
      ALinearOutput;  // [num_nodes, num_heads, alinear_out_dim]
  MySimpleNDArray<DType, thrust::device_allocator<DType>>
      ltsgemm_workspace;  // sweep or constant
  MySimpleNDArray<DType, thrust::device_allocator<DType>>
      NodeOutputFeatures;  // [num_nodes, num_heads, alinear_out_dim]
  MySimpleNDArray<DType, thrust::device_allocator<DType>>
      NodeInputFeatures;  // [num_nodes, num_heads, input_dim]
  size_t ltsgemm_workspaceSize;
  std::vector<thrust::device_vector<float>> intermediate_node_vect;
  HGTLayerIntermediateData(
      MySimpleNDArray<DType, thrust::device_allocator<DType>> &KLinearOutput,
      MySimpleNDArray<DType, thrust::device_allocator<DType>> &QLinearOutput,
      MySimpleNDArray<DType, thrust::device_allocator<DType>> &VLinearOutput,
      MySimpleNDArray<DType, thrust::device_allocator<DType>> &ALinearOutput,
      MySimpleNDArray<DType, thrust::device_allocator<DType>>
          &ltsgemm_workspace,
      MySimpleNDArray<DType, thrust::device_allocator<DType>>
          &NodeOutputFeatures,
      MySimpleNDArray<DType, thrust::device_allocator<DType>>
          &NodeInputFeatures,
      size_t ltsgemm_workspaceSize,
      std::vector<thrust::device_vector<float>> &intermediate_node_vect)
      : KLinearOutput(KLinearOutput),
        QLinearOutput(QLinearOutput),
        VLinearOutput(VLinearOutput),
        ALinearOutput(ALinearOutput),
        ltsgemm_workspace(ltsgemm_workspace),
        NodeOutputFeatures(NodeOutputFeatures),
        NodeInputFeatures(NodeInputFeatures),
        ltsgemm_workspaceSize(ltsgemm_workspaceSize),
        intermediate_node_vect(intermediate_node_vect) {
    intermediate_node_vect_d =
        GetDeviceVectorOfPointersToArrays<>(intermediate_node_vect);
  }

  HGTLayerIntermediateData(HGTLayerIntermediateData &other)
      : KLinearOutput(other.KLinearOutput),
        QLinearOutput(other.QLinearOutput),
        VLinearOutput(other.VLinearOutput),
        ALinearOutput(other.ALinearOutput),
        ltsgemm_workspace(other.ltsgemm_workspace),
        NodeOutputFeatures(NodeOutputFeatures),
        NodeInputFeatures(NodeInputFeatures),
        ltsgemm_workspaceSize(other.ltsgemm_workspaceSize),
        intermediate_node_vect(other.intermediate_node_vect) {
    intermediate_node_vect_d =
        GetDeviceVectorOfPointersToArrays<>(intermediate_node_vect);
  }
  HGTLayerIntermediateData(HGTLayerIntermediateData &&other)
      : KLinearOutput(std::move(other.KLinearOutput)),
        QLinearOutput(std::move(other.QLinearOutput)),
        VLinearOutput(std::move(other.VLinearOutput)),
        ALinearOutput(std::move(other.ALinearOutput)),
        ltsgemm_workspace(std::move(other.ltsgemm_workspace)),
        NodeOutputFeatures(NodeOutputFeatures),
        NodeInputFeatures(NodeInputFeatures),
        ltsgemm_workspaceSize(other.ltsgemm_workspaceSize),
        intermediate_node_vect(std::move(other.intermediate_node_vect)) {
    intermediate_node_vect_d =
        GetDeviceVectorOfPointersToArrays<>(intermediate_node_vect);
    other.intermediate_node_vect_d.clear();
  }

  void CheckPointerArrayValidity() const {
    assert(intermediate_node_vect_d.size() != 0);
  }
  thrust::device_vector<float *> &get_intermediate_node_vect_d {
    CheckPointerArrayValidity();
    return intermediate_node_vect_d;
  }
  const thrust::device_vector<float *> &get_intermediate_node_vect_d const {
    CheckPointerArrayValidity();
    return intermediate_node_vect_d;
  }

 private:
  thrust::device_vector<float *> intermediate_node_vect_d;
};

struct HGTLayerWeights {
  MySimpleNDArray<DType, thrust::device_allocator<DType>> KLinearWeights;
  MySimpleNDArray<DType, thrust::device_allocator<DType>> KLinearBias;
  MySimpleNDArray<DType, thrust::device_allocator<DType>> QLinearWeights;
  MySimpleNDArray<DType, thrust::device_allocator<DType>> QLinearBias;
  MySimpleNDArray<DType, thrust::device_allocator<DType>> VLinearWeights;
  MySimpleNDArray<DType, thrust::device_allocator<DType>> VLinearBias;
  MySimpleNDArray<DType, thrust::device_allocator<DType>> ALinearWeights;
  MySimpleNDArray<DType, thrust::device_allocator<DType>> ALinearBias;
  MySimpleNDArray<DType, thrust::device_allocator<DType>>
      relation_attention_matrices;
  MySimpleNDArray<DType, thrust::device_allocator<DType>>
      relation_message_matrices;
    HGTLayerWeights(MySimpleNDArray<DType, thrust::device_allocator<DType>>& KLinearWeights,
                    MySimpleNDArray<DType, thrust::device_allocator<DType>>& KLinearBias,
                    MySimpleNDArray<DType, thrust::device_allocator<DType>>& QLinearWeights,
                    MySimpleNDArray<DType, thrust::device_allocator<DType>>& QLinearBias,
                    MySimpleNDArray<DType, thrust::device_allocator<DType>>& VLinearWeights,
                    MySimpleNDArray<DType, thrust::device_allocator<DType>>& VLinearBias,
                    MySimpleNDArray<DType, thrust::device_allocator<DType>>& ALinearWeights,
                    MySimpleNDArray<DType, thrust::device_allocator<DType>>& ALinearBias,
                    MySimpleNDArray<DType, thrust::device_allocator<DType>>& relation_attention_matrices,
                    MySimpleNDArray<DType, thrust::device_allocator<DType>>& relation_message_matrices)):
        KLinearWeights(KLinearWeights),
        KLinearBias(KLinearBias),
        QLinearWeights(QLinearWeights),
        QLinearBias(QLinearBias),
        VLinearWeights(VLinearWeights),
        VLinearBias(VLinearBias),
        ALinearWeights(ALinearWeights),
        ALinearBias(ALinearBias),
        relation_attention_matrices(relation_attention_matrices),
        relation_message_matrices(relation_message_matrices)
    {}
};

std::shared_ptr<HGTLayerWeights> InitializeHGTLayerWeights(
    HGTLayerHyperParams hyper_params) {
  // TODO: implement move constructor for either HGTLayerWeights or
  // MySimpleNDArray to reduce memory footprint during initialization.
  MySimpleNDArray<DType, thrust::device_allocator<DType>> KLinearWeights =
      MySimpleNDArray<DType, thrust::device_allocator<DType>>(
          {hyper_params.num_node_types, hyper_params.num_heads,
           hyper_params.input_dim, hyper_params.klinear_out_dim});
  MySimpleNDArray<DType, thrust::device_allocator<DType>> KLinearBias =
      MySimpleNDArray<DType, thrust::device_allocator<DType>>(
          {hyper_params.num_node_types, hyper_params.num_heads,
           hyper_params.klinear_out_dim});
  MySimpleNDArray<DType, thrust::device_allocator<DType>> QLinearWeights =
      MySimpleNDArray<DType, thrust::device_allocator<DType>>(
          {hyper_params.num_node_types, hyper_params.num_heads,
           hyper_params.input_dim, hyper_params.qlinear_out_dim});
  MySimpleNDArray<DType, thrust::device_allocator<DType>> QLinearBias =
      MySimpleNDArray<DType, thrust::device_allocator<DType>>(
          {hyper_params.num_node_types, hyper_params.num_heads,
           hyper_params.qlinear_out_dim});
  MySimpleNDArray<DType, thrust::device_allocator<DType>> VLinearWeights =
      MySimpleNDArray<DType, thrust::device_allocator<DType>>(
          {hyper_params.num_node_types, hyper_params.num_heads,
           hyper_params.input_dim, hyper_params.vlinear_out_dim});
  MySimpleNDArray<DType, thrust::device_allocator<DType>> VLinearBias =
      MySimpleNDArray<DType, thrust::device_allocator<DType>>(
          {hyper_params.num_node_types, hyper_params.num_heads,
           hyper_params.vlinear_out_dim});
  // float *node_input_data; element num: num_nodes * NUM_HEADS *
  // NODE_INPUT_DIM_PER_HEAD float *relation_attention_matrices; element num:
  // num_relations * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD *
  // NODE_INPUT_DIM_PER_HEAD
  MySimpleNDArray<DType, thrust::device_allocator<DType>>
      relation_attention_matrices =
          MySimpleNDArray<DType, thrust::device_allocator<DType>>(
              {hyper_params.num_relations, hyper_params.num_heads,
               hyper_params.klinear_out_dim, hyper_params.qlinear_out_dim});
  MySimpleNDArray<DType, thrust::device_allocator<DType>>
      relation_message_matrices =
          MySimpleNDArray<DType, thrust::device_allocator<DType>>(
              {hyper_params.num_relations, hyper_params.num_heads,
               hyper_params.vlinear_out_dim, hyper_params.message_dim});
  MySimpleNDArray<DType, thrust::device_allocator<DType>> ALinearWeights =
      MySimpleNDArray<DType, thrust::device_allocator<DType>>(
          {hyper_params.num_heads, hyper_params.message_dim,
           hyper_params.alinear_out_dim});
  MySimpleNDArray<DType, thrust::device_allocator<DType>> ALinearBias =
      MySimpleNDArray<DType, thrust::device_allocator<DType>>(
          {hyper_params.num_heads, hyper_params.alinear_out_dim});
  KLinearWeights.FillInRandomData();
  KLinearBias.FillInRandomData();
  QLinearWeights.FillInRandomData();
  QLinearBias.FillInRandomData();
  VLinearWeights.FillInRandomData();
  VLinearBias.FillInRandomData();
  ALinearWeights.FillInRandomData();
  ALinearBias.FillInRandomData();
  relation_attention_matrices.FillInRandomData();
  relation_message_matrices.FillInRandomData();
  return std::make_shared<HGTLayerWeights>(
      KLinearWeights, KLinearBias, QLinearWeights, QLinearBias, VLinearWeights,
      VLinearBias, ALinearWeights, ALinearBias, relation_attention_matrices,
      relation_message_matrices);
}

std::shared_ptr<HGTLayerIntermediateData>
CreateHGTLayerInputIntermediateOutputData(HGTLayerHyperParams hyper_params) {
  // node_input_data element num: num_nodes * NUM_HEADS *
  // NODE_INPUT_DIM_PER_HEAD
  MySimpleNDArray<DType, thrust::device_allocator<DType>> KLinearOutput(
      {hyper_params.num_nodes, hyper_params.num_heads,
       hyper_params
           .klinear_out_dim});  // [num_nodes, num_heads, klinear_out_dim]
  MySimpleNDArray<DType, thrust::device_allocator<DType>> QLinearOutput(
      {hyper_params.num_nodes, hyper_params.num_heads,
       hyper_params
           .qlinear_out_dim});  // [num_nodes, num_heads, qlinear_out_dim]
  MySimpleNDArray<DType, thrust::device_allocator<DType>> VLinearOutput(
      {hyper_params.num_nodes, hyper_params.num_heads,
       hyper_params
           .vlinear_out_dim});  // [num_nodes, num_heads, vlinear_out_dim]
  MySimpleNDArray<DType, thrust::device_allocator<DType>> ALinearOutput(
      {hyper_params.num_nodes, hyper_params.num_heads,
       hyper_params
           .alinear_out_dim});  // [num_nodes, num_heads, alinear_out_dim]
  MySimpleNDArray<DType, thrust::device_allocator<DType>> ltsgemm_workspace(
      {hyper_params.PERF_ltsgemm_workspaceSize});
  MySimpleNDArray<DType, thrust::device_allocator<DType>> NodeOutputFeatures(
      {hyper_params.num_nodes, hyper_params.num_heads,
       hyper_params.alinear_out_dim});
  MySimpleNDArray<DType, thrust::device_allocator<DType>> NodeInputFeatures(
      {hyper_params.num_nodes, hyper_params.num_heads, hyper_params.input_dim});

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
        coo_matrices_column_indices_unique[idx_relation].size() * OUT_DIM);
  }

  return std::make_shared<HGTLayerIntermediateData>(
      KLinearOutput, QLinearOutput, VLinearOutput, ALinearOutput,
      ltsgemm_workspace, NodeOutputFeatures, NodeInputFeatures,
      hyper_params.PERF_ltsgemm_workspaceSize, intermediate_node_vect);
}

// preprocessing also generates primitive array of pointers for CUDA kernel to
// access
std::shared_ptr<HGTLayerExecPreprocessedData> HGTLayerPreprocessing(
    HGTLayerHyperParams hyper_params,
    cusp::coo_matrix<int, int, cusp::device_memory>::row_indices_array_type
        concatenated_coo_matrix_row_indices,
    cusp::coo_matrix<int, int, cusp::device_memory>::column_indices_array_type
        concatenated_coo_matrix_column_indices,
    std::vector<cusp::coo_matrix<
        int, int, cusp::device_memory>::column_indices_array_type>
        coo_matrices_column_indices,
    cusp::coo_matrix<int, int, cusp::device_memory>::values_array_type
        concatenated_coo_matrix_values) {
  constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
  constexpr int COARSE_SGEMM_BLOCKSIZE = (TILE_SZ_A);
  constexpr int COARSE_SGEMM_NODES_PER_BLOCK = (TILE_SZ_B);

  // generating preprocessed metadata:
  // dest_node_to_unique_index_per_relation_d,
  // unique_indices_to_column_indices_per_relation_d,
  // num_unique_indices_to_column_indices_per_relation
  std::vector<cusp::coo_matrix<int, int,
                               cusp::device_memory>::column_indices_array_type>
      coo_matrices_column_indices_unique(
          hyper_params
              .num_relations);  // need to persist (pointed by elements in
                                // unique_indices_to_column_indices_per_relation_d)
  thrust::device_vector<int *>
      unique_indices_to_column_indices_per_relation_d;  // need to output
  thrust::device_vector<int> num_unique_indices_to_column_indices_per_relation(
      hyper_params.num_relations, -1);  // need to output

  std::vector<thrust::device_vector<int>>
      dest_node_to_unique_index_per_relation =
          std::vector<thrust::device_vector<int>>(
              hyper_params.num_relations,
              thrust::device_vector<int>(
                  hyper_params.num_nodes,
                  -1));  // need to persist (pointed by elements in
                         // dest_node_to_unique_index_per_relation_d)
  thrust::device_vector<int *>
      dest_node_to_unique_index_per_relation_d;  // need to output

  // TODO: dedicated function to generate array of (pointers to arrays)
  for (int idx_relation = 0; idx_relation < hyper_params.num_relations;
       idx_relation++) {
    dest_node_to_unique_index_per_relation_d.push_back(thrust::raw_pointer_cast(
        dest_node_to_unique_index_per_relation[idx_relation].data()));
  }

  for (int idx_relation = 0; idx_relation < hyper_params.num_relations;
       idx_relation++) {
    coo_matrices_column_indices_unique[idx_relation] =
        coo_matrices_column_indices[idx_relation];
    thrust::sort(thrust::device,
                 coo_matrices_column_indices_unique[idx_relation].begin(),
                 coo_matrices_column_indices_unique[idx_relation].end());
    auto curr_unique_vector_end =
        thrust::unique(thrust::device,
                       coo_matrices_column_indices_unique[idx_relation].begin(),
                       coo_matrices_column_indices_unique[idx_relation].end());
    coo_matrices_column_indices_unique[idx_relation].resize(thrust::distance(
        coo_matrices_column_indices_unique[idx_relation].begin(),
        curr_unique_vector_end));
  }

  for (int idx_relation = 0; idx_relation < hyper_params.num_relations;
       idx_relation++) {
    thrust::counting_iterator<int> first_counting_iter(0);
    thrust::counting_iterator<int> last_counting_iter =
        first_counting_iter +
        coo_matrices_column_indices_unique[idx_relation].size();

    int *curr_dest_node_to_unique_index_per_relation_d =
        dest_node_to_unique_index_per_relation_d[idx_relation];
    thrust::for_each(
        thrust::device,
        thrust::make_zip_iterator(thrust::make_tuple(
            coo_matrices_column_indices_unique[idx_relation].begin(),
            first_counting_iter)),
        thrust::make_zip_iterator(thrust::make_tuple(
            coo_matrices_column_indices_unique[idx_relation].end(),
            last_counting_iter)),
        [=] __host__ __device__(thrust::tuple<int, int> t) {
          curr_dest_node_to_unique_index_per_relation_d[thrust::get<0>(t)] =
              thrust::get<1>(t);
        });
  }

  for (int idx_relation = 0; idx_relation < hyper_params.num_relations;
       idx_relation++) {
    num_unique_indices_to_column_indices_per_relation[idx_relation] =
        coo_matrices_column_indices_unique[idx_relation].size();
  }

  return std::make_shared<HGTLayerExecPreprocessedData>(
      coo_matrices_column_indices_unique,
      num_unique_indices_to_column_indices_per_relation,
      dest_node_to_unique_index_per_relation);
}

// extract this kernel with mysgemm_ into template specialization
// template <int NODE_INPUT_DIM_PER_HEAD/*derived from OUT_DIM and NUM_HEADS*/,
// NUM_HEADS, OUT_DIM, COARSE_SGEMM_NODES_PER_BLOCK>
template <int TILE_SZ_A, int TILE_SZ_B, int OUT_DIM, int NUM_HEADS>
__global__ void _global_EdgeMessageConcatenatedCOOKernel(
    float **__restrict__ intermediate_node_vect, int nnz,
    int *__restrict__ matCols, int *__restrict__ matRelation,
    float *__restrict__ node_input_data,
    float *__restrict__ relation_message_matrices,
    int **__restrict__ dest_node_to_unique_index_per_relation,
    int *__restrict__ sizes_unique_index_to_dest_node_per_relation,
    int num_relations,
    int *__restrict__ num_blocks_xdim_for_same_relation_per_block_vect,
    int *__restrict__ beg_node_entry_idxes_vect,
    int *__restrict__ blockid_relation_id_vect) {
  constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
  constexpr int COARSE_SGEMM_NODES_PER_BLOCK = (TILE_SZ_B);
  int beg_node_entry_idx = beg_node_entry_idxes_vect[blockIdx.x];
  int stride = num_blocks_xdim_for_same_relation_per_block_vect[blockIdx.x] *
               COARSE_SGEMM_NODES_PER_BLOCK;
  int relation_idx = blockid_relation_id_vect[blockIdx.x];

  for (int node_entry_idx = beg_node_entry_idx;
       node_entry_idx <
       sizes_unique_index_to_dest_node_per_relation[relation_idx];
       node_entry_idx += stride) {
    mysgemm_functor<TILE_SZ_A, TILE_SZ_B, OUT_DIM, NUM_HEADS, false>::
        exec_function(
            OUT_DIM, sizes_unique_index_to_dest_node_per_relation[relation_idx],
            NODE_INPUT_DIM_PER_HEAD,
            &relation_message_matrices[relation_idx * NUM_HEADS *
                                       NODE_INPUT_DIM_PER_HEAD *
                                       NODE_INPUT_DIM_PER_HEAD],
            node_input_data, intermediate_node_vect[relation_idx], nullptr,
            node_entry_idx);
  }
}

// TODO: collect input data into a struct; malloc intermediate and output data.

// assume nodes indices are currently sorted according to their node type, or
// even only monotype exists. We use naive for loop outside kernel launch to do
// the linear layer for now.
// TODO: implement more general case where nodes may not be sorted according to
// node type, thus indirection needed
// TODO: optimize the for loop by fusing multiple kernels into one
// work for both k-linear and q-linear
void LinearByNodeType(float *Adev /*weight*/, float *Bdev /*input*/,
                      float *Cdev /*output*/, float *biasDev /*bias*/,
                      int m /*hidden dim*/, int n /*num nodes*/,
                      int k /*hidden dim*/, int num_heads, void *workspace,
                      size_t workspaceSize) {
  // weight, bias, input
  // TODO: check if we can initialize ltHandle once per program run or once per
  // layer
  cublasLtHandle_t ltHandle;
  float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasLtCreate(&ltHandle));
  for (int idx_head = 0; idx_head < num_heads; idx_head += 1) {
    LtSgemm(ltHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
            &Adev[m * n * idx_head], m * num_heads, &Bdev[k * idx_head],
            k * num_heads, &beta, &Cdev[m * idx_head], m * num_heads,
            &biasDev[m * idx_head], workspace, workspaceSize);
  }
  CUBLAS_CHECK(cublasLtDestroy(ltHandle));
}

void KLinearByNodeType(
    HGTLayerHyperParams hyper_params, std::shared_ptr<HGTLayerWeights> weights,
    std::shared_ptr<HGTLayerIntermediateData> intermediate_data) {
  std::cout << "KLinearByNodeType ptrs"
            << (intermediate_data->NodeInputFeatures).Ptr() << " "
            << (intermediate_data->KLinearOutput).Ptr() << " "
            << (weights->KLinearBias).Ptr() << " "
            << (weights->KLinearWeight).Ptr() << std::endl;
  LinearByNodeType((weights->KLinearWeight).Ptr(),
                   (intermediate_data->NodeInputFeatures).Ptr(),
                   (intermediate_data->KLinearOutput).Ptr(),
                   (weights->KLinearBias).Ptr(), hyper_params.klinear_out_dim,
                   hyper_params.num_nodes, hyper_params.input_dim,
                   hyper_params.num_heads,
                   (intermediate_data->ltsgemm_workspace).Ptr(),
                   intermediate_data->ltsgemm_workspaceSize);
}

void QLinearByNodeType(
    HGTLayerHyperParams hyper_params, std::shared_ptr<HGTLayerWeights> weights,
    std::shared_ptr<HGTLayerIntermediateData> intermediate_data) {
  // TODO: check if we need to assign separate workspace to each of the linear
  // layer
  std::cout << "QLinearByNodeType ptrs"
            << (intermediate_data->NodeInputFeatures).Ptr() << " "
            << (intermediate_data->QLinearOutput).Ptr() << " "
            << (weights->QLinearBias).Ptr() << " "
            << (weights->QLinearWeight).Ptr() << std::endl;
  LinearByNodeType((weights->QLinearWeight).Ptr(),
                   (intermediate_data->NodeInputFeatures).Ptr(),
                   (intermediate_data->QLinearOutput).Ptr(),
                   (weights->QLinearBias).Ptr(), hyper_params.qlinear_out_dim,
                   hyper_params.num_nodes, hyper_params.input_dim,
                   hyper_params.num_heads,
                   (intermediate_data->ltsgemm_workspace).Ptr(),
                   intermediate_data->ltsgemm_workspaceSize);
}

void VLinearByNodeType(
    HGTLayerHyperParams hyper_params, std::shared_ptr<HGTLayerWeights> weights,
    std::shared_ptr<HGTLayerIntermediateData> intermediate_data) {
  std::cout << "VLinearByNodeType ptrs"
            << (intermediate_data->NodeInputFeatures).Ptr() << " "
            << (intermediate_data->VLinearOutput).Ptr() << " "
            << (weights->VLinearBias).Ptr() << " "
            << (weights->VLinearWeight).Ptr() << std::endl;
  LinearByNodeType((weights->VLinearWeight).Ptr(),
                   (intermediate_data->NodeInputFeatures).Ptr(),
                   (intermediate_data->VLinearOutput).Ptr(),
                   (weights->VLinearBias).Ptr(), hyper_params.vlinear_out_dim,
                   hyper_params.num_nodes, hyper_params.input_dim,
                   hyper_params.num_heads,
                   (intermediate_data->ltsgemm_workspace).Ptr(),
                   intermediate_data->ltsgemm_workspaceSize);
}

void ALinearByNodeType(
    HGTLayerHyperParams hyper_params, std::shared_ptr<HGTLayerWeights> weights,
    std::shared_ptr<HGTLayerIntermediateData> intermediate_data) {
  // TODO: the input to ALinear should be intermediate data
  std::cout << "ALinearByNodeType ptrs"
            << (intermediate_data->NodeInputFeatures).Ptr() << " "
            << (intermediate_data->ALinearOutput).Ptr() << " "
            << (weights->ALinearBias).Ptr() << " "
            << (weights->ALinearWeight).Ptr() << std::endl;
  LinearByNodeType((weights->ALinearWeight).Ptr(),
                   (intermediate_data->NodeInputFeatures).Ptr(),
                   (intermediate_data->ALinearOutput).Ptr(),
                   (weights->ALinearBias).Ptr(), hyper_params.alinear_out_dim,
                   hyper_params.num_nodes, hyper_params.input_dim,
                   hyper_params.num_heads,
                   (intermediate_data->ltsgemm_workspace).Ptr(),
                   intermediate_data->ltsgemm_workspaceSize);
}

// NB: In this implementation, message generation is done for each (source node,
// relationship this node is involved) where each (source node, relationship
// this node is involved) is mapped to a unique (relationship id, unique node
// index) and referred to in the next stage. Notice getting this unique index
// mapping is O(|R||V|) complexity and stays the same throughout the whole
// execution. We can do this mapping in the first step and reuse it thereafter.
// In this case, the it is dense operation first with scatter operation
// implicitly done by succeeding operations.
// TODO: an alternative implementation is message generation for each edge where
// there might be redundant computation of (source node, relationship this node
// is involved) pairs. In this case, only the relationship type and source node
// index for each edge is needed. This is explicit scatter operation done first
// and then dense operation.
template <int TILE_SZ_A /*128*/, int TILE_SZ_B /*8*/, int OUT_DIM /*256*/,
          int NUM_HEADS /*4*/>
void EdgeMessageConcatenatedCOOKernel(
    HGTLayerHyperParams hyper_params, std::shared_ptr<HGTLayerWeights> weights,
    std::shared_ptr<HGTLayerIntermediateData> intermediate_data,
    std::shared_ptr<HGTLayerExecPreprocessedData> preprocessed_data,
    cusp::coo_matrix<int, int, cusp::device_memory>::row_indices_array_type
        concatenated_coo_matrix_row_indices,
    cusp::coo_matrix<int, int, cusp::device_memory>::column_indices_array_type
        concatenated_coo_matrix_column_indices,
    std::vector<cusp::coo_matrix<
        int, int, cusp::device_memory>::column_indices_array_type>
        coo_matrices_column_indices,
    cusp::coo_matrix<int, int, cusp::device_memory>::values_array_type
        concatenated_coo_matrix_values) {
  constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
  constexpr int COARSE_SGEMM_BLOCKSIZE = (TILE_SZ_A);
  constexpr int COARSE_SGEMM_NODES_PER_BLOCK = (TILE_SZ_B);

  // float *node_input_data element num: num_nodes * NUM_HEADS *
  // NODE_INPUT_DIM_PER_HEAD float *relation_attention_matrices element num:
  // num_relations * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD *
  // NODE_INPUT_DIM_PER_HEAD

  // preparing op kernel launch specific preprocessed metadata:
  // num_blocks_xdim_for_same_relation_per_block_vect,
  // beg_node_entry_idxes_vect, blockid_relation_id_vect
  thrust::device_vector<int> num_blocks_xdim_for_same_relation_per_block_vect;
  thrust::device_vector<int> blockid_relation_id_vect;
  thrust::device_vector<int> beg_node_entry_idxes_vect;
  std::vector<int> num_blocks_xdim_for_same_relation_vect;
  std::vector<int> num_blocks_xdim_for_all_prev_relation_vect;
  num_blocks_xdim_for_all_prev_relation_vect.push_back(0);

  // for ease of programming equally partition the workload to different blocks
  // at this moment.
  for (int idx_relationship = 0; idx_relationship < hyper_params.num_relations;
       idx_relationship++) {
    int num_blocks_xdim_for_this_and_prev_relation =
        (idx_relationship + 1 + 0.0) / (hyper_params.num_relations + 0.0) *
        RTX_3090_GRIDSIZE;
    num_blocks_xdim_for_all_prev_relation_vect.push_back(
        num_blocks_xdim_for_this_and_prev_relation);
  }
  for (int idx_relationship = 0; idx_relationship < hyper_params.num_relations;
       idx_relationship++) {
    num_blocks_xdim_for_same_relation_vect.push_back(
        num_blocks_xdim_for_all_prev_relation_vect[idx_relationship + 1] -
        num_blocks_xdim_for_all_prev_relation_vect[idx_relationship]);
  }
  num_blocks_xdim_for_all_prev_relation_vect.erase(
      num_blocks_xdim_for_all_prev_relation_vect.begin());
  int idx_curr_relation = 0;
  int curr_beg_node_entry_idx = 0;

  // grid and thread configuration of the first stage
  //   block (0,0): (head0 (64 element), 16 nodes), (head1 (64 element), 16
  //   nodes); block(1,0): (head0 (64 element), 16 nodes), (head1 (64 element),
  //   16 nodes); ... block(BLOCKDIM_X-1,0): (head0 (64 element), 16 nodes),
  //   (head1 (64 element), 16 nodes); block (0,1): (head2 (64 element), 16
  //   nodes), (head3 (64 element), 16 nodes); block(1,1): (head2 (64 element),
  //   16 nodes), (head3 (64 element), 16 nodes); ... block(BLOCKDIM_X-1,1):
  //   (head2 (64 element), 16 nodes), (head3 (64 element), 16 nodes);

  for (int idx_block = 0; idx_block < RTX_3090_GRIDSIZE; idx_block++) {
    if (idx_curr_relation <
            num_blocks_xdim_for_all_prev_relation_vect.size() - 1 &&
        idx_block >=
            num_blocks_xdim_for_all_prev_relation_vect[idx_curr_relation]) {
      assert(curr_beg_node_entry_idx / COARSE_SGEMM_NODES_PER_BLOCK ==
             num_blocks_xdim_for_same_relation_vect[idx_curr_relation]);
      idx_curr_relation++;
      curr_beg_node_entry_idx = 0;
    }
    blockid_relation_id_vect.push_back(idx_curr_relation);
    beg_node_entry_idxes_vect.push_back(curr_beg_node_entry_idx);
    curr_beg_node_entry_idx += COARSE_SGEMM_NODES_PER_BLOCK;
    num_blocks_xdim_for_same_relation_per_block_vect.push_back(
        num_blocks_xdim_for_same_relation_vect[idx_curr_relation]);
  }

  dim3 block(COARSE_SGEMM_BLOCKSIZE, 1, 1);
  dim3 grid(RTX_3090_GRIDSIZE, 2, 1);
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  EdgeAttentionConcatenatedFirstStageWeightMulDestCOOKernel<
      TILE_SZ_A, TILE_SZ_B, OUT_DIM, NUM_HEADS><<<grid, block>>>(
      thrust::raw_pointer_cast(
          (intermediate_data->get_intermediate_node_vect_d()).data()),
      concatenated_coo_matrix_column_indices.size(),
      thrust::raw_pointer_cast(concatenated_coo_matrix_column_indices.data()),
      thrust::raw_pointer_cast(concatenated_coo_matrix_values.data()),
      intermediate_data->NodeInputFeatures,
      weights->relation_attention_matrices,
      thrust::raw_pointer_cast(
          (preprocessed_data->get_dest_node_to_unique_index_per_relation_d())
              .data()),
      thrust::raw_pointer_cast(
          (preprocessed_data
               ->get_unique_indices_to_column_indices_per_relation_d())
              .data()),
      thrust::raw_pointer_cast(
          num_unique_indices_to_column_indices_per_relation.data()),
      hyper_params.num_relations,
      thrust::raw_pointer_cast(
          num_blocks_xdim_for_same_relation_per_block_vect.data()),
      thrust::raw_pointer_cast(beg_node_entry_idxes_vect.data()),
      thrust::raw_pointer_cast(blockid_relation_id_vect.data()));

  cuda_err_chk(cudaPeekAtLastError());
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::cout
      << "GPU doGPUEdgeAttentionConcatenatedCOO_128_8 Kernel time: "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << " us" << std::endl;
  return;
}