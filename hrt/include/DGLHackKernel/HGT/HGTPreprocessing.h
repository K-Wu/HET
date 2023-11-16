#pragma once
#include <cuda_runtime.h>

#include <map>

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

// TODO: in our export implementation, we actually prefer source node to
// destination node because in this case we may reuse the routine for both
// message and the first stage of attention generation. Both processes will work
// on source node.
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
        GetDeviceVectorOfPointersToArrays<
            int, std::vector<cusp::coo_matrix<int, int, cusp::device_memory>::
                                 column_indices_array_type>>(
            coo_matrices_column_indices_unique);
    dest_node_to_unique_index_per_relation_d =
        GetDeviceVectorOfPointersToArrays<
            int, std::vector<thrust::device_vector<int>>>(
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
        GetDeviceVectorOfPointersToArrays<
            int, std::vector<cusp::coo_matrix<int, int, cusp::device_memory>::
                                 column_indices_array_type>>(
            coo_matrices_column_indices_unique);
    dest_node_to_unique_index_per_relation_d =
        GetDeviceVectorOfPointersToArrays<
            int, std::vector<thrust::device_vector<int>>>(
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
        GetDeviceVectorOfPointersToArrays<
            int, std::vector<cusp::coo_matrix<int, int, cusp::device_memory>::
                                 column_indices_array_type>>(
            coo_matrices_column_indices_unique);
    dest_node_to_unique_index_per_relation_d =
        GetDeviceVectorOfPointersToArrays<
            int, std::vector<thrust::device_vector<int>>>(
            dest_node_to_unique_index_per_relation);

    other.unique_indices_to_column_indices_per_relation_d.clear();
    other.dest_node_to_unique_index_per_relation_d.clear();
  }

  void CheckPointerArrayValidity() const {
    assert(unique_indices_to_column_indices_per_relation_d.size() != 0);
    assert(dest_node_to_unique_index_per_relation_d.size() != 0);
  }

  thrust::device_vector<int *> &
  get_unique_indices_to_column_indices_per_relation_d() {
    CheckPointerArrayValidity();
    return unique_indices_to_column_indices_per_relation_d;
  }
  const thrust::device_vector<int *> &
  get_unique_indices_to_column_indices_per_relation_d() const {
    CheckPointerArrayValidity();
    return unique_indices_to_column_indices_per_relation_d;
  }
  thrust::device_vector<int *> &get_dest_node_to_unique_index_per_relation_d() {
    CheckPointerArrayValidity();
    return dest_node_to_unique_index_per_relation_d;
  }
  const thrust::device_vector<int *> &
  get_dest_node_to_unique_index_per_relation_d() const {
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
  // TODO: make this uint64_t to avoid overflow
  int64_t PERF_ltsgemm_workspaceSize;
  // TODO: implement it once it is finished
  HGTLayerHyperParams(int num_relations, int num_node_types, int num_nodes,
                      int num_edges, int input_dim, int klinear_out_dim,
                      int qlinear_out_dim, int vlinear_out_dim, int message_dim,
                      int alinear_out_dim, int num_heads,
                      int64_t PERF_ltsgemm_workspaceSize)
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

  HGTLayerHyperParams(const std::map<std::string, int64_t> &params)
      : num_relations(params.at("num_relations")),
        num_node_types(params.at("num_node_types")),
        num_nodes(params.at("num_nodes")),
        num_edges(params.at("num_edges")),
        input_dim(params.at("input_dim")),
        klinear_out_dim(params.at("klinear_out_dim")),
        qlinear_out_dim(params.at("qlinear_out_dim")),
        vlinear_out_dim(params.at("vlinear_out_dim")),
        message_dim(params.at("message_dim")),
        alinear_out_dim(params.at("alinear_out_dim")),
        num_heads(params.at("num_heads")),
        PERF_ltsgemm_workspaceSize(params.at("PERF_ltsgemm_workspaceSize")) {}

  std::map<std::string, int64_t> GetMapRepr() const {
    std::map<std::string, int64_t> params;
    params["num_relations"] = num_relations;
    params["num_node_types"] = num_node_types;
    params["num_nodes"] = num_nodes;
    params["num_edges"] = num_edges;
    params["input_dim"] = input_dim;
    params["klinear_out_dim"] = klinear_out_dim;
    params["qlinear_out_dim"] = qlinear_out_dim;
    params["vlinear_out_dim"] = vlinear_out_dim;
    params["message_dim"] = message_dim;
    params["alinear_out_dim"] = alinear_out_dim;
    params["num_heads"] = num_heads;
    params["PERF_ltsgemm_workspaceSize"] = PERF_ltsgemm_workspaceSize;
    return params;
  }

  std::string getStrRepr() const {
    return "HGTLayerHyperParams num_relations: " +
           std::to_string(num_relations) +
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
};

// preprocessing also generates primitive array of pointers for CUDA kernel to
// access
template <int OUT_DIM, int NUM_HEADS, int TILE_SZ_A, int TILE_SZ_B>
std::shared_ptr<HGTLayerExecPreprocessedData> HGTLayerPreprocessing(
    HGTLayerHyperParams hyper_params,
    std::vector<cusp::coo_matrix<
        int, int, cusp::device_memory>::column_indices_array_type>
        coo_matrices_column_indices) {
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
  thrust::device_vector<int> num_unique_indices_to_column_indices_per_relation(
      hyper_params.num_relations, -1);  // need to output
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
    num_unique_indices_to_column_indices_per_relation[idx_relation] =
        coo_matrices_column_indices_unique[idx_relation].size();
  }

  std::vector<thrust::device_vector<int>>
      dest_node_to_unique_index_per_relation =
          std::vector<thrust::device_vector<int>>(
              hyper_params.num_relations,
              thrust::device_vector<int>(
                  hyper_params.num_nodes,
                  -1));  // need to persist (pointed by elements in
                         // dest_node_to_unique_index_per_relation_d)
                         //   thrust::device_vector<int *>
  //       dest_node_to_unique_index_per_relation_d;  // need to output

  //   // TODO: dedicated function to generate array of (pointers to arrays)
  //   for (int idx_relation = 0; idx_relation < hyper_params.num_relations;
  //        idx_relation++) {
  //     dest_node_to_unique_index_per_relation_d.push_back(thrust::raw_pointer_cast(
  //         dest_node_to_unique_index_per_relation[idx_relation].data()));
  //   }

  for (int idx_relation = 0; idx_relation < hyper_params.num_relations;
       idx_relation++) {
    thrust::counting_iterator<int> first_counting_iter(0);
    thrust::counting_iterator<int> last_counting_iter =
        first_counting_iter +
        coo_matrices_column_indices_unique[idx_relation].size();

    int *curr_dest_node_to_unique_index_per_relation_d =
        thrust::raw_pointer_cast(
            dest_node_to_unique_index_per_relation[idx_relation].data());

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
    dest_node_to_unique_index_per_relation[idx_relation].resize(
        num_unique_indices_to_column_indices_per_relation
            [idx_relation]);  // resize to actual size
  }

  return std::make_shared<HGTLayerExecPreprocessedData>(
      coo_matrices_column_indices_unique,
      num_unique_indices_to_column_indices_per_relation,
      dest_node_to_unique_index_per_relation);
}