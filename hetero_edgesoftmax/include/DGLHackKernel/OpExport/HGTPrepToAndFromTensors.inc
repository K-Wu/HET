#pragma once

// std::vector<at::Tensor> HGTLayerPreprocessing_wrapper(){
// TODO:
// }


template <typename KeyType, typename ValueType>
std::map<KeyType, ValueType> convert_torch_dict_to_map(
    const torch::Dict<KeyType, ValueType>& dict) {
  std::map<KeyType, ValueType> result;
  for (const auto& item : dict) {
    result[item.key()] = item.value();
  }
  return result;
}

template <typename KeyType, typename ValueType>
torch::Dict<KeyType, ValueType> convert_map_to_torch_dict(
    const std::map<KeyType, ValueType>& map) {
  torch::Dict<KeyType, ValueType> result;
  for (const auto& item : map) {
    result.insert(item.first, item.second);
  }
  return result;
}

template <typename ContainerType>
at::Tensor export_thrust_device_vector_to_torch_tensor(
   ContainerType& vec) {
  //static_assert(std::is_same<T, int>::value, "only support int for now");
  printf("WARNING: Narrowing Cast from size_t to int64_t in export_thrust_device_vector_to_torch_tensor!\n");
  return torch::from_blob(vec.data().get(), {(signed long) vec.size()}, at::TensorOptions(caffe2::TypeMeta::Make<int32_t>()).device(at::kCUDA));
}

template <typename T>
thrust::device_vector<T> import_thrust_device_vector_from_torch_tensor(
    const at::Tensor& tensor) {
  static_assert(std::is_same<T, int>::value, "only support int for now");
  return thrust::device_vector<T>(
      reinterpret_cast<T*>(tensor.data_ptr()), reinterpret_cast<T*>(tensor.data_ptr()) + tensor.numel());
}

// internal utility function as the end of a torch API function, to convert the input tensors to the corresponding C++ data structures
std::vector<std::vector<at::Tensor>> _export_HGTLayerExecPreprocessedData(HGTLayerExecPreprocessedData& preprocessed_data){
    std::vector<std::vector<at::Tensor>> result;
    result.push_back({});
    for (size_t idx_relation = 0; idx_relation<preprocessed_data.coo_matrices_column_indices_unique.size(); idx_relation++){
        result[0].push_back(export_thrust_device_vector_to_torch_tensor<cusp::array1d<int, cusp::device_memory>>(preprocessed_data.coo_matrices_column_indices_unique[idx_relation]));
    }
    result.push_back({});
    for (size_t idx_relation = 0; idx_relation<preprocessed_data.dest_node_to_unique_index_per_relation.size(); idx_relation++){
        result[1].push_back(export_thrust_device_vector_to_torch_tensor<thrust::device_vector<int>>(preprocessed_data.dest_node_to_unique_index_per_relation[idx_relation]));
    }
    result.push_back({export_thrust_device_vector_to_torch_tensor<thrust::device_vector<int>>(preprocessed_data.num_unique_indices_to_column_indices_per_relation)});
    return result;
}

// internal utility function as the beginning of a torch API function, to convert the corresponding C++ data structures to a torch structure
HGTLayerExecPreprocessedData _import_HGTLayerExecPreprocessedData(const std::vector<at::Tensor>& tensors){
    std::vector<cusp::coo_matrix<int, int,
                               cusp::device_memory>::column_indices_array_type>
      coo_matrices_column_indices_unique;
    std::vector<thrust::device_vector<int>>
      dest_node_to_unique_index_per_relation;
    for (size_t idx_relation = 0; idx_relation<tensors[0].size(0); idx_relation++){
        coo_matrices_column_indices_unique.push_back(import_thrust_device_vector_from_torch_tensor<int>(tensors[0][idx_relation]));
    }
    for (size_t idx_relation = 0; idx_relation<tensors[1].size(0); idx_relation++){
        dest_node_to_unique_index_per_relation.push_back(import_thrust_device_vector_from_torch_tensor<int>(tensors[1][idx_relation]));
    }
    thrust::device_vector<int> num_unique_indices_to_column_indices_per_relation = import_thrust_device_vector_from_torch_tensor<int>(tensors[2][0]);
    HGTLayerExecPreprocessedData result(coo_matrices_column_indices_unique, num_unique_indices_to_column_indices_per_relation, dest_node_to_unique_index_per_relation);
    return result;
}


// internal utility function as the end of a torch API function, to convert the input tensors to the corresponding C++ data structures
torch::Dict<std::string, int64_t> _export_HGTLayerHyperParams(const HGTLayerHyperParams& hyper_params){
    return convert_map_to_torch_dict(hyper_params.GetMapRepr());
}

// internal utility function as the beginning of a torch API function, to convert the corresponding C++ data structures to a torch structure
HGTLayerHyperParams _import_HGTLayerHyperParams(const torch::Dict<std::string, int64_t>& dict){
    return HGTLayerHyperParams(convert_torch_dict_to_map(dict));
}