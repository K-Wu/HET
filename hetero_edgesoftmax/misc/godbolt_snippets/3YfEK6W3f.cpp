// Type your code here, or load an example.
#include <stdio.h>

#include <iostream>
#include <type_traits>

struct UniqueData {
  int64_t *unique_rel_ptrs;
  int64_t *unique_node_indices;
};

struct SeparateCOOData {
  int64_t *rel_ptrs;
  int64_t *row_ptrs;
  int64_t *col_indices;
};

template <typename T> struct EdgeMapperData_ {
  typedef SeparateCOOData U;
};

template <> struct EdgeMapperData_<int> {
  typedef SeparateCOOData U;
};
template <> struct EdgeMapperData_<float> {
  typedef UniqueData U;
};

template <typename T> using EdgeMapperData = typename EdgeMapperData_<T>::U;

int main() {
  EdgeMapperData<int> emd1{};
  EdgeMapperData<float> emd2{
      .unique_rel_ptrs = nullptr,
      .unique_node_indices = nullptr,
  };
}