#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "utils.cu.h"

enum class CompactAsOfNodeKind {
  Disabled = 0,
  // by default we need to use binary search to locate the row
  Enabled,
  // when enabled, we can use direct index instead to locate the row
  EnabledWithDirectIndexing,
  EnabledWithDualList,
  EnabledWithDualListWithDirectIndexing
};

constexpr CompactAsOfNodeKind getRidDualList(CompactAsOfNodeKind kind_) {
  return kind_ == CompactAsOfNodeKind::EnabledWithDualList
             ? CompactAsOfNodeKind::Enabled
             : (kind_ == CompactAsOfNodeKind::
                             EnabledWithDualListWithDirectIndexing
                    ? CompactAsOfNodeKind::EnabledWithDirectIndexing
                    : kind_);
}

// TODO: an working example of template alias is at
// https://godbolt.org/z/3YfEK6W3f
template <typename Idx, CompactAsOfNodeKind kind> struct ETypeMapperData {};
template <typename Idx>
struct ETypeMapperData<Idx, CompactAsOfNodeKind::Enabled> {
  Idx *__restrict__ unique_srcs_and_dests_rel_ptrs{nullptr};
  Idx *__restrict__ unique_srcs_and_dests_node_indices{nullptr};
};
template <typename Idx>
struct ETypeMapperData<Idx, CompactAsOfNodeKind::EnabledWithDualList> {
  Idx *__restrict__ unique_srcs_and_dests_rel_ptrs{nullptr};
  Idx *__restrict__ unique_srcs_and_dests_node_indices{nullptr};
};
template <typename Idx>
struct ETypeMapperData<Idx, CompactAsOfNodeKind::EnabledWithDirectIndexing> {
  Idx *__restrict__ edata_idx_to_inverse_idx{nullptr};
};
template <typename Idx>
struct ETypeMapperData<
    Idx, CompactAsOfNodeKind::EnabledWithDualListWithDirectIndexing> {
  Idx *__restrict__ edata_idx_to_inverse_idx{nullptr};
};

__device__ __host__ __forceinline__ constexpr bool
IsCompact(CompactAsOfNodeKind kind) {
  return kind != CompactAsOfNodeKind::Disabled;
}

__device__ __host__ __forceinline__ constexpr bool
IsCompactWithDualList(CompactAsOfNodeKind kind) {
  return kind == CompactAsOfNodeKind::EnabledWithDualList ||
         kind == CompactAsOfNodeKind::EnabledWithDualListWithDirectIndexing;
}

__device__ __host__ __forceinline__ constexpr bool
IsBinarySearch(CompactAsOfNodeKind kind) {
  return kind == CompactAsOfNodeKind::Enabled ||
         kind == CompactAsOfNodeKind::EnabledWithDualList;
}

// TODO: define separate_coo_data with rel_ptrs, row_indices, col_indices, eids,
// TODO: wehn applying it, we need to pull eids out of gdata

// TODO: templated using to declare ETypeMapperData as an alias to
// unique_node_indices struct from
// https://subscription.packtpub.com/book/programming/9781786465184/1/ch01lvl1sec6/creating-type-aliases-and-alias-templates
// template <typename T>
// using vec_t = std::vector<T, custom_allocator<T>>;
// vec_t<int>           vi;
// vec_t<std::string>   vs;

template <typename Idx, bool ETypeRelPtrFlag> struct ETypeData {
  CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
      (std::is_same<Idx, std::int32_t>::value ||
       !std::is_same<Idx, std::int32_t>::value),
      "the program should use partial specialization of this structure");
};

template <typename Idx> struct ETypeData<Idx, true> {
  Idx *__restrict__ etypes{nullptr};
  int64_t num_relations{-1};
};

template <typename Idx> struct ETypeData<Idx, false> {
  Idx *__restrict__ etypes{nullptr};
};

// TODO: is there a way to map from (src idx, etype) instead of edge idx to (row
// index in the compact tensor)?
// TODO: optimize when warp coorperatively work on to reduce the last 4-5 global
// loads
// TODO: figure out metadata caching to optimize the performance
template <typename Idx, CompactAsOfNodeKind kind>
__device__ __forceinline__ Idx find_relational_compact_as_of_node_index(
    Idx idx_relation, Idx idx_node, Idx idx_edata,
    ETypeMapperData<Idx, kind> etype_mapper_data) {
  if constexpr (IsBinarySearch(kind)) {
    Idx idx_relation_offset =
        etype_mapper_data.unique_srcs_and_dests_rel_ptrs[idx_relation];
    Idx idx_relation_plus_one_offset =
        etype_mapper_data.unique_srcs_and_dests_rel_ptrs[idx_relation + 1];
    return idx_relation_offset +
           binary_search<Idx, Idx *>(
               idx_relation_plus_one_offset - idx_relation_offset,
               &(etype_mapper_data
                     .unique_srcs_and_dests_node_indices[idx_relation_offset]),
               idx_node);
  } else {
    return etype_mapper_data.edata_idx_to_inverse_idx[idx_edata];
  }
}
