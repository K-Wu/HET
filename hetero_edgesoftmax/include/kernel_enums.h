#pragma once
#include "cuda.h"
#include "cuda_runtime.h"

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
template <typename Idx, CompactAsOfNodeKind kind>
struct ETypeMapperData {};
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

__device__ __host__ __forceinline__ constexpr bool IsCompact(
    CompactAsOfNodeKind kind) {
  return kind != CompactAsOfNodeKind::Disabled;
}

__device__ __host__ __forceinline__ constexpr bool IsCompactWithDualList(
    CompactAsOfNodeKind kind) {
  return kind == CompactAsOfNodeKind::EnabledWithDualList ||
         kind == CompactAsOfNodeKind::EnabledWithDualListWithDirectIndexing;
}

__device__ __host__ __forceinline__ constexpr bool IsBinarySearch(
    CompactAsOfNodeKind kind) {
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
