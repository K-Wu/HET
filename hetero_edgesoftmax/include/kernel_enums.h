#pragma once
#include "cuda.h"
#include "cuda_runtime.h"

enum class CompactAsOfNodeKind {
  Disabled = 0,
  // by default we need to use binary search to locate the row
  Enabled,
  EnabledWithDualList,
  // when enabled, we can use direct index instead to locate the row
  EnabledWithDirectIndexing,
  EnabledWithDualListWithDirectIndexing
};

template <typename Idx, CompactAsOfNodeKind kind>
struct ETypeMapperData {
  Idx *__restrict__ unique_srcs_and_dests_rel_ptrs{nullptr};
  Idx *__restrict__ unique_srcs_and_dests_node_indices{nullptr};
};

// TODO: templated using to declare ETypeMapperData as an alias to
// unique_node_indices struct from
// https://subscription.packtpub.com/book/programming/9781786465184/1/ch01lvl1sec6/creating-type-aliases-and-alias-templates
// template <typename T>
// using vec_t = std::vector<T, custom_allocator<T>>;
// vec_t<int>           vi;
// vec_t<std::string>   vs;

__device__ __host__ __forceinline__ constexpr bool IsCompact(
    CompactAsOfNodeKind kind) {
  return kind != CompactAsOfNodeKind::Disabled;
}

__device__ __host__ __forceinline__ constexpr bool IsCompactWithDualList(
    CompactAsOfNodeKind kind) {
  return kind == CompactAsOfNodeKind::EnabledWithDualList ||
         kind == CompactAsOfNodeKind::EnabledWithDualListWithDirectIndexing;
}