#pragma once
#include "cuda.h"
#include "cuda_runtime.h"

enum class CompactAsOfNodeKind {
  Disabled,
  // by default we need to use binary search to locate the row
  Enabled,
  EnabledWithDualList,
  // when enabled, we can use direct index instead to locate the row
  EnabledWithDirectIndexing,
  EnabledWithDualListWithDirectIndexing
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