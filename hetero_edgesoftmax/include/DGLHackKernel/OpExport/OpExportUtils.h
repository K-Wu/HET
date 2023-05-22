#pragma once
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>

// TODO: we also needs to make sure the scheme here could handle cases involving
// const volatile and rvalue
template <typename T>
T &NO_DUMMY(T &x) {
  CONSTEXPR_TRUE_CLAUSE_NONREACHABLE(
      (std::is_same<T, int64_t>::value),
      "unspecialized NO_DUMMY should not be called");
  return x;
}
template <>
int &NO_DUMMY(int &x) {
  assert(x > 0);
  return x;
}
template <>
int64_t &NO_DUMMY(int64_t &x) {
  assert(x > 0);
  return x;
}
template <>
float &NO_DUMMY(float &x) {
  assert(x != 0);
  return x;
}
