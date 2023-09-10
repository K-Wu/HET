#pragma once
#include "utils.cu.h"

// Working compilation example: https://godbolt.org/z/h66hP77qM backup:
// https://gist.github.com/K-Wu/7db30d4cf93a1fff76fa2631f95c476b

#include <assert.h>
#include <stddef.h>

#include <tuple>

#include "variadic_tricks.h"

// from https://artificial-mind.net/blog/2020/10/31/constexpr-for
template <size_t Start, size_t End, class TupleType, class TupleTypeMayInclRef>
_HOST_DEVICE_METHOD_QUALIFIER constexpr size_t get_flattened_index(
    TupleTypeMayInclRef coord, TupleType shape) {
  size_t sum = 0;
  constexpr_for<Start, End, 1>(
      [&sum, &coord, &shape] _HOST_DEVICE_METHOD_QUALIFIER(auto i) {
        auto curr_coord = std::get<i>(coord);
        auto curr_coeff = 1;
        constexpr_for<Start, i, 1>(
            [&curr_coeff, &shape] _HOST_DEVICE_METHOD_QUALIFIER(auto j) {
              curr_coeff *= std::get<j>(shape);
            });
        sum += curr_coord * curr_coeff;
      });
  return sum;
}

template <int Start, int End, class TupleType>
_HOST_DEVICE_METHOD_QUALIFIER constexpr size_t get_tuple_prod(TupleType shape) {
  size_t prod = 1;
  constexpr_for<Start, End, 1>(
      [&prod, &shape] _HOST_DEVICE_METHOD_QUALIFIER(auto i) {
        auto curr_coord = std::get<i>(shape);
        prod *= curr_coord;
      });
  return prod;
}

template <size_t Start, size_t End, class TupleType>
_HOST_DEVICE_METHOD_QUALIFIER constexpr TupleType convert_flat_index_to_tuple(
    size_t flat_index, TupleType shape) {
  TupleType coord;
  constexpr_for<Start, End, 1>(
      [&flat_index, &coord, &shape] _HOST_DEVICE_METHOD_QUALIFIER(auto i) {
        auto curr_coeff = 1;
        constexpr_for<Start, i, 1>([&curr_coeff, &shape](auto j)
                                       _HOST_DEVICE_METHOD_QUALIFIER {
                                         curr_coeff *= std::get<j>(shape);
                                       });
        auto curr_coord = flat_index / curr_coeff;
        flat_index -= curr_coord * curr_coeff;
        std::get<i>(coord) = curr_coord;
      });
  return coord;
}

template <class TupleType, class TupleTypeMayInclRef>
_HOST_DEVICE_METHOD_QUALIFIER constexpr size_t get_flattened_index(
    TupleTypeMayInclRef coord, TupleType shape) {
  return get_flattened_index<0, std::tuple_size<TupleType>{}>(coord, shape);
}

template <class TupleType>
_HOST_DEVICE_METHOD_QUALIFIER constexpr TupleType convert_flat_index_to_tuple(
    size_t flat_index, TupleType shape) {
  return convert_flat_index_to_tuple<0, std::tuple_size<TupleType>{}>(
      flat_index, shape);
}

template <class... Ts>
class Coord {
 public:
  std::tuple<typename std::remove_reference<Ts>::type...>
      shape;  // make sure this variable is materialized locally

  _HOST_DEVICE_METHOD_QUALIFIER
  Coord(Ts... ts) : shape(ts...) {}
  _HOST_DEVICE_METHOD_QUALIFIER
  Coord(std::tuple<Ts...> shape) : shape(shape) {}
  // NB: underscore to deliberately distinct name from get_flattened_index
  // function to resolve compilation error
  _HOST_DEVICE_METHOD_QUALIFIER
  constexpr size_t _get_flattened_index(Ts&&... coord) {
    return get_flattened_index(std::forward_as_tuple(coord...), shape);
  }

  _HOST_DEVICE_METHOD_QUALIFIER
  constexpr std::tuple<typename std::remove_reference<Ts>::type...>
  _convert_flat_index_to_tuple(size_t flat_index) {
    return convert_flat_index_to_tuple<0, std::tuple_size<TupleType>{}>(
        flat_index, shape);
  }

  _HOST_DEVICE_METHOD_QUALIFIER
  constexpr size_t _get_size() {
    return get_tuple_prod<
        0, std::tuple_size<
               std::tuple<typename std::remove_reference<Ts>::type...>>{}>(
        size);
  }
};

template <int Start, int End, class TupleTypeMayInclRef2,
          class TupleTypeMayInclRef>
_HOST_DEVICE_METHOD_QUALIFIER constexpr decay_args_tuple<decltype(coord)>::type
permute_coord(TupleTypeMayInclRef coord, TupleTypeMayInclRef2 permutation) {
  decay_args_tuple<decltype(coord)>::type result;
  constexpr_for<Start, End, 1>(
      [&result, &coord, &permutation] _HOST_DEVICE_METHOD_QUALIFIER(auto i) {
        auto curr_permute_idx = std::get<i>(permutation);
        auto curr_coord = std::get<i>(coord);
        std::get<curr_permute_idx>(result) = curr_coord;
      });
  return result;
}

auto get_example_coord() {
  Coord a(10ul, 10ul, 10ul, 10ul);
  return a;
}

void test() {
  // TODO: Coord needs to store shape rather than tuple of references to shape
  // elements
  auto a2 = get_example_coord();
  assert(a2._get_flattened_index(1ul, 2ul, 3ul, 4ul) == 4321);

  Coord a(10ul, 10ul, 10ul, 10ul);
  assert(a._get_flattened_index(1ul, 2ul, 3ul, 4ul) == 4321);
  assert(get_flattened_index<>(std::make_tuple(1, 2, 3, 4),
                               std::make_tuple(10, 10, 10, 10)) == 4321);
}
