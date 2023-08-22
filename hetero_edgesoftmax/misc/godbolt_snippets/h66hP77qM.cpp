// x86-64 gcc 12.1 --std=c++17
#include <iostream>
#include <tuple>

// from https://stackoverflow.com/a/47992605
// ex: auto curr_coord = get<i>(coord...);
template <int I, class... Ts> decltype(auto) get(Ts &&...ts) {
  return std::get<I>(std::forward_as_tuple(ts...));
}

// from https://artificial-mind.net/blog/2020/10/31/constexpr-for
template <int Start, int End, int Inc, class F>
constexpr void constexpr_for(F &&f) {
  if constexpr (Start < End) {
    f(std::integral_constant<decltype(Start), Start>());
    constexpr_for<Start + Inc, End, Inc>(f);
  }
}

// from https://artificial-mind.net/blog/2020/10/31/constexpr-for
template <int Start, int End, class TupleType, class TupleTypeMayInclRef>
constexpr size_t get_flattened_index(TupleTypeMayInclRef coord,
                                     TupleType shape) {
  size_t sum = 0;
  constexpr_for<Start, End, 1>([&sum, &coord, &shape](auto i) {
    auto curr_coord = std::get<i>(coord);
    auto curr_coeff = 1;
    constexpr_for<Start, i, 1>(
        [&curr_coeff, &shape](auto j) { curr_coeff *= std::get<j>(shape); });
    sum += curr_coord * curr_coeff;
  });
  return sum;
}

template <int Start, int End, class TupleType>
constexpr size_t get_size(TupleType shape) {
  size_t prod = 1;
  constexpr_for<Start, End, 1>([&prod, &shape](auto i) {
    auto curr_coord = std::get<i>(shape);
    prod *= curr_coord;
  });
  return prod;
}

// from https://stackoverflow.com/a/12742980
template <typename... T>
using tuple_with_removed_refs =
    std::tuple<typename std::remove_reference<T>::type...>;

template <int Start, int End, class TupleTypeMayInclRef2,
          class TupleTypeMayInclRef>
constexpr TupleTypeMayInclRef permute_coord(TupleTypeMayInclRef coord,
                                            TupleTypeMayInclRef2 permutation) {
  TupleTypeMayInclRef result;
  constexpr_for<Start, End, 1>([&result, &coord, &permutation](auto i) {
    auto curr_permute_idx = get<i>(permutation);
    auto curr_coord = get<i>(coord);
    std::get<curr_permute_idx>(result) = curr_coord;
  });
  return tuple_with_removed_refs{result};
}

template <class TupleType, class TupleTypeMayInclRef>
constexpr size_t get_flattened_index(TupleTypeMayInclRef coord,
                                     TupleType shape) {
  return get_flattened_index<0, std::tuple_size<TupleType>{}>(coord, shape);
}

template <int Start, int End, class TupleType>
constexpr TupleType convert_flat_index_to_tuple(size_t flat_index,
                                                TupleType shape) {
  TupleType coord;
  constexpr_for<Start, End, 1>([&flat_index, &coord, &shape](auto i) {
    auto curr_coeff = 1;
    constexpr_for<Start, i, 1>(
        [&curr_coeff, &shape](auto j) { curr_coeff *= std::get<j>(shape); });
    auto curr_coord = flat_index / curr_coeff;
    flat_index -= curr_coord * curr_coeff;
    std::get<i>(coord) = curr_coord;
  });
  return coord;
}

template <class TupleType>
constexpr TupleType convert_flat_index_to_tuple(size_t flat_index,
                                                TupleType shape) {
  return convert_flat_index_to_tuple<0, std::tuple_size<TupleType>{}>(
      flat_index, shape);
}

template <class... Ts> class Coord {
public:
  std::tuple<typename std::remove_reference<Ts>::type...> size;

  Coord(Ts &&...ts) : size(ts...) {}
  Coord(std::tuple<Ts...> size) : size(size) {}
  constexpr size_t _get_flattened_index(Ts &&...coord) {
    return get_flattened_index(std::forward_as_tuple(coord...), size);
  }

  constexpr std::tuple<typename std::remove_reference<Ts>::type...>
  _convert_flat_index_to_tuple(size_t flat_index) {
    return convert_flat_index_to_tuple<
        0, std::tuple_size<
               std::tuple<typename std::remove_reference<Ts>::type...>>{}>(
        flat_index, size);
  }

  constexpr size_t _get_size() {
    return get_size<0, std::tuple_size<std::tuple<
                           typename std::remove_reference<Ts>::type...>>{}>(
        size);
  }
};

auto get_example_coord() {
  Coord a(10ul, 10ul, 10ul, 10ul);
  return a;
}

template <class... Ts>
struct transpose_index_nd
    : public std::tuple<typename std::remove_reference<Ts>::type...> {
  Coord<Ts...> a;
  Coord<Ts...> b;
  transpose_index_nd(Coord<Ts...> a, Coord<Ts...> b) : a(a), b(b) {}
};

template <class Tuple> struct decay_args_tuple;

template <class... Args> struct decay_args_tuple<std::tuple<Args...>> {
  using type = std::tuple<std::decay_t<Args>...>;
};

template <typename... T>
using tuple_with_removed_refs =
    std::tuple<typename std::remove_reference<T>::type...>;

int main() {
  auto a = get_example_coord();
  Coord a2(10ul, 10ul, 10ul, 10ul);
  std::cout << a2._get_size() << std::endl;
  transpose_index_nd ddd(a, a2);

  auto a2_generated_coord = a2._convert_flat_index_to_tuple(11);

  std::cout << std::get<0>(a2_generated_coord)
            << std::get<1>(a2_generated_coord)
            << std::get<2>(a2_generated_coord)
            << std::get<3>(a2_generated_coord) << std::endl;

  decay_args_tuple<decltype(a.size)>::type abc;

  // std::tuple<int,int,int,int> abc;
  std::get<0>(abc) = std::get<0>(a.size);
  std::get<1>(abc) = std::get<1>(a.size);
  std::get<2>(abc) = std::get<2>(a.size);
  std::get<3>(abc) = std::get<3>(a.size);

  std::cout << std::get<0>(abc) << std::get<1>(abc) << std::get<2>(abc)
            << std::get<3>(abc) << std::endl;

  std::cout << a._get_flattened_index(1ul, 2ul, 3ul, 4ul) << std::endl;
  for (int idx = 0; idx < 10; idx++) {
    for (int idx_1 = 0; idx_1 < 10; idx_1++) {
      for (int idx_2 = 0; idx_2 < 10; idx_2++) {
        for (int idx_3 = 0; idx_3 < 10; idx_3++) {
          std::cout << get_flattened_index<>(
                           std::make_tuple(idx, idx_1, idx_2, idx_3),
                           std::make_tuple(idx_3, idx_2, idx_1, idx))
                    << std::endl;
        }
      }
    }
  }

  std::cout << get_flattened_index<>(std::make_tuple(1, 2, 3, 4),
                                     std::make_tuple(10, 10, 10, 10))
            << std::endl;
}