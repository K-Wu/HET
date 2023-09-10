#pragma once
#include <stddef.h>

#include <tuple>

#include "utils.cu.h"

// from https://stackoverflow.com/a/12742980
// e.g.: return tuple_with_removed_refs{result};
template <typename... T>
using tuple_with_removed_refs =
    std::tuple<typename std::remove_reference<T>::type...>;

// from https://stackoverflow.com/a/72027501
template <class Tuple>
struct decay_args_tuple;
template <class... Args>
struct decay_args_tuple<std::tuple<Args...>> {
  using type = std::tuple<std::decay_t<Args>...>;
};

// from https://artificial-mind.net/blog/2020/10/31/constexpr-for
template <size_t Start, size_t End, size_t Inc, class F>
_HOST_DEVICE_METHOD_QUALIFIER constexpr void constexpr_for(F&& f) {
  if constexpr (Start < End) {
    f(std::integral_constant<decltype(Start), Start>());
    constexpr_for<Start + Inc, End, Inc>(f);
  }
}

#if 0
// from https://stackoverflow.com/a/37879730
template <class... Ts, std::size_t... Is, class Tuple>
decltype( auto ) tie_from_specified( std::index_sequence<Is...>, Tuple& tuple )
{
    return std::tuple<Ts...>{ std::get<Is>( tuple )... };
}

template <class... Ts, class Tuple>
decltype( auto ) tie_from( Tuple& tuple )
{
    return tie_from_specified<Ts...>( std::make_index_sequence<sizeof...( Ts )>{}, tuple );
}

// from https://stackoverflow.com/a/47992605
// e.g. auto curr_coord = get<i>(coord...);
template <size_t I, class... Ts>
decltype(auto) get(Ts&&... ts) {
  return std::get<I>(std::forward_as_tuple(ts...));
}
#endif
