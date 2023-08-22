// x86-64 gcc 12.2
#include <iostream>
// Type your code here, or load an example.
#define CONSTEXPR_FALSE_CLAUSE_NONREACHABLE(FLAG, reason)                      \
  static_assert(std::is_same<std::true_type,                                   \
                             std::integral_constant<bool, FLAG>>::value &&     \
                reason)

#define CONSTEXPR_TRUE_CLAUSE_NONREACHABLE(FLAG, reason)                       \
  static_assert(std::is_same<std::false_type,                                  \
                             std::integral_constant<bool, FLAG>>::value &&     \
                reason)

#define CONSTEXPR_FALSE_CLAUSE_STATIC_ASSERT(FLAG, asserted_true_expression,   \
                                             reason)                           \
  static_assert(                                                               \
      std::is_same<std::true_type,                                             \
                   std::integral_constant<bool, FLAG>>::value /*unreachable*/  \
      || (/*or true*/ asserted_true_expression && reason))

#define CONSTEXPR_TRUE_CLAUSE_STATIC_ASSERT(FLAG, asserted_true_expression,    \
                                            reason)                            \
  static_assert(                                                               \
      std::is_same<std::false_type,                                            \
                   std::integral_constant<bool, FLAG>>::value /*unreachable*/  \
      || (/*or true*/ asserted_true_expression && reason))

template <bool flag> int trial() {
  if constexpr (flag) {
    std::cout << "1" << std::endl;
  } else {
    CONSTEXPR_FALSE_CLAUSE_NONREACHABLE(flag, "abc");
  }
}

template <bool flag> int trial2() {
  if constexpr (flag) {
    CONSTEXPR_TRUE_CLAUSE_STATIC_ASSERT(flag, true, "");
    std::cout << "2" << std::endl;
  } else {
    CONSTEXPR_FALSE_CLAUSE_NONREACHABLE(flag, "abc");
  }
}

int main() {
  trial<true>();
  trial2<true>();
}