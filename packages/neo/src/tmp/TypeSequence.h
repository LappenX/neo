#ifndef NEO_TMP_TYPE_SEQUENCE_H
#define NEO_TMP_TYPE_SEQUENCE_H

#include <Common.h>

#include <type_traits>

namespace tmp {

namespace type_sequence {

namespace pred {

template <typename TTo>
struct is_convertible_to
{
  template <typename TFrom>
  struct get
  {
    static const bool value = std::is_convertible<TFrom, TTo>::value;
  };
};

} // end of ns predicate





template <typename... TSequence>
struct Sequence
{
  static const size_t SIZE = sizeof...(TSequence);
};

#include "TypeSequence.hpp"

template <size_t N, typename TSequence>
using nth_type_t = typename detail::NthType<N, TSequence>::type;

template <typename TPredicate, typename TSequence>
TVALUE(bool, all_apply_v, detail::AllApply<TPredicate, TSequence>::value)

} // end of ns type_sequence

} // end of ns tmp

#endif // NEO_TMP_TYPE_SEQUENCE_H