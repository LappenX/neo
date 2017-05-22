#ifndef NEO_TMP_VALUE_SEQUENCE_H
#define NEO_TMP_VALUE_SEQUENCE_H

#include <util/Util.h>

namespace tmp {

namespace value_sequence {

template <typename T, T... TSequence>
struct Sequence
{
  using ElementType = T;
  static const size_t SIZE = sizeof...(TSequence);
};

template <size_t... TSequence>
using IndexSequence = Sequence<size_t, TSequence...>;






#include "ValueSequence.hpp"

template <size_t TSize>
using ascending_numbers_t = typename detail::AscendingNumberSequenceGenerator<TSize>::type;
template <size_t TSize>
using descending_numbers_t = typename detail::DescendingNumberSequenceGenerator<TSize>::type;
template <typename T, T TValue, size_t N>
using repeat_value_t = typename detail::RepeatSequenceGenerator<T, TValue, N>::type;


// TODO: replace with constexpr using sequence constructor, compare performance with MTP
template <size_t N, typename TSequence>
TVALUE(auto, nth_element_v, detail::NthElement<N, TSequence>::value)

} // end of ns value_sequence

} // end of ns tmp

#endif // NEO_TMP_VALUE_SEQUENCE_H