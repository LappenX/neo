#ifndef NEO_TMP_VALUE_SEQUENCE_H
#define NEO_TMP_VALUE_SEQUENCE_H

#include <util/Util.h>
#include <util/Math.h>

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
template <typename TSequence>
using reverse_t = detail::reverse_t<TSequence>;
template <typename TSequence, size_t TLength>
using cut_to_length_from_start_t = detail::cut_to_length_from_start_t<TSequence, TLength>;
template <typename TSequence, size_t TLength>
using cut_to_length_from_end_t = detail::cut_to_length_from_end_t<TSequence, TLength>;;
template <typename T, typename TSequence, T TElement>
TVALUE(bool, contains_v, detail::SequenceContains<T, TSequence, TElement>::value)
template <size_t N, typename TSequence>
TVALUE(auto, nth_element_v, detail::NthElement<N, TSequence>::value) // TODO: replace with constexpr using sequence constructor, compare performance with MTP
template <typename T, typename TSequence, size_t TSetIndex, T TSetValue>
using set_t = detail::set_t<T, TSequence, TSetIndex, TSetValue>;

} // end of ns value_sequence

} // end of ns tmp

#endif // NEO_TMP_VALUE_SEQUENCE_H