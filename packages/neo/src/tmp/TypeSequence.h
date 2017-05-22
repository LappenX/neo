#ifndef NEO_TMP_TYPE_SEQUENCE_H
#define NEO_TMP_TYPE_SEQUENCE_H

#include <Common.h>

namespace tmp {

namespace type_sequence {

template <typename... TSequence>
struct Sequence
{
  static const size_t SIZE = sizeof...(TSequence);
};


#include "TypeSequence.hpp"

template <size_t N, typename TSequence>
using nth_type_t = typename detail::NthType<N, TSequence>::type;

} // end of ns type_sequence

} // end of ns tmp

#endif // NEO_TMP_TYPE_SEQUENCE_H