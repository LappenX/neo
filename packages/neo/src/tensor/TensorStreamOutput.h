#include "Tensor.h"

namespace tensor {

namespace detail {

template <bool TZeroSized, size_t I>
struct StreamOutputHelper
{
  template <typename TStreamType, typename TTensorType, typename... TCoords>
  __host__ __device__
  static void output(TStreamType& stream, TTensorType&& tensor, TCoords... coords)
  {
    size_t max = tensor.template dim<I - 1>();
    stream << "[";
    for (size_t i = 0; i < max; i++)
    {
      StreamOutputHelper<TZeroSized, I - 1>::output(stream, util::forward<TTensorType>(tensor), i, coords...);
      if (I == 1 && i < max - 1)
      {
        stream << " ";
      }
    }
    stream << "]";
  }
};

template <>
struct StreamOutputHelper<false, 0>
{
  template <typename TStreamType, typename TTensorType, typename... TCoords>
  __host__ __device__
  static void output(TStreamType& stream, TTensorType&& tensor, TCoords... coords)
  {
    stream << tensor(coords...);
  }
};

template <size_t I>
struct StreamOutputHelper<true, I>
{
  template <typename TStreamType, typename TTensorType, typename... TCoords>
  __host__ __device__
  static void output(TStreamType& stream, TTensorType&& tensor, TCoords... coords)
  {
    stream << "[]";
  }
};

} // end of ns detail

// ColMajor output
template <typename TStreamType, typename TTensorType, typename TElementType, size_t... TDims,
  ENABLE_IF((std::is_convertible<TStreamType, const std::ostream&>::value || std::is_convertible<TStreamType, const util::DeviceCout&>::value)
    && is_tensor_v<TTensorType>::value)>
TStreamType& operator<<(TStreamType& stream, const Tensor<TTensorType, TElementType, TDims...>& tensor)
{
  detail::StreamOutputHelper<tmp::value_sequence::nth_element_v<0, DimSeq<TDims...>>::value == 0,
                             non_trivial_dimensions_num_v<DimSeq<TDims...>>::value>
    ::output(stream, tensor);
  return stream;
}

} // end of ns tensor