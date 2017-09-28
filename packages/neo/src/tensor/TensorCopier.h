#include "Tensor.h"

namespace tensor {

namespace copier {

namespace detail {

template <size_t I>
struct LocalElwise
{
  template <typename TTensorDest, typename TTensorSrc, typename... TCoords>
  __host__ __device__
  static void copy(TTensorDest&& dest, TTensorSrc&& src, TCoords... coords)
  {
    ASSERT(dest.template dim<I - 1>() == src.template dim<I - 1>(), "Inconsistent dimensions");
    size_t max = dest.template dim<I - 1>();
    for (size_t i = 0; i < max; i++)
    {
      LocalElwise<I - 1>::copy(util::forward<TTensorDest>(dest), util::forward<TTensorSrc>(src), i, coords...);
    }
  }
};

template <>
struct LocalElwise<0>
{
  template <typename TTensorDest, typename TTensorSrc, typename... TCoords>
  __host__ __device__
  static void copy(TTensorDest&& dest, TTensorSrc&& src, TCoords... coords)
  {
    // TODO: calculate linear index only once, also for other multi tensor operations
    dest(coords...) = src(coords...);
  }
};

} // end of ns detail

struct LocalElwise
{
  template <typename TTensorDest, typename TTensorSrc>
  __host__ __device__
  static void copy(TTensorDest&& dest, TTensorSrc&& src)
  {
    static_assert(are_compatible_dimseqs_v<tensor_dimseq_t<TTensorDest>, tensor_dimseq_t<TTensorSrc>>::value, "Incompatible static dimensions");
    ASSERT(areSameDimensions(dest.dims(), src.dims()), "Inconsistent runtime dimensions");
    ASSERT(mem::is_on_current<TensorTraits<tensor_clean_t<TTensorDest>>::MEMORY_TYPE>() && mem::is_on_current<TensorTraits<tensor_clean_t<TTensorSrc>>::MEMORY_TYPE>(), "Invalid memory types");
    const size_t MAX_RANK = math::min(non_trivial_dimensions_num_v<tensor_dimseq_t<TTensorDest>>::value, non_trivial_dimensions_num_v<tensor_dimseq_t<TTensorSrc>>::value);
    detail::LocalElwise<MAX_RANK>::copy(util::forward<TTensorDest>(dest), util::forward<TTensorSrc>(src));
  }
};

} // end of ns copier

template <typename TTensorCopier, typename TTensorDest, typename TTensorSrc,
  ENABLE_IF(is_tensor_v<TTensorDest>::value && is_tensor_v<TTensorSrc>::value)>
__host__ __device__
void copy(TTensorDest&& dest, TTensorSrc&& src)
{
  TTensorCopier::copy(util::forward<TTensorDest>(dest), util::forward<TTensorSrc>(src));
}

} // end of ns tensor