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
    ASSERT(dest.template dim<I - 1>() == src.template dim<I - 1>(), "Incompatible dimensions");
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
    // TODO: calculate linear index only once, also for other multi tensor operations, with compilation switch ALLOWS_LINEAR_INDEX
    dest(coords...) = src(coords...);
  }
};

} // end of ns detail

struct LocalElwise
{
  template <typename TTensorDest, typename TTensorSrc,
    ENABLE_IF(is_tensor_v<TTensorDest>::value && is_tensor_v<TTensorSrc>::value)>
  __host__ __device__
  static void copy(TTensorDest&& dest, TTensorSrc&& src)
  {
    static_assert(are_compatible_dimseqs_v<tensor_dimseq_t<TTensorDest>, tensor_dimseq_t<TTensorSrc>>::value, "Incompatible static dimensions");
    ASSERT(areSameDimensions(dest.dims(), src.dims()), "Incompatible runtime dimensions");
    static_assert(mem::is_on_current<TensorTraits<tensor_clean_t<TTensorDest>>::MEMORY_TYPE>() && mem::is_on_current<TensorTraits<tensor_clean_t<TTensorSrc>>::MEMORY_TYPE>(), "Invalid memory types");
    
    const size_t MAX_RANK = math::min(non_trivial_dimensions_num_v<tensor_dimseq_t<TTensorDest>>::value, non_trivial_dimensions_num_v<tensor_dimseq_t<TTensorSrc>>::value);
    detail::LocalElwise<MAX_RANK>::copy(util::forward<TTensorDest>(dest), util::forward<TTensorSrc>(src));
  }
};

struct TransferStorage
{
  template <typename TTensorDest, typename TTensorSrc,
    ENABLE_IF(is_tensor_v<TTensorDest>::value && is_tensor_v<TTensorSrc>::value)>
  __host__
  static void copy(TTensorDest&& dest, TTensorSrc&& src)
  {
    static_assert(are_compatible_dimseqs_v<tensor_dimseq_t<TTensorDest>, tensor_dimseq_t<TTensorSrc>>::value, "Incompatible static dimensions");
    ASSERT(areSameDimensions(dest.dims(), src.dims()), "Incompatible runtime dimensions");

    copy_helper(dest, src);
  }

private:
  template <typename TStorageType1, typename TIndexStrategy1, typename TThisType1, typename TSuperType1,
            typename TStorageType2, typename TIndexStrategy2, typename TThisType2, typename TSuperType2>
  __host__
  static void copy_helper(DenseStorageTensor<TStorageType1, TIndexStrategy1, TThisType1, TSuperType1>& dest,
                          const DenseStorageTensor<TStorageType2, TIndexStrategy2, TThisType2, TSuperType2>& src)
  {
    static_assert(std::is_same<TIndexStrategy1, TIndexStrategy2>::value, "Storages must have the same indexing strategy");

    mem::copy<mem::is_on_host<TStorageType1::MEMORY_TYPE>(), mem::is_on_host<TStorageType2::MEMORY_TYPE>()>
          (dest.storage().ptr(), src.storage().ptr(), dimensionProduct(dest.dims()));
  }
};

} // end of ns copier

} // end of ns tensor