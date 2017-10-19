#include "Tensor.h"

namespace tensor {

namespace detail {

template <typename TThisType, typename TElementType, bool TReturnsReference>
class ElementAccessFunctions
{
public:
  template <typename... TCoordArgTypes>
  __host__ __device__
  TElementType& operator()(TCoordArgTypes&&... coords)
  {
    ASSERT(coordsAreInRange(*static_cast<const TThisType*>(this), util::forward<TCoordArgTypes>(coords)...), "Coordinates are out of range");
    return static_cast<TThisType*>(this)->get_element_impl(util::forward<TCoordArgTypes>(coords)...);
  }

  template <typename... TCoordArgTypes>
  __host__ __device__
  const TElementType& operator()(TCoordArgTypes&&... coords) const
  {
    ASSERT(coordsAreInRange(*static_cast<const TThisType*>(this), util::forward<TCoordArgTypes>(coords)...), "Coordinates are out of range");
    return static_cast<const TThisType*>(this)->get_element_impl(util::forward<TCoordArgTypes>(coords)...);
  }
};

template <typename TThisType, typename TElementType>
class ElementAccessFunctions<TThisType, TElementType, false>
{
public:
  template <typename... TCoordArgTypes>
  __host__ __device__
  TElementType operator()(TCoordArgTypes&&... coords) const
  {
    ASSERT(coordsAreInRange(*static_cast<const TThisType*>(this), util::forward<TCoordArgTypes>(coords)...), "Coordinates are out of range");
    return static_cast<const TThisType*>(this)->get_element_impl(util::forward<TCoordArgTypes>(coords)...);
  }
};

namespace detail {

template <typename... TArgs>
constexpr size_t multiply_all_but_first(size_t dim0, TArgs... dims)
{
  return math::multiply(dims...);
}

template <bool TStatic>
struct EvalHelper;

template <>
struct EvalHelper<true>
{
  template <typename TTensorCopier, typename TAllocator, typename TIndexStrategy, typename TTensor>
  __host__ __device__
  static DenseLocalStorageTensorFromSequence<tensor_elementtype_t<TTensor>, TIndexStrategy, tensor_dimseq_t<TTensor>> eval(TTensor&& tensor)
  {
    DenseLocalStorageTensorFromSequence<tensor_elementtype_t<TTensor>, TIndexStrategy, tensor_dimseq_t<TTensor>> result;
    TTensorCopier::copy(result, tensor);
    return result;
  }
};

template <>
struct EvalHelper<false>
{
  template <typename TTensorCopier, typename TAllocator, typename TIndexStrategy, typename TTensor>
  __host__ __device__
  static DenseAllocStorageTensorFromSequence<tensor_elementtype_t<TTensor>, TAllocator, TIndexStrategy, tensor_dimseq_t<TTensor>> eval(TTensor&& tensor)
  {
    DenseAllocStorageTensorFromSequence<tensor_elementtype_t<TTensor>, TAllocator, TIndexStrategy, tensor_dimseq_t<TTensor>> result(tensor.dims());
    TTensorCopier::copy(result, tensor);
    return result;
  }
};

} // end of ns detail

template <typename TThisType, typename TElementType, TENSOR_DIMS_DECLARE_NO_DEFAULT>
class Tensor : public ElementAccessFunctions<TThisType, TElementType, TensorTraits<TThisType>::RETURNS_REFERENCE>
{
public:
  static_assert(detail::multiply_all_but_first(TENSOR_DIMS_USE) != 0, "Only first dimension can be zero");
  using ThisType = TThisType;

  template <size_t TIndex>
  __host__ __device__
  size_t dim() const
  {
    return static_cast<const TThisType*>(this)->template dim_impl<TIndex>();
  }

  __host__ __device__
  size_t dim(size_t index) const
  {
    return static_cast<const TThisType*>(this)->dim_impl(index);
  }
  // TODO: return dims vector not as VectorXs but as general reference to Tensor vector
  template <size_t TLength = non_trivial_dimensions_num_v<DimSeq<TENSOR_DIMS_USE>>::value>
  __host__ __device__
  VectorXs<TLength> dims() const
  {
    static_assert(TLength >= non_trivial_dimensions_num_v<DimSeq<TENSOR_DIMS_USE>>::value, "Non-trivial dimensions are cut off");
    return static_cast<const TThisType*>(this)->template dims_impl<TLength>();
  }

  template <typename TTensorCopier = tensor::copier::Default, typename TAllocator = mem::alloc::heap, typename TIndexStrategy = ColMajorIndexStrategy>
  __host__ __device__
  auto eval() const
  RETURN_AUTO(detail::EvalHelper<is_static_dimseq_v<DimSeq<TENSOR_DIMS_USE>>::value>::template eval<TTensorCopier, TAllocator, TIndexStrategy>(*this))
};

} // end of ns detail

} // end of ns tensor
