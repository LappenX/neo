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

template <typename TThisType, typename TElementType, TENSOR_DIMS_DECLARE_NO_DEFAULT>
class Tensor : public ElementAccessFunctions<TThisType, TElementType, TensorTraits<TThisType>::RETURNS_REFERENCE>
{
public:
  static_assert(!is_tensor_v<TElementType>::value, "Tensors of tensors are not allowed");
  using ThisType = TThisType;
  // TODO: private member access for sub types, only allow size_t... there, same for dim methods?

  __host__ __device__
  size_t rows() const
  {
    return this->template dim<0>();
  }

  __host__ __device__
  size_t cols() const
  {
    return this->template dim<1>();
  }

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

  template <size_t TLength = non_trivial_dimensions_num_v<DimSeq<TENSOR_DIMS_USE>>::value>
  __host__ __device__
  VectorXs<TLength> dims() const
  {
    static_assert(TLength >= non_trivial_dimensions_num_v<DimSeq<TENSOR_DIMS_USE>>::value, "Non-trivial dimensions are cut off");
    return static_cast<const TThisType*>(this)->template dims_impl<TLength>();
  }
};

} // end of ns detail

} // end of ns tensor
