#include "Tensor.h"

namespace tensor {

namespace detail {

template <bool TInDynRange>
struct DynamicTensorDimChecker
{
  template <size_t I, size_t TDim0, size_t... TDims, typename TVectorType2, typename TElementType2, size_t TVectorLength2>
  __host__ __device__
  static bool check(const Vector<TVectorType2, TElementType2, TVectorLength2>& dims)
  {
    static_assert(TVectorLength2 != DYN, "Dimension vector must be statically sized");
    return (TDim0 == DYN || TDim0 == dims(I))
      && DynamicTensorDimChecker<math::lt(I + 1, TVectorLength2)>::template check<I + 1, TDims...>(dims);
  }

  template <size_t I, typename TVectorType2, typename TElementType2, size_t TVectorLength2>
  __host__ __device__
  static bool check(const Vector<TVectorType2, TElementType2, TVectorLength2>& dims)
  {
    static_assert(TVectorLength2 != DYN, "Dimension vector must be statically sized");
    return (1 == dims(I))
      && DynamicTensorDimChecker<math::lt(I + 1, TVectorLength2)>::template check<I + 1>(dims);
  }
};

template <>
struct DynamicTensorDimChecker<false>
{
  template <size_t I, size_t TDim0, size_t... TDims, typename TVectorType2, typename TElementType2, size_t TVectorLength2>
  __host__ __device__
  static bool check(const Vector<TVectorType2, TElementType2, TVectorLength2>& dims)
  {
    static_assert(TVectorLength2 != DYN, "Dimension vector must be statically sized");
    return (TDim0 == DYN || TDim0 == 1)
      && DynamicTensorDimChecker<false>::template check<I + 1, TDims...>(dims);
  }

  template <size_t I, typename TVectorType2, typename TElementType2, size_t TVectorLength2>
  __host__ __device__
  static bool check(const Vector<TVectorType2, TElementType2, TVectorLength2>& dims)
  {
    static_assert(TVectorLength2 != DYN, "Dimension vector must be statically sized");
    return true;
  }
};

} // end of ns detail

template <typename TThisType, typename TElementType, size_t... TDims>
class DynamicTensor : public Tensor<TThisType, TElementType, TDims...>
{
public:
  static const size_t NON_TRIVIAL_DIMENSIONS_NUM = non_trivial_dimensions_num_v<DimSeq<TDims...>>::value;
  using SuperType = Tensor<TThisType, TElementType, TDims...>;

  template <typename... TDimensionArgs, ENABLE_IF_ARE_SIZE_T(TDimensionArgs...)>
  __host__ __device__
  DynamicTensor(TDimensionArgs&&... args)
  {
    ASSERT((detail::DynamicTensorDimChecker<math::lt(0, sizeof...(TDimensionArgs))>::template check<0, TDims...>(
        VectorXs<sizeof...(TDimensionArgs)>(util::forward<TDimensionArgs>(args)...)
      )), "Dynamic dimensions do not match static dimensions");
  }

  template <typename TVectorType2, typename TElementType2, size_t TVectorLength2>
  __host__ __device__
  DynamicTensor(const Vector<TVectorType2, TElementType2, TVectorLength2>& dims)
  {
    ASSERT((detail::DynamicTensorDimChecker<math::lt(0, TVectorLength2)>::template check<0, TDims...>(dims)),
      "Dynamic dimensions do not match static dimensions");
  }

  template <size_t TIndex>
  __host__ __device__
  size_t dim_impl() const
  {
    return static_cast<const TThisType*>(this)->template dyn_dim_impl<TIndex>();
  }

  __host__ __device__
  size_t dim_impl(size_t index) const
  {
    return static_cast<const TThisType*>(this)->dyn_dim_impl(index);
  }

  template <size_t TLength = non_trivial_dimensions_num_v<DimSeq<TDims...>>::value>
  __host__ __device__
  VectorXs<TLength> dims_impl() const
  {
    return static_cast<const TThisType*>(this)->template dyn_dims_impl<TLength>();
  }
};

template <typename TThisType, typename TElementType, size_t... TDims>
class DynamicTensorStoreDimensions : public DynamicTensor<TThisType, TElementType, TDims...>
{
public:
  using DimSeq = tensor::DimSeq<TDims...>;
  using SuperType = DynamicTensor<TThisType, TElementType, TDims...>;

  template <typename... TDimensionArgs>
  __host__ __device__
  DynamicTensorStoreDimensions(TDimensionArgs&&... args)
    : SuperType(util::forward<TDimensionArgs>(args)...)
  {
    tensor::copyDims(m_dims, util::forward<TDimensionArgs>(args)...);
  }

  template <size_t TIndex>
  __host__ __device__
  size_t dyn_dim_impl() const
  {
    return getNthDimension<TIndex>(m_dims);
  }

  __host__ __device__
  size_t dyn_dim_impl(size_t index) const
  {
    return math::lt(index, non_trivial_dimensions_num_v<DimSeq>::value) ? m_dims(index) : 1;
  }

  TENSOR_DIMS_IMPL_FROM_IND(dyn_dims_impl)

private:
  VectorXs<non_trivial_dimensions_num_v<DimSeq>::value> m_dims;
};





template <typename TStorageType, typename TElementType, typename TIndexStrategy, size_t... TDims>
class DenseDynamicStorageTensor : public DynamicTensorStoreDimensions<DenseDynamicStorageTensor<TStorageType, TElementType, TIndexStrategy, TDims...>, TElementType, TDims...>
{
public:
  static_assert(std::is_same<TElementType, typename TStorageType::ElementType>::value, "Invalid storage type");
  static_assert(TStorageType::HAS_DYN_SIZE_CONSTRUCTOR, "Invalid storage type");

  using ThisType = DenseDynamicStorageTensor<TStorageType, TElementType, TIndexStrategy, TDims...>;
  using SuperType = DynamicTensorStoreDimensions<ThisType, TElementType, TDims...>;
  using IndexStrategy = TIndexStrategy;

  template <typename... TDimensionArgs, ENABLE_IF_SUPERTENSOR_CONSTRUCTIBLE(TDimensionArgs...)>
  __host__ __device__
  DenseDynamicStorageTensor(TDimensionArgs&&... args)
    : SuperType(util::forward<TDimensionArgs>(args)...)
    , m_storage(dimensionProduct(util::forward<TDimensionArgs>(args)...))
  {
  }
  
  __host__ __device__
  TStorageType& storage()
  {
    return m_storage;
  }

  __host__ __device__
  const TStorageType& storage() const
  {
    return m_storage;
  }

  TENSOR_ASSIGN

  template <typename... TCoordArgTypes>
  __host__ __device__
  TElementType& get_element_impl(TCoordArgTypes&&... coords)
  {
    return m_storage[TIndexStrategy::template toIndex(this->template dims<ThisType::NON_TRIVIAL_DIMENSIONS_NUM>(), util::forward<TCoordArgTypes>(coords)...)];
  }

  template <typename... TCoordArgTypes>
  __host__ __device__
  const TElementType& get_element_impl(TCoordArgTypes&&... coords) const
  {
    return m_storage[TIndexStrategy::template toIndex(this->template dims<ThisType::NON_TRIVIAL_DIMENSIONS_NUM>(), util::forward<TCoordArgTypes>(coords)...)];
  }

private:
  TStorageType m_storage;
};

template <typename TStorageType, typename TElementType, typename TIndexStrategy, size_t... TDims>
struct TensorTraits<DenseDynamicStorageTensor<TStorageType, TElementType, TIndexStrategy, TDims...>>
{
  static const bool RETURNS_REFERENCE = true;
  static const mem::MemoryType MEMORY_TYPE = TStorageType::MEMORY_TYPE;
};

} // end of ns tensor
