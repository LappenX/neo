#include "Tensor.h"

namespace tensor {

namespace detail {

template <size_t I>
struct StridedElementAccessHelper
{
  template <typename TStrideVector, typename... TCoordArgTypes>
  __host__ __device__
  static size_t index(TStrideVector&& stride, TCoordArgTypes&&... coords)
  {
    return stride(I) * getNthCoordinate<I>(util::forward<TCoordArgTypes>(coords)...)
      + StridedElementAccessHelper<I - 1>::index(stride, util::forward<TCoordArgTypes>(coords)...);
  }
};

template <>
struct StridedElementAccessHelper<0>
{
  template <typename TStrideVector, typename... TCoordArgTypes>
  __host__ __device__
  static size_t index(TStrideVector&& stride, TCoordArgTypes&&... coords)
  {
    return stride(0) * getNthCoordinate<0>(util::forward<TCoordArgTypes>(coords)...);
  }
};

} // end of ns detail





template <typename TStorageType, typename TElementType, size_t... TDims>
class StridedStorageTensor : public StaticOrDynamicTensor<StridedStorageTensor<TStorageType, TElementType, TDims...>, TElementType, true, TDims...>
{
public:
  using ThisType = StridedStorageTensor<TStorageType, TElementType, TDims...>;
  using SuperType = StaticOrDynamicTensor<StridedStorageTensor<TStorageType, TElementType, TDims...>, TElementType, true, TDims...>;

  __host__ __device__
  StridedStorageTensor()
  {
    // TODO: assert that all elements accessible via stride_vector are in bounds of other_storage
    // TODO: Can anything be asserted with strides at all?
  }

  template <typename TStrideVector, typename... TDimensionArgs, ENABLE_IF_SUPERTENSOR_CONSTRUCTIBLE(TDimensionArgs...)>
  __host__ __device__
  StridedStorageTensor(TStrideVector&& stride_vector, TDimensionArgs&&... args)
    : SuperType(util::forward<TDimensionArgs>(args)...)
    , m_storage(TStorageType::makeFromSize(dimensionProduct(util::forward<TDimensionArgs>(args)...)))
    , m_stride_vector(stride_vector)
  {
    // TODO: assert that all elements accessible via stride_vector are in bounds of other_storage
    // TODO: Can anything be asserted with strides at all?
  }

  template <typename TStrideVector, typename TStorageType2, typename... TDimensionArgs, ENABLE_IF_SUPERTENSOR_CONSTRUCTIBLE(TDimensionArgs...)>
  __host__ __device__
  StridedStorageTensor(TStrideVector&& stride_vector, TStorageType2&& other_storage, TDimensionArgs&&... args)
    : SuperType(util::forward<TDimensionArgs>(args)...)
    , m_storage(util::forward<TStorageType2>(other_storage))
    , m_stride_vector(stride_vector)
  {
    // TODO: assert that all elements accessible via stride_vector are in bounds of other_storage 
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
    return m_storage[detail::StridedElementAccessHelper<non_trivial_dimensions_num_v<DimSeq<TDims...>>::value - 1>
      ::index(m_stride_vector, util::forward<TCoordArgTypes>(coords)...)];
  }

  template <typename... TCoordArgTypes>
  __host__ __device__
  const TElementType& get_element_impl(TCoordArgTypes&&... coords) const
  {
    return m_storage[detail::StridedElementAccessHelper<non_trivial_dimensions_num_v<DimSeq<TDims...>>::value - 1>
      ::index(m_stride_vector, util::forward<TCoordArgTypes>(coords)...)];
  }

private:
  TStorageType m_storage;
  VectorXs<non_trivial_dimensions_num_v<DimSeq<TDims...>>::value> m_stride_vector;
};

template <typename TStorageType, typename TElementType, size_t... TDims>
struct TensorTraits<StridedStorageTensor<TStorageType, TElementType, TDims...>>
{
  static const bool RETURNS_REFERENCE = true;
  static const mem::MemoryType MEMORY_TYPE = TStorageType::MEMORY_TYPE;
};

} // end of ns tensor