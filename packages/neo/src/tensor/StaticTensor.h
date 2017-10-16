#include "Tensor.h"

namespace tensor {


namespace detail {

template <size_t... TDims>
struct StaticDimAccessHelper;

template <size_t TFirst, size_t... TRest>
struct StaticDimAccessHelper<TFirst, TRest...>
{
  static size_t get(size_t index)
  {
    return index == 0 ? TFirst : StaticDimAccessHelper<TRest...>::get(index - 1);
  }
};

template <>
struct StaticDimAccessHelper<>
{
  static size_t get(size_t index)
  {
    ASSERT(false, "Index out of bounds");
    return 0;
  }
};

} // end of ns detail

template <typename TThisType, typename TElementType, size_t... TDims>
class StaticTensor : public Tensor<TThisType, TElementType, TDims...>
{
public:
  static_assert(is_static_dimseq_v<DimSeq<TDims...>>::value, "Dimensions not static");
  using SuperType = Tensor<TThisType, TElementType, TDims...>;

  template <typename... TDimensionArgs>
  __host__ __device__
  StaticTensor()
  {
  }

  template <typename TVectorType, size_t TVectorLength>
  __host__ __device__
  StaticTensor(const Vector<TVectorType, size_t, TVectorLength>& dims_d)
  {
    ASSERT(areSameDimensions(dims_d, VectorXs<sizeof...(TDims)>(TDims...)), "Invalid dynamic dimensions of static tensor");
  }

  template <typename... TDimensionArgs, ENABLE_IF_ARE_SIZE_T(TDimensionArgs...)>
  __host__ __device__
  StaticTensor(size_t first_dim, TDimensionArgs... rest_dims)
  {
    ASSERT(areSameDimensions(VectorXs<sizeof...(TDimensionArgs) + 1>(first_dim, rest_dims...), VectorXs<sizeof...(TDims)>(TDims...)), "Invalid dynamic dimensions of static tensor");
  }

  template <size_t TIndex>
  __host__ __device__
  constexpr size_t dim_impl() const
  {
    return nth_dimension_v<TIndex, DimSeq<TDims...>>::value;
  }

  __host__ __device__
  size_t dim_impl(size_t index) const
  {
    return math::lt(index, non_trivial_dimensions_num_v<tensor_dimseq_t<TThisType>>::value) ?
          detail::StaticDimAccessHelper<TDims...>::get(index)
        : 1;
  }

  TENSOR_DIMS_IMPL_FROM_IND(dims_impl)
};



template <typename TStorageType, typename TElementType, typename TIndexStrategy, size_t... TDims>
class DenseStaticStorageTensor : public DenseStorageTensor<TStorageType, TIndexStrategy,
                                                    DenseStaticStorageTensor<TStorageType, TElementType, TIndexStrategy, TDims...>,
                                                    StaticTensor<DenseStaticStorageTensor<TStorageType, TElementType, TIndexStrategy, TDims...>, TElementType, TDims...>>
{
public:
  static_assert(math::multiply(TDims...) == TStorageType::SIZE, "Invalid storage size");
  static_assert(std::is_same<TElementType, typename TStorageType::ElementType>::value, "Invalid storage type");
  using SuperType = DenseStorageTensor<TStorageType, TIndexStrategy,
                      DenseStaticStorageTensor<TStorageType, TElementType, TIndexStrategy, TDims...>,
                      StaticTensor<DenseStaticStorageTensor<TStorageType, TElementType, TIndexStrategy, TDims...>, TElementType, TDims...>>;
  using ThisType = DenseStaticStorageTensor<TStorageType, TElementType, TIndexStrategy, TDims...>;
  using IndexStrategy = TIndexStrategy;

  __host__ __device__
  DenseStaticStorageTensor()
    : m_storage()
  {
  }

  template <typename... TStorageArgs, ENABLE_IF(sizeof...(TStorageArgs) == math::multiply(TDims...)
                                                && std::is_constructible<TStorageType, TStorageArgs...>::value)>
  __host__ __device__
  DenseStaticStorageTensor(TStorageArgs&&... args)
    : m_storage(util::forward<TStorageArgs>(args)...)
  {
  }

  template <typename TTensorType2, typename TElementType2, size_t... TDims2, ENABLE_IF(are_compatible_dimseqs_v<DimSeq<TDims...>, DimSeq<TDims2...>>::value)>
  __host__ __device__
  DenseStaticStorageTensor(const Tensor<TTensorType2, TElementType2, TDims2...>& other)
    : m_storage()
  {
    static_assert(are_compatible_dimseqs_v<DimSeq<TDims...>, DimSeq<TDims2...>>::value, "Incompatible dimensions");
    ASSERT(areSameDimensions(this->dims(), other.dims()), "Inconsistent dimensions: " << this->dims() << " and " << other.dims() << "\nOther: " << other);
    *this = other;
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
    return m_storage[TIndexStrategy::template toIndex<TDims...>(util::forward<TCoordArgTypes>(coords)...)];
  }

  template <typename... TCoordArgTypes>
  __host__ __device__
  const TElementType& get_element_impl(TCoordArgTypes&&... coords) const
  {
    return m_storage[TIndexStrategy::template toIndex<TDims...>(util::forward<TCoordArgTypes>(coords)...)];
  }

private:
  TStorageType m_storage;
};

template <typename TStorageType, typename TElementType, typename TIndexStrategy, size_t... TDims>
struct TensorTraits<DenseStaticStorageTensor<TStorageType, TElementType, TIndexStrategy, TDims...>>
{
  static const bool RETURNS_REFERENCE = true;
  static const mem::MemoryType MEMORY_TYPE = TStorageType::MEMORY_TYPE;
};

} // end of ns tensor
