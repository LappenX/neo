#include "Tensor.h"

namespace tensor {

template <typename TThisType, typename TElementType, size_t... TDims>
class DynamicTensor : public Tensor<TThisType, TElementType, TDims...>
{
public:
  static const size_t NON_TRIVIAL_DIMENSIONS_NUM = non_trivial_dimensions_num_v<DimSeq<TDims...>>::value;
  using SuperType = Tensor<TThisType, TElementType, TDims...>;

  // TODO: allow for arguments that dont construct VectorXs if they end in trivial dimensions or imply trivial dimensions
  template <typename... TDimensionArgs>
  __host__ __device__
  DynamicTensor(TDimensionArgs&&... args)
  {
  }

  template <typename TVectorType2, typename TElementType2, size_t TVectorLength2>
  __host__ __device__
  DynamicTensor(const Vector<TVectorType2, TElementType2, TVectorLength2>& dims)
  {
    // TODO: ASSERT dims satisfies TDims... (rest equals 1)
  }
};

template <typename TThisType, typename TElementType, size_t... TDims>
class DynamicTensorStoreDimensions : public DynamicTensor<TThisType, TElementType, TDims...>
{
public:
  static const size_t NON_TRIVIAL_DIMENSIONS_NUM = non_trivial_dimensions_num_v<DimSeq<TDims...>>::value;
  using SuperType = DynamicTensor<TThisType, TElementType, TDims...>;

  template <typename... TDimensionArgs>
  __host__ __device__
  DynamicTensorStoreDimensions(TDimensionArgs&&... args)
    : SuperType(util::forward<TDimensionArgs>(args)...)
  {
    copyDims(m_dims, util::forward<TDimensionArgs>(args)...);
  }

  template <size_t TIndex>
  __host__ __device__
  size_t dim() const
  {
    return getNthDimension<TIndex>(m_dims);
  }

  __host__ __device__
  size_t dim(size_t index) const
  {
    return math::lt(index, non_trivial_dimensions_num_v<DimSeq<TDims...>>::value) ? m_dims(index) : 1;
  }

  template <size_t TLength = non_trivial_dimensions_num_v<DimSeq<TDims...>>::value>
  __host__ __device__
  VectorXs<TLength> dims() const
  { // TODO: no new vector when length equals saved length
    ASSERT(math::gte(TLength, getNonTrivialDimensionsNum(m_dims)), "Non-trivial dimensions are cut off");
    return vectorSizeHelper<TLength>(tmp::value_sequence::ascending_numbers_t<TLength>());
  }

private:
  VectorXs<NON_TRIVIAL_DIMENSIONS_NUM> m_dims;

  template <size_t TLength, size_t... TIndices>
  __host__ __device__
  VectorXs<TLength> vectorSizeHelper(tmp::value_sequence::Sequence<size_t, TIndices...>) const
  {
    return VectorXs<TLength>(this->template dim<TIndices>()...);
  }
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

  template <typename... TDimensionArgs, ENABLE_IF_SUPERTENSOR_CONSTRUCTIBLE(TDimensionArgs...), ENABLE_IF_ARE_SIZE_T(TDimensionArgs...)>
  __host__ __device__
  DenseDynamicStorageTensor(TDimensionArgs&&... args)
    : SuperType(util::forward<TDimensionArgs>(args)...)
    , m_storage(math::multiply(args...))
  {
  }

  template <typename TVectorType2, typename TElementType2, size_t TVectorLength2,
    ENABLE_IF_SUPERTENSOR_CONSTRUCTIBLE(const Vector<TVectorType2, TElementType2, TVectorLength2>&)>
  __host__ __device__
  DenseDynamicStorageTensor(const Vector<TVectorType2, TElementType2, TVectorLength2>& dim_vec)
    : SuperType(dim_vec)
    , m_storage(vectorProduct(dim_vec, tmp::value_sequence::ascending_numbers_t<TVectorLength2>()))
  {
    static_assert(TVectorLength2 != DYN, "Cannot have dynamic length dimension vector");
  }
  
  template <typename... TCoordArgTypes>
  __host__ __device__
  TElementType& operator()(TCoordArgTypes&&... coords)
  {
    return m_storage[TIndexStrategy::template toIndex(this->template dims<ThisType::NON_TRIVIAL_DIMENSIONS_NUM>(), util::forward<TCoordArgTypes>(coords)...)];
  }

  template <typename... TCoordArgTypes>
  __host__ __device__
  const TElementType& operator()(TCoordArgTypes&&... coords) const
  {
    return m_storage[TIndexStrategy::template toIndex(this->template dims<ThisType::NON_TRIVIAL_DIMENSIONS_NUM>(), util::forward<TCoordArgTypes>(coords)...)];
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

private:
  TStorageType m_storage;

  template <typename TVectorType2, typename TElementType2, size_t TVectorLength2, size_t... TIndices>
  static size_t vectorProduct(const Vector<TVectorType2, TElementType2, TVectorLength2>& dim_vec, tmp::value_sequence::Sequence<size_t, TIndices...>)
  {
    return math::multiply(dim_vec(TIndices)...);
  }
};

template <typename TStorageType, typename TElementType, typename TIndexStrategy, size_t... TDims>
struct TensorTraits<DenseDynamicStorageTensor<TStorageType, TElementType, TIndexStrategy, TDims...>>
{
  static const bool RETURNS_REFERENCE = true;
  static const mem::MemoryType MEMORY_TYPE = TStorageType::MEMORY_TYPE;
};

} // end of ns tensor
