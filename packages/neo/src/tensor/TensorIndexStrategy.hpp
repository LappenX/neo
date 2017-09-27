namespace detail {

struct ColMajorIndexStrategyHelper1
{
  template <size_t TDim0, size_t... TDims, typename... TRestCoords>
  __host__ __device__
  static size_t toIndex(size_t coord0, TRestCoords&&... coords)
  {
    return coord0 + TDim0 * toIndex<TDims...>(util::forward<TRestCoords>(coords)...);
  }

  template <size_t TDim0, size_t... TDims>
  __host__ __device__
  static size_t toIndex(size_t coord0)
  {
    return coord0;
  }

  template <typename... TRestCoords>
  __host__ __device__
  static size_t toIndex(size_t coord0, TRestCoords&&... coords)
  {
    ASSERT(coord0 == 0, "Coordinate out of bounds");
    return toIndex(util::forward<TRestCoords>(coords)...);
  }

  template <size_t... TDims>
  __host__ __device__
  static size_t toIndex()
  {
    return 0;
  }
};

template <bool TInDimRange>
struct ColMajorIndexStrategyHelper2
{
  template <size_t I = 0, typename TVectorType, typename TElementType, size_t TRank, typename... TRestCoords>
  __host__ __device__
  static size_t toIndex(const Vector<TVectorType, TElementType, TRank>& dims, size_t coord0, TRestCoords&&... coords)
  {
    return coord0 + dims(I) * ColMajorIndexStrategyHelper2<math::lt(I + 1, TRank)>::template toIndex<I + 1>(dims, util::forward<TRestCoords>(coords)...);
  }

  template <size_t I = 0, typename TVectorType, typename TElementType, size_t TRank>
  __host__ __device__
  static size_t toIndex(const Vector<TVectorType, TElementType, TRank>& dims, size_t coord0)
  {
    return coord0;
  }

  template <size_t I = 0, typename TVectorType, typename TElementType, size_t TRank>
  __host__ __device__
  static size_t toIndex(const Vector<TVectorType, TElementType, TRank>& dims)
  {
    return 0;
  }
};

template <>
struct ColMajorIndexStrategyHelper2<false>
{
  template <size_t I = 0, typename TVectorType, typename TElementType, size_t TRank, typename... TRestCoords>
  __host__ __device__
  static size_t toIndex(const Vector<TVectorType, TElementType, TRank>& dims, size_t coord0, TRestCoords&&... coords)
  {
    ASSERT(coord0 == 0, "Coordinate out of bounds");
    return ColMajorIndexStrategyHelper2<false>::toIndex(dims, util::forward<TRestCoords>(coords)...);
  }

  template <size_t I = 0, typename TVectorType, typename TElementType, size_t TRank>
  __host__ __device__
  static size_t toIndex(const Vector<TVectorType, TElementType, TRank>& dims)
  {
    return 0;
  }
};

} // end of ns detail



template <size_t... TDims, typename... TCoords, typename>
__host__ __device__
size_t ColMajorIndexStrategy::toIndex(TCoords&&... coords)
{
  return detail::ColMajorIndexStrategyHelper1::toIndex<TDims...>(util::forward<TCoords>(coords)...);
}

template <typename TVectorType, typename TElementType, size_t TRank, typename... TCoords>
__host__ __device__
size_t ColMajorIndexStrategy::toIndex(const Vector<TVectorType, TElementType, TRank>& dims, TCoords&&... coords)
{
  static_assert(TRank != DYN, "Must be static vector!");
  return detail::ColMajorIndexStrategyHelper2<math::lt(0, TRank)>::toIndex(dims, util::forward<TCoords>(coords)...);
}
















namespace detail {

struct RowMajorIndexStrategyHelper1
{
  template <size_t TDim0, size_t TDim1, size_t... TDims, typename... TRestCoords>
  __host__ __device__
  static size_t toIndex(size_t coord0, size_t coord1, TRestCoords&&... coords)
  {
    return toIndex<TDim0 * TDim1, TDims...>(coord0 * TDim1 + coord1, util::forward<TRestCoords>(coords)...);
  }

  template <size_t TDim0, size_t TDim1, size_t... TDims>
  __host__ __device__
  static size_t toIndex(size_t coord0)
  {
    return toIndex<TDim0 * TDim1, TDims...>(coord0 * TDim1);
  }

  template <size_t TDim0, size_t TDim1, size_t... TDims>
  __host__ __device__
  static size_t toIndex()
  {
    return 0;
  }



  template <size_t TDim0, typename... TRestCoords>
  __host__ __device__
  static size_t toIndex(size_t coord0, size_t coord1, TRestCoords&&... coords)
  {
    ASSERT(coord1 == 0, "Coordinate out of bounds");
    return toIndex<TDim0>(coord0, util::forward<TRestCoords>(coords)...);
  }

  template <size_t TDim0>
  __host__ __device__
  static size_t toIndex(size_t coord0)
  {
    return coord0;
  }



  template <typename... TRestCoords>
  __host__ __device__
  static size_t toIndex(size_t coord0, size_t coord1, TRestCoords&&... coords)
  {
    ASSERT(coord0 == 0, "Coordinate out of bounds");
    return toIndex(coord1, util::forward<TRestCoords>(coords)...);
  }

  __host__ __device__
  static size_t toIndex(size_t coord0)
  {
    ASSERT(coord0 == 0, "Coordinate out of bounds");
    return 0;
  }
};

template <bool TInDimRange, bool TInDimRangePlusOne>
struct RowMajorIndexStrategyHelper2
{
  template <size_t I = 0, typename TVectorType, typename TElementType, size_t TRank, typename... TRestCoords>
  __host__ __device__
  static size_t toIndex(const Vector<TVectorType, TElementType, TRank>& dims, size_t coord0, size_t coord1, TRestCoords&&... coords)
  {
    return RowMajorIndexStrategyHelper2<math::lt(I + 1, TRank), math::lt(I + 2, TRank)>::template toIndex<I + 1>
      (dims, coord0 * dims(I + 1) + coord1, util::forward<TRestCoords>(coords)...);
  }

  template <size_t I = 0, typename TVectorType, typename TElementType, size_t TRank>
  __host__ __device__
  static size_t toIndex(const Vector<TVectorType, TElementType, TRank>& dims, size_t coord0)
  {
    return coord0 * dims(I + 1);
  }

  template <size_t I = 0, typename TVectorType, typename TElementType, size_t TRank>
  __host__ __device__
  static size_t toIndex(const Vector<TVectorType, TElementType, TRank>& dims)
  {
    return 0;
  }
};

template <>
struct RowMajorIndexStrategyHelper2<true, false>
{
  template <size_t I = 0, typename TVectorType, typename TElementType, size_t TRank, typename... TRestCoords>
  __host__ __device__
  static size_t toIndex(const Vector<TVectorType, TElementType, TRank>& dims, size_t coord0, size_t coord1, TRestCoords&&... coords)
  {
    ASSERT(coord1 == 0, "Coordinate out of bounds");
    return toIndex<I>(dims, coord0, util::forward<TRestCoords>(coords)...);
  }

  template <size_t I = 0, typename TVectorType, typename TElementType, size_t TRank>
  __host__ __device__
  static size_t toIndex(const Vector<TVectorType, TElementType, TRank>& dims, size_t coord0)
  {
    return coord0;
  }

  template <size_t I = 0, typename TVectorType, typename TElementType, size_t TRank>
  __host__ __device__
  static size_t toIndex(const Vector<TVectorType, TElementType, TRank>& dims)
  {
    return 0;
  }
};

template <>
struct RowMajorIndexStrategyHelper2<false, false>
{
  template <size_t I = 0, typename TVectorType, typename TElementType, size_t TRank, typename... TRestCoords>
  __host__ __device__
  static size_t toIndex(const Vector<TVectorType, TElementType, TRank>& dims, size_t coord0, size_t coord1, TRestCoords&&... coords)
  {
    ASSERT(coord1 == 0, "Coordinate out of bounds");
    return toIndex<I>(dims, coord1, util::forward<TRestCoords>(coords)...);
  }

  template <size_t I = 0, typename TVectorType, typename TElementType, size_t TRank>
  __host__ __device__
  static size_t toIndex(const Vector<TVectorType, TElementType, TRank>& dims, size_t coord0)
  {
    ASSERT(coord0 == 0, "Coordinate out of bounds");
    return 0;
  }

  template <size_t I = 0, typename TVectorType, typename TElementType, size_t TRank>
  __host__ __device__
  static size_t toIndex(const Vector<TVectorType, TElementType, TRank>& dims)
  {
    return 0;
  }
};

} // end of ns detail

template <size_t... TDims, typename... TCoords, typename>
__host__ __device__
size_t RowMajorIndexStrategy::toIndex(TCoords&&... coords)
{
  return detail::RowMajorIndexStrategyHelper1::toIndex<TDims...>(util::forward<TCoords>(coords)...);
}

template <typename TVectorType, typename TElementType, size_t TRank, typename... TCoords>
__host__ __device__
size_t RowMajorIndexStrategy::toIndex(const Vector<TVectorType, TElementType, TRank>& dims, TCoords&&... coords)
{
  static_assert(TRank != DYN, "Must be static vector!");
  return detail::RowMajorIndexStrategyHelper2<math::lt(0, TRank), math::lt(1, TRank)>::toIndex(dims, util::forward<TCoords>(coords)...);
}