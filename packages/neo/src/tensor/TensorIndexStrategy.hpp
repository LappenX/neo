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