namespace detail {

template <typename TDimSeq>
struct DimsHelper;

template <size_t... TDims>
struct DimsHelper<DimSeq<TDims...>>
{
  static const size_t SIZE = DimSeq<TDims...>::SIZE;

  static constexpr size_t non_trivial_dimensions_num()
  {
    return non_trivial_dimensions_num(mem::LocalStorage<size_t, SIZE>(TDims...), SIZE);
  }

  static constexpr size_t non_trivial_dimensions_num(const mem::LocalStorage<size_t, SIZE> dims, size_t i)
  {
    return i == 0 ? 0
         : (dims.ptr()[i - 1] == 1 ? non_trivial_dimensions_num(dims, i - 1)
            : i);
  }

  static constexpr bool are_static_dims()
  {
    return are_static_dims(mem::LocalStorage<size_t, SIZE>(TDims...), 0);
  }

  static constexpr bool are_static_dims(const mem::LocalStorage<size_t, SIZE> dims, size_t i)
  {
    return i == SIZE ? true : dims.ptr()[i] != DYN && are_static_dims(dims, i + 1);
  }

  static constexpr size_t nth_dimension(size_t n)
  {
    return nth_dimension(mem::LocalStorage<size_t, SIZE>(TDims...), n);
  }

  static constexpr size_t nth_dimension(const mem::LocalStorage<size_t, SIZE> dims, size_t n)
  {
    return SIZE == 0 || n >= SIZE ? 1 : dims.ptr()[n];
  }

  static constexpr size_t nth_coordinate(size_t n)
  {
    return nth_dimension(mem::LocalStorage<size_t, SIZE>(TDims...), n);
  }

  static constexpr size_t nth_coordinate(const mem::LocalStorage<size_t, SIZE> dims, size_t n)
  {
    return SIZE == 0 ||  n >= SIZE ? 0 : dims.ptr()[n];
  }
};

static_assert(DimsHelper<DimSeq<1, 2, 1>>::non_trivial_dimensions_num() == 2, "non_trivial_dimensions_num not working");
static_assert(DimsHelper<DimSeq<4, 2, 6>>::nth_dimension(0) == 4, "nth_dimension not working");
static_assert(DimsHelper<DimSeq<4, 2, 6>>::nth_dimension(1) == 2, "nth_dimension not working");
static_assert(DimsHelper<DimSeq<4, 2, 6>>::nth_dimension(2) == 6, "nth_dimension not working");
static_assert(DimsHelper<DimSeq<4, 2, 6>>::nth_dimension(3) == 1, "nth_dimension not working");

template <size_t N, typename TDimSeq>
TVALUE(size_t, nth_dimension_v, DimsHelper<TDimSeq>::nth_dimension(N))

template <typename TDimSeq>
TVALUE(size_t, non_trivial_dimensions_num_v, detail::DimsHelper<TDimSeq>::non_trivial_dimensions_num());

template <typename TDimSeq>
TVALUE(size_t, is_static_dimseq_v, detail::DimsHelper<TDimSeq>::are_static_dims());








template <size_t N, typename TDimSeq1, typename TDimSeq2>
struct AreCompatibleDimSeqs2
{
  static const bool value = (nth_dimension_v<N - 1, TDimSeq1>::value == DYN || nth_dimension_v<N - 1, TDimSeq2>::value == DYN
                        || nth_dimension_v<N - 1, TDimSeq1>::value == nth_dimension_v<N - 1, TDimSeq2>::value)
                        && AreCompatibleDimSeqs2<N - 1, TDimSeq1, TDimSeq2>::value;
};

template <typename TDimSeq1, typename TDimSeq2>
struct AreCompatibleDimSeqs2<0, TDimSeq1, TDimSeq2>
{
  static const bool value = true;
};

template <typename... TDimSeqs>
struct AreCompatibleDimseqs;

template <typename TDimSeq1, typename TDimSeq2, typename... TDimSeqRest>
struct AreCompatibleDimseqs<TDimSeq1, TDimSeq2, TDimSeqRest...>
{
  static const bool value = AreCompatibleDimSeqs2<math::max(non_trivial_dimensions_num_v<TDimSeq1>::value, non_trivial_dimensions_num_v<TDimSeq2>::value), TDimSeq1, TDimSeq2>::value
                         && AreCompatibleDimseqs<TDimSeq1, TDimSeqRest...>::value
                         && AreCompatibleDimseqs<TDimSeq2, TDimSeqRest...>::value;
};

template <typename TDimSeq1>
struct AreCompatibleDimseqs<TDimSeq1>
{
  static const bool value = true;
};

template <>
struct AreCompatibleDimseqs<>
{
  static const bool value = true;
};

static_assert(AreCompatibleDimseqs<DimSeq<1, 2>, DimSeq<1, 2>>::value, "are_compatible_dimseqs_v not working");
static_assert(AreCompatibleDimseqs<DimSeq<1, 2, 1>, DimSeq<1, 2>>::value, "are_compatible_dimseqs_v not working");
static_assert(!AreCompatibleDimseqs<DimSeq<1, 3>, DimSeq<1>>::value, "are_compatible_dimseqs_v not working");
static_assert(AreCompatibleDimseqs<DimSeq<>, DimSeq<1, 1, 1>>::value, "are_compatible_dimseqs_v not working");
static_assert(!AreCompatibleDimseqs<DimSeq<0>, DimSeq<>>::value, "are_compatible_dimseqs_v not working");






template <typename TDimSeq>
struct IsDimSeq
{
  static const bool value = false;
};

template <size_t... TDims>
struct IsDimSeq<DimSeq<TDims...>>
{
  static const bool value = true;
};





template <bool TInRange, size_t TRank, size_t TDefault>
struct NthValueHelper;

template <size_t TRank, size_t TDefault>
struct NthValueHelper<true, TRank, TDefault>
{
  template <typename... TCoordArgTypes>
  __host__ __device__
  static size_t get(TCoordArgTypes&&... coords)
  {
    return tuple::nth_element<TRank>::get(coords...);
  }
};

template <size_t TRank, size_t TDefault>
struct NthValueHelper<false, TRank, TDefault>
{
  template <typename... TCoordArgTypes>
  __host__ __device__
  static size_t get(TCoordArgTypes&&... coords)
  {
    return TDefault;
  }
};

template <bool TInRange, size_t TRank, size_t TDefault>
struct NthValueOfVectorHelper;

template <size_t TRank, size_t TDefault>
struct NthValueOfVectorHelper<true, TRank, TDefault>
{
  template <typename TVectorType, size_t TVectorLength> 
  __host__ __device__
  static size_t get(const Vector<TVectorType, size_t, TVectorLength>& vector)
  {
    return vector(TRank);
  }
};

template <size_t TRank, size_t TDefault>
struct NthValueOfVectorHelper<false, TRank, TDefault>
{
  template <typename TVectorType, size_t TVectorLength> 
  __host__ __device__
  static size_t get(const Vector<TVectorType, size_t, TVectorLength>& vector)
  {
    return TDefault;
  }
};





template <size_t TVectorLength1, size_t TVectorLength2>
struct AreSameDimensionsHelper
{
  template <typename TVectorType1, typename TElementType1, typename TVectorType2, typename TElementType2>
  __host__ __device__
  static bool areSameDimensions(const Vector<TVectorType1, TElementType1, TVectorLength1>& dims1,
                           const Vector<TVectorType2, TElementType2, TVectorLength2>& dims2)
  {
    for (size_t i = 0; i < math::min(dims1.template dim<0>(), dims2.template dim<0>()); i++)
    {
      if (dims1(i) != dims2(i))
      {
        return false;
      }
    }
    if (dims1.template dim<0>() > dims2.template dim<0>())
    {
      for (size_t i = dims2.template dim<0>(); i < dims1.template dim<0>(); i++)
      {
        if (dims1(i) != 1)
        {
          return false;
        }
      }
    }
    else
    {
      for (size_t i = dims1.template dim<0>(); i < dims2.template dim<0>(); i++)
      {
        if (dims2(i) != 1)
        {
          return false;
        }
      }
    }
    return true;
  }
};

template <size_t TVectorLength1>
struct AreSameDimensionsHelper<TVectorLength1, 0>
{
  template <typename TVectorType1, typename TElementType1, typename TVectorType2, typename TElementType2>
  __host__ __device__
  static bool areSameDimensions(const Vector<TVectorType1, TElementType1, TVectorLength1>& dims1,
                           const Vector<TVectorType2, TElementType2, 0>& dims2)
  {
    for (size_t i = 0; i < dims1.template dim<0>(); i++)
    {
      if (dims1(i) != 1)
      {
        return false;
      }
    }
    return true;
  }
};

template <size_t TVectorLength2>
struct AreSameDimensionsHelper<0, TVectorLength2>
{
  template <typename TVectorType1, typename TElementType1, typename TVectorType2, typename TElementType2>
  __host__ __device__
  static bool areSameDimensions(const Vector<TVectorType1, TElementType1, 0>& dims1,
                           const Vector<TVectorType2, TElementType2, TVectorLength2>& dims2)
  {
    for (size_t i = 0; i < dims2.template dim<0>(); i++)
    {
      if (dims2(i) != 1)
      {
        return false;
      }
    }
    return true;
  }
};

template <>
struct AreSameDimensionsHelper<0, 0>
{
  template <typename TVectorType1, typename TElementType1, typename TVectorType2, typename TElementType2>
  __host__ __device__
  static bool areSameDimensions(const Vector<TVectorType1, TElementType1, 0>& dims1,
                           const Vector<TVectorType2, TElementType2, 0>& dims2)
  {
    return true;
  }
};

template <typename TVectorType1, typename TElementType1, size_t TVectorLength1, typename TVectorType2, typename TElementType2, size_t TVectorLength2, typename... TRestVectorTypes>
__host__ __device__
bool areSameDimensions(const Vector<TVectorType1, TElementType1, TVectorLength1>& dims1,
                         const Vector<TVectorType2, TElementType2, TVectorLength2>& dims2,
                         TRestVectorTypes&&... rest)
{
  return AreSameDimensionsHelper<TVectorLength1, TVectorLength2>::areSameDimensions(dims1, dims2) && areSameDimensions(dims1, util::forward<TRestVectorTypes>(rest)...);
}

template <typename TVectorType1, typename TElementType1, size_t TVectorLength1>
__host__ __device__
bool areSameDimensions(const Vector<TVectorType1, TElementType1, TVectorLength1>& dims1)
{
  return true;
}





template <size_t TRank, typename... TDimArgTypes, ENABLE_IF_ARE_SIZE_T(TDimArgTypes...)> 
__host__ __device__
size_t getNthDimension(TDimArgTypes&&... dims)
{
  return detail::NthValueHelper<math::lt(TRank, sizeof...(TDimArgTypes)), TRank, 1>::get(util::forward<TDimArgTypes>(dims)...);
}

template <size_t TRank, typename TDimVectorType, size_t TDimVectorLength> 
__host__ __device__
size_t getNthDimension(const Vector<TDimVectorType, size_t, TDimVectorLength>& dim_vector)
{
  static_assert(TDimVectorLength != DYN, "Dimension vector must have static size");
  return detail::NthValueOfVectorHelper<math::lt(TRank, TDimVectorLength), TRank, 1>::get(dim_vector);
}

template <size_t TRank, typename... TCoordArgTypes, ENABLE_IF_ARE_SIZE_T(TCoordArgTypes...)> 
__host__ __device__
size_t getNthCoordinate(TCoordArgTypes&&... coords)
{
  return detail::NthValueHelper<math::lt(TRank, sizeof...(TCoordArgTypes)), TRank, 0>::get(util::forward<TCoordArgTypes>(coords)...);
}

template <size_t TRank, typename TCoordVectorType, size_t TCoordVectorLength> 
__host__ __device__
size_t getNthCoordinate(const Vector<TCoordVectorType, size_t, TCoordVectorLength>& coord_vector)
{
  static_assert(TCoordVectorLength != DYN, "Coordinate vector must have static size");
  return detail::NthValueOfVectorHelper<math::lt(TRank, TCoordVectorLength), TRank, 0>::get(coord_vector);
}

template <typename... TCoordArgTypes, ENABLE_IF_ARE_SIZE_T(TCoordArgTypes...)> 
__host__ __device__
constexpr size_t getCoordinateNum()
{
  return sizeof...(TCoordArgTypes);
}

template <typename TCoordVectorType, typename... TDummies, ENABLE_IF(is_tensor_v<tensor_clean_t<TCoordVectorType>>::value)> 
__host__ __device__
constexpr size_t getCoordinateNum()
{
  static_assert(is_static_dimseq_v<tensor_dimseq_t<TCoordVectorType>>::value, "Coordinate vector must have static dimensions");
  return nth_dimension_v<0, tensor_dimseq_t<TCoordVectorType>>::value;
}

} // end of ns detail





template <size_t TRank, typename... TDimArgTypes> 
__host__ __device__
size_t getNthDimension(TDimArgTypes&&... dims)
{
  return detail::getNthDimension<TRank>(util::forward<TDimArgTypes>(dims)...);
}

template <size_t TRank, typename... TCoordArgTypes> 
__host__ __device__
size_t getNthCoordinate(TCoordArgTypes&&... coords)
{
  return detail::getNthCoordinate<TRank>(util::forward<TCoordArgTypes>(coords)...);
}

template <typename... TCoordArgTypes> 
__host__ __device__
constexpr size_t getCoordinateNum()
{
  return detail::getCoordinateNum<TCoordArgTypes...>();
}





template <typename... TVectorTypes>
__host__ __device__
bool areSameDimensions(TVectorTypes&&... dims)
{
  return detail::areSameDimensions(util::forward<TVectorTypes>(dims)...);
}

template <typename TVectorType, size_t TVectorLength>
__host__ __device__
size_t getNonTrivialDimensionsNum(const Vector<TVectorType, size_t, TVectorLength>& dims)
{
  static_assert(TVectorLength != DYN, "Dimension vector must have static size");
  size_t i = TVectorLength;
  while (i > 0 && dims(i - 1) == 1)
  {
    i--;
  }
  return i;
}





namespace detail {

template <bool TDestInRange, bool TSrcInRange>
struct CopyDimHelper;

template <size_t I = 0, typename TVectorType1, typename... TSrcDimArgs>
__host__ __device__
void copyDims(TVectorType1&& dest, TSrcDimArgs&&... src)
{
  const size_t DEST_DIM_NUM = nth_dimension_v<0, tensor_dimseq_t<tensor_clean_t<TVectorType1>>>::value;
  const size_t SRC_DIM_NUM = tensor::getCoordinateNum<TSrcDimArgs...>();

  const bool DEST_IN_RANGE = I < DEST_DIM_NUM;
  const bool SRC_IN_RANGE = I < SRC_DIM_NUM;
  
  detail::CopyDimHelper<DEST_IN_RANGE, SRC_IN_RANGE>::template copy<I>(dest, util::forward<TSrcDimArgs>(src)...);
}

template <bool TDestInRange, bool TSrcInRange>
struct CopyDimHelper
{
  template <size_t I, typename TVectorType1, typename... TSrcDimArgs>
  __host__ __device__
  static void copy(TVectorType1&& dest, TSrcDimArgs&&... src)
  {
    dest(I) = getNthDimension<I>(util::forward<TSrcDimArgs>(src)...);
    detail::copyDims<I + 1>(dest, util::forward<TSrcDimArgs>(src)...);
  }
};

template <>
struct CopyDimHelper<true, false>
{
  template <size_t I, typename TVectorType1, typename... TSrcDimArgs>
  __host__ __device__
  static void copy(TVectorType1&& dest, TSrcDimArgs&&... src)
  {
    dest(I) = 0;
    detail::copyDims<I + 1>(dest, util::forward<TSrcDimArgs>(src)...);
  }
};

template <>
struct CopyDimHelper<false, true>
{
  template <size_t I, typename TVectorType1, typename... TSrcDimArgs>
  __host__ __device__
  static void copy(TVectorType1&& dest, TSrcDimArgs&&... src)
  {
    ASSERT(getNthDimension<I>(util::forward<TSrcDimArgs>(src)...) == 1, "Invalid dimension, should be trivial");
    detail::copyDims<I + 1>(dest, util::forward<TSrcDimArgs>(src)...);
  }
};

template <>
struct CopyDimHelper<false, false>
{
  template <size_t I, typename TVectorType1, typename... TSrcDimArgs>
  __host__ __device__
  static void copy(TVectorType1&& dest, TSrcDimArgs&&... src)
  {
  }
};

} // end of ns detail

template <typename TVectorType1, typename... TSrcDimArgs>
__host__ __device__
void copyDims(TVectorType1&& dest, TSrcDimArgs&&... src)
{
  detail::copyDims(dest, util::forward<TSrcDimArgs>(src)...);
}





namespace detail {

template <typename... TTensorTypes>
struct GetStaticDimSeqFromTensors;

template <typename TTensorType0, typename TTensorType1, typename... TTensorTypes>
struct GetStaticDimSeqFromTensors<TTensorType0, TTensorType1, TTensorTypes...>
{
  using type = typename std::conditional<is_static_dimseq_v<tensor_dimseq_t<TTensorType0>>::value, tensor_dimseq_t<TTensorType0>, typename GetStaticDimSeqFromTensors<TTensorType1, TTensorTypes...>::type>::type;
};

template <typename TTensorType0>
struct GetStaticDimSeqFromTensors<TTensorType0>
{
  using type = tensor_dimseq_t<TTensorType0>;
};

} // end of ns detail





namespace detail {

template <size_t I>
struct CoordsAreInRangeHelper
{
  template <typename TTensorType, typename... TCoordArgTypes>
  __host__ __device__
  static bool get(TTensorType&& tensor, TCoordArgTypes&&... coords)
  {
    return getNthCoordinate<I - 1>(util::forward<TCoordArgTypes>(coords)...) < tensor.template dim<I - 1>()
            && CoordsAreInRangeHelper<I - 1>::get(util::forward<TTensorType>(tensor), util::forward<TCoordArgTypes>(coords)...);
  }
};

template <>
struct CoordsAreInRangeHelper<0>
{
  template <typename TTensorType, typename... TCoordArgTypes>
  __host__ __device__
  static bool get(TTensorType&& tensor, TCoordArgTypes&&... coords)
  {
    return true;
  }
};

} // end of ns detail

template <typename TTensorType, typename... TCoordArgTypes>
__host__ __device__
bool coordsAreInRange(TTensorType&& tensor, TCoordArgTypes&&... coords)
{
  return detail::CoordsAreInRangeHelper<getCoordinateNum<TCoordArgTypes...>()>::get
            (util::forward<TTensorType>(tensor), util::forward<TCoordArgTypes>(coords)...);
}





namespace detail {

struct ProductHelper
{
  template <typename... TDimensionArgs, ENABLE_IF_ARE_SIZE_T(TDimensionArgs...)>
  __host__ __device__
  static size_t product(TDimensionArgs&&... dim_args)
  {
    return math::multiply(dim_args...);
  }

  template <typename TVectorType2, typename TElementType2, size_t TVectorLength2>
  __host__ __device__
  static size_t product(const Vector<TVectorType2, TElementType2, TVectorLength2>& dim_vec)
  {
    static_assert(TVectorLength2 != DYN, "Cannot have dynamic length dimension vector");
    return vectorProduct(dim_vec, tmp::value_sequence::ascending_numbers_t<TVectorLength2>());
  }

  template <typename TVectorType2, typename TElementType2, size_t TVectorLength2, size_t... TIndices>
  __host__ __device__
  static size_t vectorProduct(const Vector<TVectorType2, TElementType2, TVectorLength2>& dim_vec, tmp::value_sequence::Sequence<size_t, TIndices...>)
  {
    return math::multiply(dim_vec(TIndices)...);
  }
};

} // end of ns detail

template <typename... TDimensionArgs>
__host__ __device__
size_t dimensionProduct(TDimensionArgs&&... dim_args)
{
  return detail::ProductHelper::product(util::forward<TDimensionArgs>(dim_args)...);
}